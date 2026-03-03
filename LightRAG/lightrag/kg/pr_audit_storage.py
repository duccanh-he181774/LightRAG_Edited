"""
PR Audit Storage Module

This module provides PostgreSQL-based storage for GitHub pull request audit records.
It's designed to work with Neon PostgreSQL (serverless Postgres) but is compatible
with any PostgreSQL instance.

The storage tracks:
- PR metadata (number, title, URL, author, state)
- Code diff content
- LLM-generated summary
- Matched requirements
- Full LLM conversation log
- Processing timestamps and status

Schema:
    gap_findings: Main table for audit records

Usage:
    storage = PRAuditStorage.from_url("postgresql://...")
    await storage.initialize()
    record_id = await storage.save_audit(audit_record)
    audit = await storage.get_audit(record_id)
    await storage.close()
"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import ssl

from dotenv import load_dotenv

import pipmaster as pm

if not pm.is_installed("asyncpg"):
    pm.install("asyncpg")

import asyncpg
from asyncpg import Pool

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)


@dataclass
class PRAuditRecord:
    """
    Data class representing a PR audit record.

    Matches the Neon database schema:
        - id: UUID PRIMARY KEY DEFAULT gen_random_uuid()
        - use_case_id: UUID (reference to use case)
        - title: VARCHAR(255) NOT NULL
        - commit_id: VARCHAR(100) (PR number as string)
        - description: TEXT (LLM summary + matched requirements)
        - severity: VARCHAR(50) - must be Low/Medium/High/Critical
        - created_at: TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        - updated_at: TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        - status: VARCHAR(20) - must be pending/approved/rejected
    """
    title: str
    commit_id: str = ""
    use_case_id: Optional[str] = None
    description: str = ""
    severity: str = "Medium"
    status: str = "pending"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for database insertion."""
        return {
            "id": self.id,
            "use_case_id": self.use_case_id,
            "title": self.title,
            "commit_id": self.commit_id,
            "description": self.description,
            "severity": self.severity,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "PRAuditRecord":
        """Create record from database row."""
        return cls(
            id=str(row["id"]),
            use_case_id=str(row["use_case_id"]
                            ) if row["use_case_id"] else None,
            title=row["title"] or "",
            commit_id=row["commit_id"] or "",
            description=row["description"] or "",
            severity=row["severity"] or "normal",
            status=row["status"] or "pending",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PRAuditStorage:
    """
    PostgreSQL storage for PR audit records.

    This class manages the connection pool and provides CRUD operations
    for PR audit records. It's designed to work with Neon PostgreSQL
    but is compatible with any PostgreSQL instance.

    Attributes:
        pool: asyncpg connection pool
        table_name: Name of the audit table (default: gap_findings)
    """

    # SQL for creating the audit table
    # Matches user's Neon database schema
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        use_case_id UUID,
        title VARCHAR(255) NOT NULL,
        commit_id VARCHAR(100),
        description TEXT,
        severity VARCHAR(50),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'pending'
    );

    -- Create indexes if they don't exist
    CREATE INDEX IF NOT EXISTS idx_{table_name}_use_case ON {table_name}(use_case_id);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_status ON {table_name}(status);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_created ON {table_name}(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_commit ON {table_name}(commit_id);
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        table_name: str = "gap_findings",
        ssl_mode: str = "require",
        max_connections: int = 5,
    ):
        """
        Initialize PR audit storage.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            user: Database user
            password: Database password
            database: Database name
            table_name: Name of the audit table
            ssl_mode: SSL mode (require, prefer, disable)
            max_connections: Maximum pool connections
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name
        self.ssl_mode = ssl_mode
        self.max_connections = max_connections
        self.pool: Optional[Pool] = None

    @classmethod
    def from_url(cls, database_url: str, table_name: str = "gap_findings") -> "PRAuditStorage":
        """
        Create storage instance from a database URL.

        Args:
            database_url: PostgreSQL connection URL
            table_name: Name of the audit table

        Returns:
            PRAuditStorage instance
        """
        # Parse URL: postgresql://user:password@host:port/database?sslmode=require
        import urllib.parse

        parsed = urllib.parse.urlparse(database_url)

        # Extract query parameters
        query_params = urllib.parse.parse_qs(parsed.query)
        ssl_mode = query_params.get("sslmode", ["require"])[0]

        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            user=parsed.username or "",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") if parsed.path else "",
            table_name=table_name,
            ssl_mode=ssl_mode,
        )

    @classmethod
    def from_env(cls, table_name: str = "gap_findings") -> "PRAuditStorage":
        """
        Create storage instance from environment variables.

        Environment variables:
            NEON_DATABASE_URL: Full connection URL (preferred)
            Or individual variables:
            NEON_HOST, NEON_PORT, NEON_USER, NEON_PASSWORD, NEON_DATABASE

        Args:
            table_name: Name of the audit table

        Returns:
            PRAuditStorage instance
        """
        url = os.getenv("NEON_DATABASE_URL")
        if url:
            return cls.from_url(url, table_name)

        return cls(
            host=os.getenv("NEON_HOST", "localhost"),
            port=int(os.getenv("NEON_PORT", "5432")),
            user=os.getenv("NEON_USER", ""),
            password=os.getenv("NEON_PASSWORD", ""),
            database=os.getenv("NEON_DATABASE", ""),
            table_name=table_name,
            ssl_mode=os.getenv("NEON_SSL_MODE", "require"),
        )

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections."""
        if self.ssl_mode == "disable":
            return None

        ctx = ssl.create_default_context()
        if self.ssl_mode == "require":
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def initialize(self) -> None:
        """
        Initialize the storage by creating the connection pool and table.
        """
        ssl_context = self._create_ssl_context()

        self.pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            ssl=ssl_context,
            min_size=1,
            max_size=self.max_connections,
        )

        # Create table if not exists
        async with self.pool.acquire() as conn:
            await conn.execute(
                self.CREATE_TABLE_SQL.format(table_name=self.table_name)
            )

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def save_audit(self, record: PRAuditRecord) -> str:
        """
        Save an audit record to the database.

        Args:
            record: The audit record to save

        Returns:
            str: The ID of the saved record
        """
        if not self.pool:
            raise RuntimeError(
                "Storage not initialized. Call initialize() first.")

        data = record.to_dict()

        sql = f"""
        INSERT INTO {self.table_name} (
            id, use_case_id, title, commit_id, description,
            severity, status, created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9
        )
        ON CONFLICT (id)
        DO UPDATE SET
            description = EXCLUDED.description,
            severity = EXCLUDED.severity,
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at
        RETURNING id
        """

        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                sql,
                uuid.UUID(data["id"]),
                uuid.UUID(data["use_case_id"]) if data["use_case_id"] else None,
                data["title"],
                data["commit_id"],
                data["description"],
                data["severity"],
                data["status"],
                data["created_at"],
                data["updated_at"],
            )
            return str(result)

    async def lookup_use_case_uuid(self, uc_id: str) -> Optional[str]:
        """
        Look up the UUID of a use case by its uc_id (e.g. 'UC-001').

        Args:
            uc_id: The use case identifier like 'UC-001'

        Returns:
            str: The UUID of the matching srs_use_cases row, or None if not found
        """
        if not self.pool:
            raise RuntimeError(
                "Storage not initialized. Call initialize() first.")

        sql = "SELECT id FROM srs_use_cases WHERE uc_id = $1 LIMIT 1"
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(sql, uc_id)
            return str(result) if result else None

    async def get_audit(self, audit_id: str) -> Optional[PRAuditRecord]:
        """
        Get an audit record by ID.

        Args:
            audit_id: The audit record ID

        Returns:
            PRAuditRecord or None if not found
        """
        if not self.pool:
            raise RuntimeError(
                "Storage not initialized. Call initialize() first.")

        sql = f"SELECT * FROM {self.table_name} WHERE id = $1"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, uuid.UUID(audit_id))
            if row:
                return PRAuditRecord.from_row(row)
            return None

    async def list_audits(
        self,
        use_case_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PRAuditRecord]:
        """
        List audit records with optional filters.

        Args:
            use_case_id: Filter by use case
            status: Filter by status
            limit: Maximum records to return
            offset: Offset for pagination

        Returns:
            List of audit records
        """
        if not self.pool:
            raise RuntimeError(
                "Storage not initialized. Call initialize() first.")

        conditions = []
        params = []
        param_idx = 1

        if use_case_id:
            conditions.append(f"use_case_id = ${param_idx}")
            params.append(uuid.UUID(use_case_id))
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        sql = f"""
        SELECT * FROM {self.table_name}
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [PRAuditRecord.from_row(row) for row in rows]

    async def get_audits_by_commit(
        self,
        commit_id: str,
        use_case_id: Optional[str] = None
    ) -> list[PRAuditRecord]:
        """
        Get all audit records for a specific commit/PR.

        Args:
            commit_id: Commit/PR ID (as string)
            use_case_id: Optional filter by use case

        Returns:
            List of audit records for the commit
        """
        if not self.pool:
            raise RuntimeError(
                "Storage not initialized. Call initialize() first.")

        if use_case_id:
            sql = f"""
            SELECT * FROM {self.table_name}
            WHERE commit_id = $1 AND use_case_id = $2
            ORDER BY created_at DESC
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, commit_id, uuid.UUID(use_case_id))
                return [PRAuditRecord.from_row(row) for row in rows]
        else:
            sql = f"""
            SELECT * FROM {self.table_name}
            WHERE commit_id = $1
            ORDER BY created_at DESC
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, commit_id)
                return [PRAuditRecord.from_row(row) for row in rows]

    async def delete_audit(self, audit_id: str) -> bool:
        """
        Delete an audit record.

        Args:
            audit_id: The audit record ID

        Returns:
            True if deleted, False if not found
        """
        if not self.pool:
            raise RuntimeError(
                "Storage not initialized. Call initialize() first.")

        sql = f"DELETE FROM {self.table_name} WHERE id = $1 RETURNING id"

        async with self.pool.acquire() as conn:
            result = await conn.fetchval(sql, uuid.UUID(audit_id))
            return result is not None

    async def get_stats(self, use_case_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get statistics about audit records.

        Args:
            use_case_id: Optional filter by use case

        Returns:
            dict with statistics (total, by_status, by_severity, etc.)
        """
        if not self.pool:
            raise RuntimeError(
                "Storage not initialized. Call initialize() first.")

        where_clause = "WHERE use_case_id = $1" if use_case_id else ""
        params = [uuid.UUID(use_case_id)] if use_case_id else []

        sql = f"""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'approved') as approved,
            COUNT(*) FILTER (WHERE status = 'pending') as pending,
            COUNT(*) FILTER (WHERE status = 'rejected') as rejected,
            COUNT(*) FILTER (WHERE severity = 'Critical') as critical,
            COUNT(*) FILTER (WHERE severity = 'High') as high,
            COUNT(*) FILTER (WHERE severity = 'Medium') as medium,
            COUNT(*) FILTER (WHERE severity = 'Low') as low
        FROM {self.table_name}
        {where_clause}
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, *params)
            return {
                "total": row["total"],
                "by_status": {
                    "approved": row["approved"],
                    "pending": row["pending"],
                    "rejected": row["rejected"],
                },
                "by_severity": {
                    "Critical": row["critical"],
                    "High": row["high"],
                    "Medium": row["medium"],
                    "Low": row["low"],
                },
            }
