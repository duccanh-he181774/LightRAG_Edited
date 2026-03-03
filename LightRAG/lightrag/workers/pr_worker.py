"""
Pull Request Processing Worker

This module contains Celery tasks for processing GitHub pull request webhooks.
It fetches PR diffs from GitHub, uses LightRAG to summarize code changes and
match them against indexed requirements, then stores the audit log in PostgreSQL.

Tasks:
    process_pull_request: Main task for processing a PR webhook payload
"""

import asyncio
import os
import logging
from typing import Any, Optional
import httpx

from celery import shared_task
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.kg.pr_audit_storage import PRAuditStorage, PRAuditRecord

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

logger = logging.getLogger(__name__)


class PRProcessorConfig:
    """Configuration for PR processor."""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.github_api_url = os.getenv(
            "GITHUB_API_URL", "https://api.github.com")

        # LightRAG configuration
        self.working_dir = os.getenv("WORKING_DIR", "./rag_storage")
        self.llm_binding = os.getenv("LLM_BINDING", "openai")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        self.llm_api_key = os.getenv("LLM_BINDING_API_KEY", "")
        self.llm_host = os.getenv(
            "LLM_BINDING_HOST", "https://api.openai.com/v1")

        # Embedding configuration (separate from LLM)
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
        self.embedding_api_key = os.getenv(
            "EMBEDDING_BINDING_API_KEY", "") or self.llm_api_key
        self.embedding_host = os.getenv(
            "EMBEDDING_BINDING_HOST", "") or self.llm_host
        self.embedding_token_limit = int(
            os.getenv("EMBEDDING_TOKEN_LIMIT", "8192"))

        # Storage configuration (must match main server)
        self.kv_storage = os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        self.vector_storage = os.getenv(
            "LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
        self.graph_storage = os.getenv(
            "LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
        self.doc_status_storage = os.getenv(
            "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")

        # Neon PostgreSQL for audit storage
        self.neon_database_url = os.getenv("NEON_DATABASE_URL", "")

        # Use case ID for linking audits to a use case
        self.use_case_id = os.getenv("PR_USE_CASE_ID", "")

        # Query configuration
        self.query_mode = os.getenv("PR_QUERY_MODE", "mix")
        self.query_top_k = int(os.getenv("PR_QUERY_TOP_K", "30"))

        # Diff size limits
        self.max_diff_chars = int(os.getenv("PR_MAX_DIFF_CHARS", "50000"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.github_token:
            errors.append("GITHUB_TOKEN is required")
        if not self.neon_database_url:
            errors.append("NEON_DATABASE_URL is required")
        if not self.llm_api_key:
            errors.append("LLM_BINDING_API_KEY is required")
        return errors


def get_config() -> PRProcessorConfig:
    """Get processor configuration."""
    return PRProcessorConfig()


async def fetch_pr_diff(
    owner: str,
    repo: str,
    pr_number: int,
    github_token: str,
    github_api_url: str = "https://api.github.com"
) -> str:
    """
    Fetch the diff of a pull request from GitHub API.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
        github_token: GitHub personal access token
        github_api_url: GitHub API base URL

    Returns:
        str: The PR diff as text
    """
    url = f"{github_api_url}/repos/{owner}/{repo}/pulls/{pr_number}"

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


async def fetch_pr_files(
    owner: str,
    repo: str,
    pr_number: int,
    github_token: str,
    github_api_url: str = "https://api.github.com"
) -> list[dict]:
    """
    Fetch the list of files changed in a pull request.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
        github_token: GitHub personal access token
        github_api_url: GitHub API base URL

    Returns:
        list[dict]: List of changed files with their patches
    """
    url = f"{github_api_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def truncate_diff(diff: str, max_chars: int) -> tuple[str, bool]:
    """
    Truncate diff if it exceeds the maximum character limit.

    Args:
        diff: The original diff text
        max_chars: Maximum allowed characters

    Returns:
        tuple: (truncated_diff, was_truncated)
    """
    if len(diff) <= max_chars:
        return diff, False

    truncated = diff[:max_chars]
    # Try to truncate at a line boundary
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:  # At least 80% of content
        truncated = truncated[:last_newline]

    truncated += "\n\n[... diff truncated due to size ...]"
    return truncated, True


async def create_lightrag_instance(config: PRProcessorConfig) -> LightRAG:
    """
    Create and initialize a LightRAG instance for querying.

    Args:
        config: Processor configuration

    Returns:
        LightRAG: Initialized LightRAG instance
    """
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    # Define LLM function based on binding
    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        keyword_extraction: bool = False,
        **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            model=config.llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=config.llm_api_key,
            base_url=config.llm_host,
            **kwargs
        )

    # Define embedding function (use dedicated embedding endpoint, not LLM endpoint)
    # Use openai_embed.func to bypass the hardcoded embedding_dim=1536 in its
    # decorator — baai/bge-m3 returns 1024-dim, not 1536
    async def embedding_func(texts: list[str]) -> list[list[float]]:
        return await openai_embed.func(
            texts=texts,
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_host,
        )

    from lightrag.utils import EmbeddingFunc

    rag = LightRAG(
        working_dir=config.working_dir,
        kv_storage=config.kv_storage,
        vector_storage=config.vector_storage,
        graph_storage=config.graph_storage,
        doc_status_storage=config.doc_status_storage,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.embedding_dim,
            max_token_size=config.embedding_token_limit,
            model_name=config.embedding_model,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    return rag


async def query_lightrag_for_summary(
    rag: LightRAG,
    diff: str,
    pr_title: str,
    config: PRProcessorConfig
) -> dict[str, Any]:
    """
    Query LightRAG to summarize code changes and match requirements.

    Args:
        rag: LightRAG instance
        diff: The PR diff text
        pr_title: Title of the pull request
        config: Processor configuration

    Returns:
        dict: Contains 'summary' and 'matched_requirements'
    """
    query_prompt = f"""You are a software requirements analyst. A pull request has been submitted with the following code changes. Your task is to compare these changes against the project requirements stored in the knowledge base and identify any gaps.

Pull Request Title: {pr_title}

Code Diff:
```
{diff}
```

Based on the requirements in the knowledge base, please analyze and respond strictly in this format:

TITLE: <one-line high-level overview of what this diff does, e.g. "Implements soft delete for items endpoint (BR-31)">
SUMMARY: <2-3 sentence detailed analysis of what changed and how it relates to requirements>
FULFILLED: <list of requirement IDs or names that are fully implemented, one per line starting with "-", or "None" if none>
GAPS: <list of requirement IDs or names that are missing or incomplete, one per line starting with "-", include brief explanation of what is missing, or "None" if none>
ASSESSMENT: <overall verdict — exactly one of: Compliant / Partial / Non-compliant>
USE_CASES: <list of use case IDs from the knowledge base that this PR relates to, one per line starting with "-", or "None" if not found>
"""

    param = QueryParam(
        mode=config.query_mode,
        top_k=config.query_top_k,
        response_type="Multiple Paragraphs",
    )

    try:
        response = await rag.aquery(query_prompt, param=param)

        if not response:
            return {
                "summary": "No response from LightRAG query.",
                "matched_requirements": [],
                "raw_response": "",
                "error": "Empty response",
            }

        # Parse response into sections
        title = ""
        summary = ""
        fulfilled = []
        gaps = []
        assessment = ""
        use_case_ids = []

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("TITLE:"):
                current_section = "title"
                title = line[6:].strip()
            elif line.startswith("SUMMARY:"):
                current_section = "summary"
                summary = line[8:].strip()
            elif line.startswith("FULFILLED:"):
                current_section = "fulfilled"
                text = line[10:].strip()
                if text.startswith("-") or text.startswith("•"):
                    text = text[1:].strip()
                if text and text.lower() != "none":
                    fulfilled.append(text)
            elif line.startswith("GAPS:"):
                current_section = "gaps"
                text = line[5:].strip()
                if text.startswith("-") or text.startswith("•"):
                    text = text[1:].strip()
                if text and text.lower() != "none":
                    gaps.append(text)
            elif line.startswith("ASSESSMENT:"):
                current_section = "assessment"
                assessment = line[11:].strip()
            elif line.startswith("USE_CASES:"):
                current_section = "use_cases"
                text = line[10:].strip()
                if text.startswith("-") or text.startswith("•"):
                    text = text[1:].strip()
                if text and text.lower() != "none":
                    use_case_ids.append(text)
            elif current_section == "title" and line:
                title += " " + line
            elif current_section == "summary" and line:
                summary += " " + line
            elif current_section in ("fulfilled", "gaps", "use_cases") and line:
                if line.startswith("-") or line.startswith("•"):
                    item = line[1:].strip()
                else:
                    item = line
                if current_section == "fulfilled":
                    fulfilled.append(item)
                elif current_section == "gaps":
                    gaps.append(item)
                else:
                    use_case_ids.append(item)
            elif current_section == "assessment" and line:
                assessment += " " + line

        return {
            "title": title,
            "summary": summary or response[:500],
            "fulfilled": fulfilled,
            "gaps": gaps,
            "assessment": assessment,
            "use_case_ids": use_case_ids,
            "raw_response": response,
        }

    except Exception as e:
        logger.error(f"LightRAG query failed: {e}")
        return {
            "summary": f"Error generating summary: {str(e)}",
            "matched_requirements": [],
            "raw_response": "",
            "error": str(e),
        }


async def process_pr_async(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Async implementation of PR processing.

    Args:
        payload: GitHub webhook payload for pull_request event

    Returns:
        dict: Processing result with audit record ID
    """
    config = get_config()

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

    # Extract PR information from payload
    pr_data = payload.get("pull_request", {})
    repo_data = payload.get("repository", {})

    pr_number = pr_data.get("number")
    pr_title = pr_data.get("title", "")
    pr_url = pr_data.get("html_url", "")
    pr_body = pr_data.get("body", "") or ""
    pr_state = pr_data.get("state", "")
    pr_author = pr_data.get("user", {}).get("login", "")

    repo_full_name = repo_data.get("full_name", "")
    owner, repo = repo_full_name.split(
        "/") if "/" in repo_full_name else ("", "")

    action = payload.get("action", "")

    logger.info(f"Processing PR #{pr_number} ({action}) from {repo_full_name}")

    # Initialize audit storage
    audit_storage = PRAuditStorage.from_url(config.neon_database_url)
    await audit_storage.initialize()

    # Base title fallback (will be overwritten by LLM-generated title if available)
    base_title = f"PR #{pr_number}: {pr_title[:180]}" if pr_title else f"PR #{pr_number}"
    description = f"Processing PR #{pr_number} from {repo_full_name}..."
    severity = "Medium"
    status = "pending"
    saved_ids = []

    try:
        # Fetch PR diff
        logger.info(f"Fetching diff for PR #{pr_number}")
        diff = await fetch_pr_diff(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            github_token=config.github_token,
            github_api_url=config.github_api_url,
        )

        # Truncate if needed
        diff, _ = truncate_diff(diff, config.max_diff_chars)

        # Initialize LightRAG and query
        logger.info(f"Querying LightRAG for PR #{pr_number}")
        rag = await create_lightrag_instance(config)

        try:
            result = await query_lightrag_for_summary(rag, diff, pr_title, config)

            llm_title = result.get("title", "").strip()
            summary = result.get("summary", "")
            gaps = result.get("gaps", [])
            assessment = result.get("assessment", "")
            use_case_ids = result.get("use_case_ids", [])
            query_error = result.get("error", "")

            # Use LLM-generated title if available, otherwise fallback
            final_title = llm_title[:255] if llm_title else base_title

            description = summary or "No summary available."

            if query_error:
                status = "rejected"
                severity = "Critical"
            else:
                status = "approved"
                if len(gaps) >= 3 or "Non-compliant" in assessment:
                    severity = "High"
                elif len(gaps) >= 1 or "Partial" in assessment:
                    severity = "Medium"
                else:
                    severity = "Low"

            # Resolve LLM-returned uc_ids (e.g. "UC-001") to UUIDs via srs_use_cases table
            resolved_uuids = []
            for uc_id in use_case_ids:
                uuid_val = await audit_storage.lookup_use_case_uuid(uc_id)
                if uuid_val:
                    resolved_uuids.append(uuid_val)
                else:
                    logger.warning(f"No srs_use_cases row found for uc_id={uc_id!r}, skipping")

            # If resolved UUIDs found, create one record per use case
            # Otherwise fall back to config.use_case_id (or None)
            effective_use_case_ids = resolved_uuids if resolved_uuids else (
                [config.use_case_id] if config.use_case_id else [None]
            )

        finally:
            await rag.finalize_storages()

    except Exception as e:
        logger.error(f"Error processing PR #{pr_number}: {e}")
        final_title = base_title
        description = f"=== ERROR ===\n{str(e)}\n\n--- PR: {pr_url} | Author: {pr_author} | Action: {action} ---"
        status = "rejected"
        severity = "Critical"
        effective_use_case_ids = [config.use_case_id if config.use_case_id else None]

    # Save one audit record per use case ID
    for uc_id in effective_use_case_ids:
        record = PRAuditRecord(
            title=final_title,
            commit_id=str(pr_number),
            use_case_id=uc_id if uc_id else None,
            description=description,
            severity=severity,
            status=status,
        )
        record_id = await audit_storage.save_audit(record)
        saved_ids.append(record_id)
        logger.info(f"Saved audit record {record_id} for use_case_id={uc_id}")

    await audit_storage.close()

    logger.info(f"PR #{pr_number} processing complete. Audit IDs: {saved_ids}")

    return {
        "audit_ids": saved_ids,
        "pr_number": pr_number,
        "repo": repo_full_name,
        "status": status,
        "severity": severity,
        "title": final_title,
    }


@shared_task(
    bind=True,
    name="lightrag.workers.pr_worker.process_pull_request",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(httpx.HTTPError, ConnectionError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def process_pull_request(self, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Celery task to process a GitHub pull request webhook.

    This task:
    1. Fetches the PR diff from GitHub
    2. Uses LightRAG to summarize the code changes
    3. Matches the changes against indexed requirements
    4. Stores the full audit log in Neon PostgreSQL

    Args:
        payload: GitHub webhook payload for pull_request event

    Returns:
        dict: Processing result with status and audit record ID

    Raises:
        Retry: On transient errors (HTTP errors, connection issues)
    """
    logger.info(
        f"Starting PR processing task. Task ID: {self.request.id}"
    )

    try:
        # Run async code in sync context
        result = asyncio.run(process_pr_async(payload))
        return result

    except ValueError as e:
        # Configuration errors - don't retry
        logger.error(f"Configuration error: {e}")
        raise

    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise self.retry(exc=e)
