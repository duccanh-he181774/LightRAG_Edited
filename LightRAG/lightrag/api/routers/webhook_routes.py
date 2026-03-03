"""
GitHub Webhook Routes

This module provides FastAPI endpoints for receiving GitHub webhooks,
specifically for pull request events. It validates webhook signatures,
filters relevant events, and enqueues tasks for async processing.

Endpoints:
    POST /webhooks/github: Receive GitHub webhook events
    GET /webhooks/github/status: Check webhook integration status
    GET /webhooks/audits: List PR audit records
    GET /webhooks/audits/{audit_id}: Get specific audit record

Security:
    - Webhook signature verification using X-Hub-Signature-256
    - Optional API key authentication for audit endpoints
"""

import hashlib
import hmac
import json
import logging
import os
from typing import Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class WebhookResponse(BaseModel):
    """Response model for webhook endpoint."""
    status: str = Field(..., description="Status of webhook processing")
    message: str = Field(..., description="Descriptive message")
    task_id: Optional[str] = Field(
        None, description="Celery task ID if enqueued")
    pr_number: Optional[int] = Field(None, description="Pull request number")
    repo: Optional[str] = Field(None, description="Repository full name")


class WebhookStatusResponse(BaseModel):
    """Response model for webhook status endpoint."""
    enabled: bool = Field(...,
                          description="Whether webhook integration is enabled")
    celery_connected: bool = Field(...,
                                   description="Whether Celery broker is connected")
    audit_db_connected: bool = Field(...,
                                     description="Whether audit DB is connected")
    supported_events: list[str] = Field(
        default=["pull_request"],
        description="List of supported GitHub event types"
    )
    supported_actions: list[str] = Field(
        default=["opened", "synchronize", "reopened"],
        description="List of supported PR actions"
    )


class AuditRecordResponse(BaseModel):
    """Response model for audit record."""
    id: str
    pr_number: int
    repo_full_name: str
    pr_title: str
    pr_url: str
    pr_author: str
    webhook_action: str
    summary: str
    matched_requirements: list[str]
    status: str
    created_at: datetime
    processed_at: Optional[datetime]


class AuditListResponse(BaseModel):
    """Response model for audit list."""
    total: int
    items: list[AuditRecordResponse]
    limit: int
    offset: int


class AuditStatsResponse(BaseModel):
    """Response model for audit statistics."""
    total: int
    by_status: dict[str, int]
    avg_processing_seconds: float


# =============================================================================
# Webhook Signature Verification
# =============================================================================

def verify_github_signature(
    payload_body: bytes,
    signature_header: str,
    secret: str
) -> bool:
    """
    Verify GitHub webhook signature.

    GitHub signs webhook payloads with HMAC-SHA256 using the webhook secret.
    This function verifies that the signature matches.

    Args:
        payload_body: Raw request body bytes
        signature_header: X-Hub-Signature-256 header value
        secret: Webhook secret configured in GitHub

    Returns:
        bool: True if signature is valid
    """
    if not signature_header:
        return False

    # GitHub signature format: sha256=<hex_digest>
    if not signature_header.startswith("sha256="):
        return False

    expected_signature = signature_header[7:]  # Remove "sha256=" prefix

    # Calculate HMAC-SHA256
    mac = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256
    )
    calculated_signature = mac.hexdigest()

    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(calculated_signature, expected_signature)


# =============================================================================
# Router Factory
# =============================================================================

def create_webhook_routes(
    api_key: Optional[str] = None,
    webhook_secret: Optional[str] = None,
) -> APIRouter:
    """
    Create webhook router with configured authentication.

    Args:
        api_key: Optional API key for audit endpoint authentication
        webhook_secret: GitHub webhook secret for signature verification

    Returns:
        APIRouter: Configured router instance
    """
    from lightrag.api.utils_api import get_combined_auth_dependency

    router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

    # Get webhook secret from parameter or environment
    _webhook_secret = webhook_secret or os.getenv("GITHUB_WEBHOOK_SECRET", "")

    # Supported event types and actions
    SUPPORTED_EVENTS = {"pull_request"}
    SUPPORTED_ACTIONS = {"opened", "synchronize", "reopened"}

    # ==========================================================================
    # Webhook Endpoint
    # ==========================================================================

    @router.post(
        "/github",
        response_model=WebhookResponse,
        summary="Receive GitHub Webhook",
        description="Endpoint for receiving GitHub webhook events. "
                    "Validates signature and enqueues PR processing tasks.",
        responses={
            200: {"description": "Webhook processed successfully"},
            202: {"description": "Task enqueued for processing"},
            400: {"description": "Invalid payload or unsupported event"},
            401: {"description": "Invalid webhook signature"},
        }
    )
    async def receive_github_webhook(
        request: Request,
        x_hub_signature_256: Optional[str] = Header(None),
        x_github_event: Optional[str] = Header(None),
        x_github_delivery: Optional[str] = Header(None),
    ) -> JSONResponse:
        """
        Receive and process GitHub webhook events.

        This endpoint:
        1. Verifies the webhook signature
        2. Filters for supported events (pull_request)
        3. Filters for supported actions (opened, synchronize, reopened)
        4. Enqueues a Celery task for async processing

        Headers:
            X-Hub-Signature-256: HMAC signature of the payload
            X-GitHub-Event: Event type (e.g., pull_request)
            X-GitHub-Delivery: Unique delivery ID

        Returns:
            202 Accepted with task ID if enqueued
            200 OK if acknowledged but not processed (filtered out)
            400 Bad Request for invalid payloads
            401 Unauthorized for invalid signatures
        """
        # Get raw body for signature verification
        body = await request.body()

        # Verify signature if secret is configured
        if _webhook_secret:
            if not verify_github_signature(body, x_hub_signature_256 or "", _webhook_secret):
                logger.warning(
                    f"Invalid webhook signature. Delivery ID: {x_github_delivery}"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )

        # Parse payload
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse webhook payload: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload"
            )

        # Check event type
        if x_github_event not in SUPPORTED_EVENTS:
            logger.info(f"Ignoring unsupported event type: {x_github_event}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=WebhookResponse(
                    status="ignored",
                    message=f"Event type '{x_github_event}' is not supported",
                ).model_dump()
            )

        # Check action for pull_request events
        action = payload.get("action", "")
        if action not in SUPPORTED_ACTIONS:
            logger.info(f"Ignoring unsupported action: {action}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=WebhookResponse(
                    status="ignored",
                    message=f"Action '{action}' is not supported",
                ).model_dump()
            )

        # Extract PR info for response
        pr_data = payload.get("pull_request", {})
        pr_number = pr_data.get("number")
        repo_data = payload.get("repository", {})
        repo_full_name = repo_data.get("full_name", "")

        logger.info(
            f"Received PR webhook: #{pr_number} ({action}) from {repo_full_name}. "
            f"Delivery ID: {x_github_delivery}"
        )

        # Enqueue Celery task
        try:
            from lightrag.workers.pr_worker import process_pull_request

            task = process_pull_request.delay(payload)
            task_id = task.id

            logger.info(f"Enqueued task {task_id} for PR #{pr_number}")

            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=WebhookResponse(
                    status="accepted",
                    message="PR processing task enqueued",
                    task_id=task_id,
                    pr_number=pr_number,
                    repo=repo_full_name,
                ).model_dump()
            )

        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to enqueue processing task: {str(e)}"
            )

    # ==========================================================================
    # Status Endpoint
    # ==========================================================================

    @router.get(
        "/github/status",
        response_model=WebhookStatusResponse,
        summary="Webhook Integration Status",
        description="Check the status of webhook integration components."
    )
    async def get_webhook_status() -> WebhookStatusResponse:
        """
        Check webhook integration status.

        Returns information about:
        - Whether webhook integration is enabled
        - Celery broker connectivity
        - Audit database connectivity
        - Supported events and actions
        """
        celery_connected = False
        audit_db_connected = False

        # Check Celery connection
        try:
            from lightrag.workers.celery_app import celery_app
            celery_app.control.ping(timeout=2.0)
            celery_connected = True
        except Exception as e:
            logger.debug(f"Celery connection check failed: {e}")

        # Check audit DB connection
        try:
            neon_url = os.getenv("NEON_DATABASE_URL")
            if neon_url:
                from lightrag.kg.pr_audit_storage import PRAuditStorage
                storage = PRAuditStorage.from_url(neon_url)
                await storage.initialize()
                await storage.close()
                audit_db_connected = True
        except Exception as e:
            logger.debug(f"Audit DB connection check failed: {e}")

        return WebhookStatusResponse(
            enabled=bool(_webhook_secret),
            celery_connected=celery_connected,
            audit_db_connected=audit_db_connected,
            supported_events=list(SUPPORTED_EVENTS),
            supported_actions=list(SUPPORTED_ACTIONS),
        )

    # ==========================================================================
    # Test Endpoint (for debugging queue flow)
    # ==========================================================================

    class TestWebhookRequest(BaseModel):
        """Request model for test webhook."""
        pr_number: int = Field(default=999, description="Fake PR number")
        repo: str = Field(default="test/repo", description="Fake repo name")
        title: str = Field(default="Test PR", description="Fake PR title")
        action: str = Field(default="opened", description="PR action")

    class TestWebhookResponse(BaseModel):
        """Response model for test webhook."""
        status: str
        message: str
        task_id: Optional[str] = None
        queue_info: Optional[dict] = None

    @router.post(
        "/test",
        response_model=TestWebhookResponse,
        summary="Test Webhook Queue",
        description="Send a fake PR event to test the queue flow. For debugging only."
    )
    async def test_webhook_queue(
        request: TestWebhookRequest = TestWebhookRequest()
    ) -> TestWebhookResponse:
        """
        Send a fake PR event to test the queue.
        
        This helps debug:
        1. Whether Celery/RabbitMQ is connected
        2. Whether tasks are being enqueued
        3. Whether workers are processing tasks
        """
        # Create fake payload
        fake_payload = {
            "action": request.action,
            "pull_request": {
                "number": request.pr_number,
                "title": request.title,
                "html_url": f"https://github.com/{request.repo}/pull/{request.pr_number}",
                "body": "This is a test PR for debugging webhook flow.",
                "state": "open",
                "user": {"login": "test-user"},
            },
            "repository": {
                "full_name": request.repo,
            },
        }

        # Try to enqueue
        try:
            from lightrag.workers.pr_worker import process_pull_request
            from lightrag.workers.celery_app import celery_app

            # Check broker connection first
            try:
                inspect = celery_app.control.inspect()
                active_queues = inspect.active_queues()
                registered_tasks = inspect.registered()
                
                queue_info = {
                    "broker_url": celery_app.conf.broker_url,
                    "active_queues": active_queues,
                    "registered_tasks": registered_tasks,
                    "workers_online": active_queues is not None,
                }
            except Exception as e:
                queue_info = {
                    "broker_url": celery_app.conf.broker_url,
                    "error": str(e),
                    "workers_online": False,
                }

            # Enqueue task
            task = process_pull_request.delay(fake_payload)
            
            return TestWebhookResponse(
                status="enqueued",
                message=f"Test task enqueued for PR #{request.pr_number}",
                task_id=task.id,
                queue_info=queue_info,
            )

        except Exception as e:
            logger.error(f"Test webhook failed: {e}")
            return TestWebhookResponse(
                status="error",
                message=f"Failed to enqueue: {str(e)}",
                queue_info={"error": str(e)},
            )

    @router.get(
        "/test/task/{task_id}",
        summary="Check Task Status",
        description="Check the status of a Celery task by ID."
    )
    async def check_task_status(task_id: str) -> dict:
        """Check the status of a queued task."""
        try:
            from lightrag.workers.celery_app import celery_app
            
            result = celery_app.AsyncResult(task_id)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "ready": result.ready(),
                "successful": result.successful() if result.ready() else None,
                "result": result.result if result.ready() and result.successful() else None,
                "error": str(result.result) if result.ready() and not result.successful() else None,
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
            }

    # ==========================================================================
    # Audit Endpoints (Protected)
    # ==========================================================================

    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/audits",
        response_model=AuditListResponse,
        summary="List PR Audits",
        description="List PR audit records with optional filters.",
        dependencies=[Depends(combined_auth)],
    )
    async def list_audits(
        repo: Optional[str] = None,
        audit_status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AuditListResponse:
        """
        List PR audit records.

        Args:
            repo: Filter by repository (owner/repo)
            audit_status: Filter by status (pending, processing, completed, failed)
            limit: Maximum records to return (default: 50)
            offset: Offset for pagination

        Returns:
            List of audit records with pagination info
        """
        neon_url = os.getenv("NEON_DATABASE_URL")
        if not neon_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Audit storage not configured"
            )

        try:
            from lightrag.kg.pr_audit_storage import PRAuditStorage

            storage = PRAuditStorage.from_url(neon_url)
            await storage.initialize()

            try:
                records = await storage.list_audits(
                    repo_full_name=repo,
                    status=audit_status,
                    limit=limit,
                    offset=offset,
                )

                stats = await storage.get_stats(repo_full_name=repo)

                items = [
                    AuditRecordResponse(
                        id=r.id,
                        pr_number=r.pr_number,
                        repo_full_name=r.repo_full_name,
                        pr_title=r.pr_title,
                        pr_url=r.pr_url,
                        pr_author=r.pr_author,
                        webhook_action=r.webhook_action,
                        summary=r.summary,
                        matched_requirements=r.matched_requirements,
                        status=r.status,
                        created_at=r.created_at,
                        processed_at=r.processed_at,
                    )
                    for r in records
                ]

                return AuditListResponse(
                    total=stats["total"],
                    items=items,
                    limit=limit,
                    offset=offset,
                )

            finally:
                await storage.close()

        except Exception as e:
            logger.error(f"Failed to list audits: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve audit records: {str(e)}"
            )

    @router.get(
        "/audits/stats",
        response_model=AuditStatsResponse,
        summary="Audit Statistics",
        description="Get statistics about PR audits.",
        dependencies=[Depends(combined_auth)],
    )
    async def get_audit_stats(
        repo: Optional[str] = None,
    ) -> AuditStatsResponse:
        """
        Get audit statistics.

        Args:
            repo: Optional filter by repository

        Returns:
            Statistics including totals, status breakdown, and average processing time
        """
        neon_url = os.getenv("NEON_DATABASE_URL")
        if not neon_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Audit storage not configured"
            )

        try:
            from lightrag.kg.pr_audit_storage import PRAuditStorage

            storage = PRAuditStorage.from_url(neon_url)
            await storage.initialize()

            try:
                stats = await storage.get_stats(repo_full_name=repo)
                return AuditStatsResponse(**stats)
            finally:
                await storage.close()

        except Exception as e:
            logger.error(f"Failed to get audit stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve audit statistics: {str(e)}"
            )

    @router.get(
        "/audits/{audit_id}",
        response_model=AuditRecordResponse,
        summary="Get PR Audit",
        description="Get a specific PR audit record by ID.",
        dependencies=[Depends(combined_auth)],
    )
    async def get_audit(audit_id: str) -> AuditRecordResponse:
        """
        Get a specific audit record.

        Args:
            audit_id: The audit record UUID

        Returns:
            The audit record
        """
        neon_url = os.getenv("NEON_DATABASE_URL")
        if not neon_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Audit storage not configured"
            )

        try:
            from lightrag.kg.pr_audit_storage import PRAuditStorage

            storage = PRAuditStorage.from_url(neon_url)
            await storage.initialize()

            try:
                record = await storage.get_audit(audit_id)
                if not record:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Audit record not found: {audit_id}"
                    )

                return AuditRecordResponse(
                    id=record.id,
                    pr_number=record.pr_number,
                    repo_full_name=record.repo_full_name,
                    pr_title=record.pr_title,
                    pr_url=record.pr_url,
                    pr_author=record.pr_author,
                    webhook_action=record.webhook_action,
                    summary=record.summary,
                    matched_requirements=record.matched_requirements,
                    status=record.status,
                    created_at=record.created_at,
                    processed_at=record.processed_at,
                )

            finally:
                await storage.close()

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get audit: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve audit record: {str(e)}"
            )

    return router
