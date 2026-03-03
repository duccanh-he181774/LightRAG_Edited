"""
Celery Application Configuration for LightRAG Workers

This module configures the Celery app for processing background tasks,
including GitHub PR webhooks. It uses RabbitMQ as the message broker
and optionally PostgreSQL or Redis as the result backend.

Usage:
    celery -A lightrag.workers.celery_app worker --loglevel=info

Environment Variables:
    CELERY_BROKER_URL: RabbitMQ connection URL (required)
    CELERY_RESULT_BACKEND: Result backend URL (optional, defaults to rpc://)
    CELERY_TASK_SERIALIZER: Task serializer format (default: json)
    CELERY_RESULT_SERIALIZER: Result serializer format (default: json)
    CELERY_ACCEPT_CONTENT: Accepted content types (default: json)
    CELERY_TIMEZONE: Timezone for scheduled tasks (default: UTC)
    CELERY_TASK_TRACK_STARTED: Track task started state (default: True)
    CELERY_TASK_TIME_LIMIT: Hard time limit per task in seconds (default: 300)
"""

import os
from celery import Celery
from dotenv import load_dotenv
from kombu import Exchange, Queue

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)


def get_celery_config() -> dict:
    """
    Build Celery configuration from environment variables.

    Returns:
        dict: Celery configuration dictionary
    """
    return {
        # Broker settings (RabbitMQ)
        "broker_url": os.getenv(
            "CELERY_BROKER_URL",
            "amqp://guest:guest@localhost:5672//"
        ),

        # Result backend (optional - can use PostgreSQL, Redis, or RPC)
        "result_backend": os.getenv(
            "CELERY_RESULT_BACKEND",
            "rpc://"  # Default to RPC for simple setups
        ),

        # Serialization
        "task_serializer": os.getenv("CELERY_TASK_SERIALIZER", "json"),
        "result_serializer": os.getenv("CELERY_RESULT_SERIALIZER", "json"),
        "accept_content": ["json"],

        # Timezone
        "timezone": os.getenv("CELERY_TIMEZONE", "UTC"),
        "enable_utc": True,

        # Task execution settings
        "task_track_started": True,
        "task_time_limit": int(os.getenv("CELERY_TASK_TIME_LIMIT", "300")),
        "task_soft_time_limit": int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "270")),

        # Retry settings
        "task_acks_late": True,  # Acknowledge after task completes
        "task_reject_on_worker_lost": True,

        # Worker settings
        "worker_prefetch_multiplier": 1,  # One task at a time per worker
        "worker_concurrency": int(os.getenv("CELERY_WORKER_CONCURRENCY", "2")),

        # Task result settings
        # 24 hours
        "result_expires": int(os.getenv("CELERY_RESULT_EXPIRES", "86400")),

        # Task routing
        "task_queues": (
            Queue(
                "pr_webhook",
                Exchange("pr_webhook"),
                routing_key="pr_webhook",
                queue_arguments={"x-max-priority": 10}
            ),
            Queue(
                "default",
                Exchange("default"),
                routing_key="default"
            ),
        ),
        "task_default_queue": "default",
        "task_default_exchange": "default",
        "task_default_routing_key": "default",

        # Task routes
        "task_routes": {
            "lightrag.workers.pr_worker.process_pull_request": {
                "queue": "pr_webhook",
                "routing_key": "pr_webhook",
            },
        },
    }


# Create Celery application
celery_app = Celery("lightrag_workers")

# Apply configuration
celery_app.config_from_object(get_celery_config())

# Explicitly include task modules (autodiscover looks for tasks.py by default)
celery_app.conf.update(
    include=["lightrag.workers.pr_worker"]
)


# Optional: Add startup/shutdown hooks
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    Configure periodic tasks (if needed in the future).

    This hook is called after Celery is configured and can be used
    to set up periodic tasks like cleanup jobs.
    """
    pass


if __name__ == "__main__":
    celery_app.start()
