"""
Worker Entry Point Script

This script provides the entry point for running Celery workers.
It can be run directly or used as a systemd service.

Usage:
    # Run worker directly
    python -m lightrag.workers.run_worker

    # Or use celery command
    celery -A lightrag.workers.celery_app worker --loglevel=info -Q pr_webhook,default

    # Run with specific concurrency
    celery -A lightrag.workers.celery_app worker --loglevel=info --concurrency=4

    # Run with flower monitoring (optional)
    celery -A lightrag.workers.celery_app flower --port=5555
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables before importing celery
load_dotenv(dotenv_path=".env", override=False)


def setup_logging():
    """Configure logging for the worker."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    """Main entry point for the worker."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting LightRAG PR Webhook Worker...")

    # Import celery app
    from lightrag.workers.celery_app import celery_app

    # Default arguments if not provided
    # Use 'solo' pool on Windows (prefork has issues)
    default_args = [
        "worker",
        "--loglevel=info",
        "-Q", "pr_webhook,default",
        "--pool=solo",  # Required for Windows compatibility
    ]

    # Use command line args if provided, otherwise use defaults
    if len(sys.argv) > 1:
        celery_app.start(argv=sys.argv[1:])
    else:
        celery_app.start(argv=default_args)


if __name__ == "__main__":
    main()
