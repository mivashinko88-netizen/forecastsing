"""
Background job scheduler for automated sync operations.
Uses APScheduler to run daily syncs of all connected integrations.
"""
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from database import SessionLocal
from db_models import Integration
from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

scheduler = AsyncIOScheduler()


async def sync_all_integrations():
    """Daily job to sync all active integrations."""
    logger.info(f"Starting scheduled sync at {datetime.utcnow()}")

    db = SessionLocal()
    try:
        # Import here to avoid circular imports
        from services.integrations.sync_service import SyncService

        # Get all connected integrations
        integrations = db.query(Integration).filter(
            Integration.status == "connected"
        ).all()

        logger.info(f"Found {len(integrations)} integrations to sync")

        sync_service = SyncService(db)

        for integration in integrations:
            try:
                logger.info(f"Syncing integration {integration.id} ({integration.provider}) for business {integration.business_id}")
                sync_log = await sync_service.sync_integration(
                    integration,
                    sync_type="orders"  # Daily sync focuses on new orders
                )
                if sync_log.status == "completed":
                    logger.info(f"Integration {integration.id} synced {sync_log.records_synced} records")
                else:
                    logger.warning(f"Integration {integration.id} sync failed: {sync_log.error_message}")
            except Exception as e:
                logger.error(f"Error syncing integration {integration.id}: {e}")

    except Exception as e:
        logger.error(f"Scheduled sync job failed: {e}")
    finally:
        db.close()

    logger.info(f"Scheduled sync completed at {datetime.utcnow()}")


def start_scheduler():
    """Start the background scheduler."""
    if not settings.SYNC_ENABLED:
        logger.info("Scheduled sync is disabled (SYNC_ENABLED=false)")
        return

    # Schedule daily sync at configured hour (default 2 AM UTC)
    scheduler.add_job(
        sync_all_integrations,
        trigger=CronTrigger(hour=settings.SYNC_SCHEDULE_HOUR, minute=0),
        id="daily_sync",
        name="Daily Integration Sync",
        replace_existing=True
    )

    scheduler.start()
    logger.info(f"Scheduler started - daily sync at {settings.SYNC_SCHEDULE_HOUR}:00 UTC")


def shutdown_scheduler():
    """Gracefully shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shutdown complete")
