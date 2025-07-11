from pytz import timezone
# from config import app_settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
# from sqlalchemy.ext.asyncio import AsyncSession
from app.db_logic.database import engine
from datetime import datetime, timedelta
# Assume this is your SQLAlchemy model
from app.db_logic.models import Appointment
# from database import AsyncSessionLocal  # Updated to use async session
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.cron import CronTrigger
import logging
import ast
from pytz import timezone
from zoneinfo import ZoneInfo
import httpx
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure WAT timezone
WAT = ZoneInfo("Africa/Lagos")
WAT_timezone = timezone('Africa/Lagos')

# Global scheduler instance (for access without passing as argument)
scheduler = None

# Constants
WHATSTAPP_URL = settings.WHATSAPP_API_URL
WHATSAPP_AUTHORIZATION = settings.WHATSAPP_API_AUTHORIZATION
OFFICE_LONGITUDE = settings.OFFICE_LONGITUDE
OFFICE_LATITUDE = settings.OFFICE_LATITUDE
OFFICE_ADDRESS = settings.OFFICE_ADDRESS


async def get_async_db():
    """Dependency for async database session."""
    async with AsyncSession(engine) as session:
        yield session


def setup_scheduler():
    """Initialize the scheduler with a SQLAlchemy job store."""
    jobstores = {
        'default': SQLAlchemyJobStore(
            url=settings.NON_ASYNC_DATABASE_URL
        )
    }

    global scheduler
    scheduler = AsyncIOScheduler(jobstores=jobstores, timezone=WAT)
    scheduler.add_job(
        check_and_delete_old_appointments,
        trigger=CronTrigger(day="28-31", hour=23, minute=59, timezone=WAT),
        id="delete_old_appointments",
        replace_existing=True,
        args=[]  # Pass database session factory
    )

    return scheduler


async def delete_old_appointments(db: AsyncSession):
    """Delete appointments older than 30 days and remove associated scheduler jobs."""
    try:
        cutoff_date = datetime.now(WAT_timezone).date() - timedelta(days=30)
        result = await db.execute(
            select(Appointment).filter(
                Appointment.appointment_date < cutoff_date.strftime("%Y-%m-%d")
            )
        )
        old_appointments = result.scalars().all()
        if not old_appointments:
            logger.info("No appointments older than one month found.")
            return

        for appointment in old_appointments:
            await db.execute(
                delete(Appointment).filter(Appointment.id == appointment.id)
            )
            job_id = f"reminder_{appointment.id}"
            if scheduler and scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
                logger.info(
                    f"Removed scheduler job for appointment {appointment.id}")
        await db.commit()
        logger.info(
            f"Deleted {len(old_appointments)} appointments older than {cutoff_date}")
    except Exception as e:
        logger.error(f"Failed to delete old appointments: {str(e)}")
        await db.rollback()


async def check_and_delete_old_appointments():
    async with AsyncSession(engine) as db:
        """Check if today is the last day of the month and run delete_old_appointments."""
        today = datetime.now(WAT).date()
        # Check if tomorrow is the 1st of the next month
        tomorrow = today + timedelta(days=1)
        if tomorrow.day == 1:
            logger.info("Running end-of-month appointment cleanup")
            await delete_old_appointments(db)
        else:
            logger.debug(
                f"Skipping cleanup; today ({today}) is not the last day of the month")

# Initialize scheduler
scheduler = setup_scheduler()


async def schedule_appointment_reminder(appointment: Appointment, scheduler: AsyncIOScheduler):
    """Schedule a reminder for a new appointment in WAT."""
    try:
        # Parse appointment date and time in WAT
        appointment_date = datetime.strptime(
            appointment.appointment_date,
            "%Y-%m-%d"
        ).date()

        # Parse time tuple string "(11, 23)" -> (hour, minute)
        try:
            time_tuple = ast.literal_eval(appointment.appointment_time)
            if not isinstance(time_tuple, tuple) or len(time_tuple) != 2:
                raise ValueError
            hour, minute = time_tuple
        except (ValueError, SyntaxError):
            logger.error(
                f"Invalid time format for appointment {appointment.id}")
            return

        # Create WAT-aware datetime
        appointment_dt = WAT_timezone.localize(
            datetime.combine(
                appointment_date,
                datetime.min.time().replace(hour=hour, minute=minute)
            )
        )

        # Calculate reminder time (6 hours before in WAT)
        # reminder_time = appointment_dt - timedelta(hours=4)
        reminder_time = appointment_dt - timedelta(minutes=1)
        current_time = datetime.now(WAT)

        if reminder_time <= current_time:
            logger.info(
                f"Appointment {appointment.id} is within 4 hours, no reminder scheduled")
            return

        # Create unique job ID
        job_id = f"reminder_{appointment.id}"

        # Add job to scheduler with WAT time, passing appointment ID
        scheduler.add_job(
            send_reminder_notification,
            trigger=DateTrigger(run_date=reminder_time),
            # Pass ID instead of object for serialization
            args=[appointment.id],
            id=job_id,
            replace_existing=True,
            timezone=WAT
        )
        logger.info(f"Scheduled reminder for {appointment.id} at "
                    f"{reminder_time.strftime('%Y-%m-%d %H:%M %Z')}")
        job = scheduler.get_job(job_id)
        if job:
            logger.info(
                f"Job {job_id} confirmed in scheduler: next run at {job.next_run_time}")
        else:
            logger.error(
                f"Job {job_id} not found in scheduler after scheduling")
    except Exception as e:
        logger.error(f"Failed to schedule reminder: {str(e)}")


async def send_whatsapp_message(phone_number: str, name: str, appointment_date, appointment_time):
    media_metadata_url = WHATSTAPP_URL
    headers = {"Authorization": WHATSAPP_AUTHORIZATION,
               "Content-Type": "application/json"}

    data = {"messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": str(phone_number),
            "type": "template",
            "template": {
                "name": "appointment_reminder",
                "language": {
                    "code": "en_US"
                },
                "components": [
                    {
                        "type": "header",
                        "parameters": [
                            {
                                "type": "location",
                                "location": {
                                    "latitude": OFFICE_LATITUDE,
                                    "longitude": OFFICE_LONGITUDE,
                                    "name": "Apostle Uche's office",
                                    "address": OFFICE_ADDRESS
                                }
                            }
                        ]
                    },
                    {
                        "type": "body",
                        "parameters": [
                            {
                                "type": "text",
                                "parameter_name": "name",
                                "text": str(name)
                            },
                            {
                                "type": "text",
                                "parameter_name": "apostle",
                                "text": "Apostle Uche Raymond"
                            },
                            {
                                "type": "text",
                                "parameter_name": "date_and_time",
                                "text": f"{str(appointment_date)} at {str(appointment_time)}"
                            }
                        ]
                    }
                ]}}
    async with httpx.AsyncClient() as client:
        metadata_response = await client.post(media_metadata_url, headers=headers, json=data)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        try:
            reply = metadata.get("messages")
            reply_status = reply[0]['message_status']
            logging.info("Accepted appoitment message Succesfully sent")
        except BaseException as e:
            logging.error(f"This went wrong woth accepted appointment {e}")


async def send_reminder_notification(appointment_id: int):
    """Send reminder in WAT time context using async database session."""
    async with AsyncSession(engine) as db:
        try:
            # Query appointment asynchronously
            result = await db.execute(
                select(Appointment).filter(Appointment.id == appointment_id)
            )
            appointment = result.scalars().first()
            if not appointment:
                logger.warning(f"Appointment {appointment_id} not found")
                return

            # Format time in WAT
            try:
                hour, minute = ast.literal_eval(appointment.appointment_time)
                wat_time = f"{hour:02d}:{minute:02d} WAT"
                appointment_date = datetime.strptime(
                    appointment.appointment_date,
                    "%Y-%m-%d"
                ).strftime('%d %b %Y')
            except (ValueError, SyntaxError):
                wat_time = "Invalid Time"
                appointment_date = "Unknown Date"

            # Placeholder for async WhatsApp message sending
            await send_whatsapp_message(appointment.phone_number, appointment.user_name, appointment_date, wat_time)
            logger.info(
                f"Sent reminder for {appointment_id} to {appointment.phone_number}")

        except Exception as e:
            logger.error(f"Reminder failed for {appointment_id}: {str(e)}")
