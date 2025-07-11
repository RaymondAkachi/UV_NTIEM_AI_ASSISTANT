import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select
# from database import engine  # Used for in script testing
from app.db_logic.models import Appointment
from ast import literal_eval
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
# import asyncio # Used for in script testing
import httpx
import logging
# from app_reminder import setup_scheduler  # Used for in-script testing
from typing import Dict, List, Any
from app.settings import settings
from dotenv import load_dotenv

load_dotenv('.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHATSAPP_API_URL = settings.WHATSAPP_API_URL
WHATSAPP_API_AUTHORIZATION = settings.WHATSAPP_API_AUTHORIZATION


async def on_appointment_deleted(name: str, phone_number: str, date: str, time: str) -> None:
    # async with AsyncSession(engine) as db:
    # result = await db.execute(
    #         select(Appointment).filter(Appointment.id == appointment_id)
    #     )
    # appointment = result.scalars().first()
    # if not appointment:
    #     return
    # try:
    #     # hour, minute = literal_eval(time)
    #     # appointment_time = f"{hour:02d}:{minute:02d} WAT"
    #     # appointment_date = datetime.strptime(
    #     #     date,
    #     #     "%Y-%m-%d"
    #     # ).strftime('%d %b %Y')
    # except (ValueError, SyntaxError):
    #     appointment_time = "Invalid Time"
    #     appointment_date = "Unknown Date"

    media_metadata_url = WHATSAPP_API_URL
    headers = {"Authorization": WHATSAPP_API_AUTHORIZATION,
               "Content-Type": "application/json"}
    json_data = {"messaging_product": "whatsapp",
                 "recipient_type": "individual",
                 "to": str(phone_number),
                 "type": "template",
                 "template": {
                     "name": "cancelled_appointment",
                     "language": {
                         "code": "en_US"
                     },
                     "components": [
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
                                     "parameter_name": "date",
                                     "text": f"{str(date)}"
                                 },
                                 {
                                     "type": "text",
                                     "parameter_name": "time",
                                     "text": f"{str(time)}"
                                 }
                             ]
                         }
                     ]}}

    async with httpx.AsyncClient() as client:
        try:
            metadata_response = await client.post(media_metadata_url, headers=headers, json=json_data)
            metadata_response.raise_for_status()
            metadata = metadata_response.json()
            logging.info("Appointment deletion message successfully sent")
            return metadata
        except BaseException as e:
            logging.info(
                f"This went wrong with sending app cancellation message: {e}")


class DeleteAppointments:
    allowed_numbers = ['2348032235209', '2349094540644']

    def __init__(self, user_input: str, session: AsyncSession, scheduler: AsyncIOScheduler) -> None:
        self.input = user_input
        self.session = session
        self.scheduler = scheduler
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["user_message"],
            template="""Extract appointment IDs from the user's message for deleting appointments. Message: {user_message}

Extract:
- **Appointment IDs**: Numeric IDs mentioned in the message (e.g., "123", "appointment 456").

**Rules**:
- Output a JSON array of integers, each a distinct appointment ID.
- Extract only numeric values that represent IDs (ignore other numbers like dates or phone numbers).
- Return [] if no valid appointment IDs are found.
- Case-insensitive.

**Examples**:
- "delete appointment 123" → [123]
- "remove 456 and 789" → [456, 789]
- "appointment id 101, 202" → [101, 202]
- "no id here" → []
- "date 2024-11-11" → []

Output only the JSON array.
""")
        self.chain = self.prompt | self.llm | StrOutputParser(
        ) | RunnableLambda(self._parse_json)

    def _parse_json(self, output: str) -> list:
        """Parse the output string into a JSON object, with fallback for errors."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return []

    async def validate(self) -> list:
        """Extract and validate appointment IDs from user input."""
        results = await self.chain.ainvoke({'user_message': self.input})
        if not results:
            return []
        try:
            return [int(id) for id in results]
        except (ValueError, TypeError):
            return []

    async def delete_appointments(self, number: str) -> str:
        """Main function to delete appointments by ID."""
        if number not in DeleteAppointments.allowed_numbers:
            return "You are not allowed to carry out this action"

        appointment_ids = await self.validate()
        if not appointment_ids:
            return "Please provide valid appointment IDs to delete"

        try:
            # Check if appointments exist and retrieve name and phone number
            stmt = select(Appointment.id, Appointment.user_name, Appointment.phone_number,
                          Appointment.appointment_date, Appointment.appointment_time).where(Appointment.id.in_(appointment_ids))
            result = await self.session.execute(stmt)
            # List of tuples (id, user_name, phone_number)
            existing_appointments = result.all()

            if not existing_appointments:
                return "No appointments found with the provided IDs"

            # Delete appointments and call user-defined function
            deleted_info = []
            for appt_id, name, phone, date, time in existing_appointments:
                # Delete the specific appointment
                delete_stmt = delete(Appointment).where(
                    Appointment.id == appt_id)
                await self.session.execute(delete_stmt)
                # Call user-defined function

                hour, minute = literal_eval(time)
                appointment_time = f"{hour:02d}:{minute:02d} WAT"
                appointment_date = datetime.strptime(
                    date,
                    "%Y-%m-%d"
                ).strftime('%d %b %Y')
                await on_appointment_deleted(name, phone, appointment_date, appointment_time)
                deleted_info.append(
                    f"ID {appt_id} (Name: {name}, Phone: {phone}, Date: {appointment_date}, Time: {appointment_time})")

                job_id = f"reminder_{appt_id}"
                if self.scheduler.get_job(job_id):
                    self.scheduler.remove_job(job_id)
                    logger.info(
                        f"Removed reminder job for appointment {appt_id}")

            await self.session.commit()
            return f"Deleted {len(deleted_info)} appointment(s):\n" + "\n".join(deleted_info)

        except Exception as e:
            await self.session.rollback()
            return f"Error deleting appointments: {str(e)}"


async def get_all_scheduler_jobs(scheduler: AsyncIOScheduler) -> List[Dict[str, Any]]:
    """
    Retrieve all jobs from the APScheduler instance.

    Args:
        scheduler (AsyncIOScheduler): The APScheduler instance.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing job details.
    """
    try:
        jobs = scheduler.get_jobs()
        job_details = []

        for job in jobs:
            job_info = {
                "id": job.id,
                "name": job.name,
                "func": str(job.func),
                "args": job.args,
                "trigger": str(job.trigger),
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "timezone": str(job.next_run_time.tzinfo) if job.next_run_time else None
            }
            job_details.append(job_info)

        logger.info(f"Retrieved {len(job_details)} jobs from scheduler")
        return job_details

    except Exception as e:
        logger.error(f"Failed to retrieve jobs: {str(e)}")
        return []


# if __name__ == "__main__":
#     async def see_jobs():
#         scheduler = setup_scheduler()
#         scheduler.start()
#         res = await get_all_scheduler_jobs(scheduler)
#         print(res)
#         scheduler.shutdown()
#     asyncio.run(see_jobs())


# if __name__ == "__main__":
#     async def del_app_test():
#         scheduler = setup_scheduler()
#         scheduler.start()
#         async with AsyncSession(engine) as session:
#             delete_appointment = DeleteAppointments(
#                 "delete this appointment of id 60", session, scheduler)
#             res = await delete_appointment.delete_appointments('2349094540644')
#             print(res)
#         scheduler.shutdown()
#     asyncio.run(del_app_test())
