########## ACCEPT APPOINTMENT LOGIC ##############

import json
from sqlalchemy.ext.asyncio import AsyncSession
from app.db_logic.database import engine, execute_with_retry
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession  # Used for in-script testing
from .app_reminder import schedule_appointment_reminder
from .app_reminder import setup_scheduler  # This is used for in-script testing
import logging
from sqlalchemy import select
import regex as re
from app.db_logic.models import User, Appointment, AdminUser
import httpx
from dotenv import load_dotenv
import asyncio  # Just used for in script testing
import traceback  # Just used for in script testing
from app.settings import settings

load_dotenv()
WHATSAPP_API_URL = settings.WHATSAPP_API_URL
WHATSAPP_API_AUTHORIZATION = settings.WHATSAPP_API_AUTHORIZATION
OFFICE_LONGITUDE = settings.OFFICE_LONGITUDE
OFFICE_LATITUDE = settings.OFFICE_LATITUDE
OFFICE_ADDRESS = settings.OFFICE_ADDRESS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AcceptAppointment:
    def __init__(self, user_input, session: AsyncSession, scheduler):
        self.input = user_input
        self.session = session
        self.scheduler = scheduler
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        self.prompt = PromptTemplate(
            input_variables=["user_message"],
            template="""You are an assistant that extracts date, time, and ID from a user's message for appointment processing. The user's message is: {user_message}

Follow these steps to process the input:

1. **Date Processing:**
   - Check if the user has provided a full date, which must include the exact day, month, and year (e.g., '12/12/2024' or 'October 12th 2024').
   - Do not treat relative dates (e.g., 'tomorrow', 'next week') as full dates.
   - If a full date is provided and valid, convert it to the format 'day with ordinal suffix month year' (e.g., '10th October 2024').
   - If the date is not a full date (e.g., missing the year) or is invalid (e.g., 'February 30th'), set the 'date' to null.

2. **Time Processing:**
   - Extract the time from the user’s message if provided.
   - If a valid time is given (e.g., '2:30 PM', '14:30'), convert it to 24-hour format 'HH:MM' (e.g., '14:30').
   - If no time is provided or the time is invalid (e.g., '25:00'), set the 'time' to null.

3. **ID Processing:**
   - Look for an ID in the exact phrases 'The user's id is [number]' or 'the id is [number]' or 'user_id is [number]' (e.g., 'The user'id is 2323', 'the id is 234', the user_id is 123).
   - If found, set the 'id' to the number as a string (e.g., "2323"). If not present or in a different format, set 'id' to null.

4. **Output:**
   - Return the result in JSON format with the keys 'date', 'time', and 'id'.
   - Use double quotes for the JSON keys and string values.
   - Provide only the JSON output without any additional text.

**Examples:**
- Input: 'I want to book an appointment on 12/12/2024 at 2:30 PM the user_id is 234'
  Output: {{"date": "12th December 2024", "time": "14:30", "id": "234"}}
- Input: 'Let’s meet tomorrow'
  Output: {{"date": null, "time": null, "id": null}}
- Input: 'Appointment on October 12th 2024 The user_id is 2323'
  Output: {{"date": "12th October 2024", "time": null, "id": "2323"}}
- Input: 'At 15:00'
  Output: {{"date": null, "time": "15:00", "id": null}}
- Input: 'See you on February 30th'
  Output: {{"date": null, "time": null, "id": null}}
- Input: 'Accept meeting for 31st December 2024 at 11:59 PM the id is 789'
  Output: {{"date": "31st December 2024", "time": "23:59", "id": "789"}}

Process the user's message according to these rules and provide only the JSON output with double quotes.""")
        self.chain = self.prompt | self.llm | StrOutputParser(
        ) | RunnableLambda(self._parse_json)

    def _parse_json(self, output: str) -> dict:
        """Parse the output string into a JSON object, with fallback for errors."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"title": None, "date": None}

    async def validate_params(self):
        results = await self.chain.ainvoke({'user_message': self.input})
        if results['date'] and results['time'] and results['id']:
            return results
        else:
            return False

    def parse_date(self, date_str: str):
        try:
            date_str = date_str.strip().lower()
            if date_str == 'tomorrow':
                return (datetime.now() + timedelta(days=1)).date()

            parts = re.split(r'\s+', date_str)
            parts = [p.replace(',', '') for p in parts]

            day_part = parts[0]
            month_part = parts[1]
            year_part = parts[2] if len(parts) >= 3 else None

            day = int(re.sub(r'\D', '', day_part))
            month = datetime.strptime(month_part, "%B").month

            if year_part:
                year = int(year_part)
            else:
                year = datetime.now().year

            candidate_date = datetime(year, month, day).date()
            today = datetime.now().date()

            if candidate_date < today:
                year = year + 1

            parsed_date = datetime(year, month, day).date()
            return parsed_date
        except ValueError as error:
            return None

    def parse_time(self, time_str: str):
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours, minutes
        except:
            return None

    def split_date(self, date_str: str):
        months = ['january', 'february', "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december"]
        # Split the date string into parts and remove commas
        parts = [part.replace(',', '') for part in date_str.split()]

        # Extract day, month, and year
        day_part = parts[0]
        month = months.index(parts[1].lower()) + 1
        year = parts[2]

        # Extract numeric day (e.g., "10th" → 10)
        day = int(''.join(filter(str.isdigit, day_part)))
        year = int(year)

        list = [day, month, year]
        return list

    async def add_appt_to_db(self, information_dict):
        new_appointment = Appointment(**information_dict)
        self.session.add(new_appointment)
        await self.session.commit()
        await self.session.refresh(new_appointment)
        return new_appointment

    async def get_user_name(self, user_id):
        result = await self.session.execute(select(User.name).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_number(self, user_id):
        result = await self.session.execute(select(User.phone_number).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def appointment_approval_msg(self, number, user_name, date, time):
        """Method responsible for sending out the appointment accepted whatsapp message to the user."""
        async with httpx.AsyncClient() as client:
            message_response = await client.post(str(WHATSAPP_API_URL),
                                                 headers={
                                                     "Content-Type": "application/json", "Authorization": str(WHATSAPP_API_AUTHORIZATION)},
                                                 json={"messaging_product": "whatsapp",
                                                       "recipient_type": "individual",
                                                       "to": str(number),
                                                       "type": "template",
                                                       "template": {
                                                           "name": "accepted_appointment",
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
                                                                               "latitude": str(OFFICE_LATITUDE),
                                                                               "longitude": str(OFFICE_LONGITUDE),
                                                                               "name": "Apostle Uche's office",
                                                                               "address": str(OFFICE_ADDRESS)
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
                                                                           "text": str(user_name)
                                                                       },
                                                                       {
                                                                           "type": "text",
                                                                           "parameter_name": "apostle",
                                                                           "text": "Apostle Uche Raymond"
                                                                       },
                                                                       {
                                                                           "type": "text",
                                                                           "parameter_name": "date_and_time",
                                                                           "text": f"{str(date)} at {str(time)}"
                                                                       }
                                                                   ]
                                                               }
                                                           ]}})
            message_response.raise_for_status()
            upload_data = message_response.json()
            return upload_data

    async def is_admin(self, number):
        """Method responsible for checking if the user is an admin, returns admin details if yes returns None if the user is not."""
        result = await self.session.execute(select(AdminUser.phone_number).where(AdminUser.phone_number == number))
        return result.scalar_one_or_none()

    async def approve_appointment(self, number: str) -> str:
        if await execute_with_retry(self.is_admin, number):
            # if await self.is_admin(number):
            parameters = await self.validate_params()
            if parameters:
                try:
                    date = parameters['date']
                    time = parameters['time']
                    user_id = parameters['id']
                    date_str = self.parse_date(date)
                    time_str = self.parse_time(time)
                    if not date_str:
                        return "Invalid date format, please use this date format preferrably '12th October 2024'"

                    if not time_str:
                        return "Invalid date format used, please use this format preferrably '13:00'"

                except Exception as e:
                    return f"An error {e}: this is not valid use this format for date 10th October 2024 and this for time 11:24 try checking your spellings"

                try:
                    user_id = int(user_id)
                    number = await execute_with_retry(self.get_number, user_id)
                    # number = await self.get_number(user_id)
                    # user_name = await self.get_user_name(user_id)
                    user_name = await execute_with_retry(self.get_user_name, user_id)
                    appointment = await execute_with_retry(self.add_appt_to_db,
                                                           {
                                                               "user_id": user_id,
                                                               "user_name": user_name,
                                                               "phone_number": number,
                                                               "appointment_date": str(date_str),
                                                               "appointment_time": str(time_str),
                                                               "is_confirmed": True}
                                                           )
                    # appointment = await self.add_appt_to_db({
                    #     "user_id": user_id,
                    #     "user_name": user_name,
                    #     "phone_number": number,
                    #     "appointment_date": str(date_str),
                    #     "appointment_time": str(time_str),
                    #     "is_confirmed": True})
                    if not number:
                        logger.info("Invalid user_id entered by Admin")
                        return "user_id provided is not valid, please enter a valid user_id"

                    # await schedule_appointment_reminder(appointment, scheduler=self.scheduler)
                    await execute_with_retry(schedule_appointment_reminder, scheduler=self.scheduler)
                except Exception as e:
                    logger.error(
                        "Appointment acceptance failed: Failed to communicate with database")
                    return "It seems an error occured while trying to accept appointment, failed communicating with the database"

                await self.appointment_approval_msg(number, user_name,  date, time)

                return f"Successfully scheduled appointment with {user_name} at {date} by {time}, appointment_id is {appointment.id}"
            else:
                return "It appears you have not entered all the necessary detiails to perform this action. Please make sure you enter a date and time"
        else:
            return "You are not allowed to carry out this action"


if __name__ == "__main__":
    async def main():
        try:
            async with AsyncSession(engine) as session:
                scheduler = setup_scheduler()
                scheduler.start()
                acc_app = AcceptAppointment(
                    user_input="Accept this appointment the user's id is 1, date is 8th June 2025 at 22:10", session=session, scheduler=scheduler)
                response = await acc_app.approve_appointment("2349094540644")
                scheduler.shutdown()
                return response
        except BaseException as e:
            print(traceback.format_tb(e))

    x = asyncio.run(main())
    print(x)
