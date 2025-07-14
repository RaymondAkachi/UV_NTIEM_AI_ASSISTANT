import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import calendar
from sqlalchemy.ext.asyncio import AsyncSession
from app.db_logic.database import engine  # Used for in script testing
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import regex as re
from app.db_logic.models import Appointment, AdminUser
from sqlalchemy import select, distinct
from app.db_logic.database import execute_with_retry
import asyncio  # Used for in-script testing
import time  # Used for in-script testing
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class ReadAppointments:
    allowed_numbers = ['2348032235209', '2349094540644']

    month_names = {}
    for i in range(1, 13):
        name = calendar.month_name[i].lower()
        abbr = calendar.month_abbr[i].lower()
        month_names[name] = i
        month_names[abbr] = i

    def __init__(self, user_input: str, session: AsyncSession) -> None:
        self.input = user_input
        self.session = session
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["user_message"],
            template="""Extract mentions of days, dates, or time periods from the user's message for reading appointments. Message: {user_message}

Extract:
- **Days**: Weekdays (e.g., "Monday"), relative days (e.g., "today", "tomorrow", "next Monday").
- **Dates**: Full dates (e.g., "12/12/2024", "12th May 2024") or partial (e.g., "12th May").
- **Time Periods**: Ranges (e.g., "this week", "next two weeks", "next month", "in three days").

**Rules**:
- Output a JSON array of strings, each a distinct mention.
- Combine day and date if specified together (e.g., "Monday, 12th May 2024" → "Monday, 12th May 2024").
- Standardize full dates to "day ordinal month year" (e.g., "12/12/2024" → "12th December 2024"). Preserve partial/relative dates as provided.
- Autocorrect common misspellings (e.g., "Tusday" → "Tuesday", "tommorow" → "tomorrow", "wee" → "week", "apointments" → "appointments").
- Ignore possessive terms (e.g., "my") or verbs (e.g., "get", "show") but extract the date/time period.
- Return [] if no valid mentions are found.
- Case-insensitive.

**Examples**:
- "get me my appointments for today" → ["today"]
- "get all my appointments for 12th May 2024" → ["12th May 2024"]
- "show me my appointments for this wee" → ["this week"]
- "tomorrow and next week" → ["tomorrow", "next week"]
- "12/12/2024, next Monday" → ["12th December 2024", "next Monday"]
- "next two weeks, in three days" → ["next two weeks", "in three days"]
- "nothing" → []

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

    async def validate(self) -> str:
        results = await self.chain.ainvoke({'user_message': self.input})
        if results == []:
            return False
        else:
            return results

    def parse_date(self, input_date_str, current_date):
        input_lower = input_date_str.lower().strip()

        natural_dates = {
            'yesterday': (current_date - timedelta(days=1), ),
            'tomorrow': (current_date + timedelta(days=1), ),
            'today': (current_date,),
            'next two weeks': (current_date, current_date + timedelta(days=13)),
            'next month': (current_date.replace(day=1) + relativedelta(months=1),
                           (current_date.replace(day=1) + relativedelta(months=2)) - timedelta(days=1)),
            'this month': (current_date.replace(day=1),
                           (current_date.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)),
        }
        if input_lower in natural_dates:
            return natural_dates[input_lower]

        # Updated week calculations (Sunday as start)
        elif input_lower == 'this week':
            # Calculate Sunday as start of week
            days_to_subtract = (current_date.weekday() +
                                1) % 7  # Key adjustment
            start_of_week = current_date - timedelta(days=days_to_subtract)
            end_of_week = start_of_week + timedelta(days=6)
            return (start_of_week, end_of_week)

        elif input_lower == 'next week':
            # Calculate next week (Sunday to Saturday)
            days_to_subtract = (current_date.weekday() + 1) % 7
            start_of_this_week = current_date - \
                timedelta(days=days_to_subtract)
            start_of_next_week = start_of_this_week + timedelta(weeks=1)
            end_of_next_week = start_of_next_week + timedelta(days=6)
            return (start_of_next_week, end_of_next_week)

        # Check ISO format (YYYY-MM-DD) with regex validation
        iso_date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if re.match(iso_date_pattern, input_date_str):
            try:
                parsed_date = datetime.strptime(
                    input_date_str, "%Y-%m-%d").date()
                return (parsed_date,)
            except ValueError:
                pass  # Invalid date (e.g., 2024-13-32)

        # Handle other date formats (e.g., "12th", "3rd October")
        cleaned = re.sub(r'[,.]', ' ', input_date_str)
        parts = re.split(r'\s+', cleaned.strip())
        day = None
        month = current_date.month
        year = current_date.year
        for part in parts:
            part_lower = part.lower()
            # Check for day with suffix
            day_match = re.match(r'^(\d+)(st|nd|rd|th)$', part_lower)
            if day_match:
                day = int(day_match.group(1))
                continue
            # Check for month
            if part_lower in ReadAppointments.month_names:
                month = ReadAppointments.month_names[part_lower]
                continue
            # Check for 4-digit year
            if re.match(r'^\d{4}$', part_lower):
                year = int(part_lower)
                continue
            # Check for day without suffix
            if day is None and re.match(r'^\d{1,2}$', part_lower):
                day = int(part_lower)
        # Validate day is found
        if day is None:
            raise ValueError(f"Unable to determine day from input: {
                input_date_str}")
        # Validate and create the date
        try:
            parsed_date = datetime(year, month, day).date()
            return (parsed_date,)
        except ValueError as e:
            raise ValueError(
                f"Invalid date components: {input_date_str} - {e}")

    def time_recognizer(self, input_date_str, date_list):
        current_date = datetime.today().date()
        try:
            parsed_result = self.parse_date(input_date_str, current_date)
        except ValueError:
            return []

        # Handle date ranges or single dates
        if isinstance(parsed_result, tuple):
            if len(parsed_result) == 1:
                # Single date (e.g., "tomorrow", "2024-11-11")
                target_date = parsed_result[0]
                return [date_str for date_str in date_list
                        if datetime.strptime(date_str, "%Y-%m-%d").date() == target_date]
            else:
                # Date range (e.g., "next week", "this month")
                start_date, end_date = parsed_result
                matches = []
                for date_str in date_list:
                    try:
                        parsed_date = datetime.strptime(
                            date_str, "%Y-%m-%d").date()
                        if start_date <= parsed_date <= end_date:
                            matches.append(date_str)
                    except ValueError:
                        continue
                return matches
        else:
            return []

    async def filter_dates_by_criteria(self, criteria):
        """Use your time_recognizer to find matching dates"""
        stmt = select(distinct(Appointment.appointment_date))
        result = await self.session.execute(stmt)
        all_dates = [row[0] for row in result.all()]
        # all_dates = get_all_appointment_dates()
        # Your existing function
        return self.time_recognizer(criteria, all_dates)

    async def is_admin(self, number):
        result = await self.session.execute(select(AdminUser.phone_number).where(AdminUser.phone_number == number))
        return result.scalar_one_or_none()

    async def get_appointments_by_criteria(self, number):
        """Main function to get appointments matching date criteria"""
        if await execute_with_retry(self.is_admin, number):
            # if number in ReadAppointments.allowed_numbers:
            results = await self.validate()
            if results:
                response = ''
                response_string = ""
                for criteria in results:
                    try:
                        try:
                            criteria = criteria.lower()
                            # matching_dates = await self.filter_dates_by_criteria(criteria)
                            matching_dates = await execute_with_retry(self.filter_dates_by_criteria, criteria)

                            if not matching_dates:
                                # No dates matched
                                response = f"No appointments found, to re-confirm please use a specific date such as '12th October 2024'"
                        except Exception as e:
                            return f"This went wrong {e}"
                    # Query appointments for the matched dates
                        stmt = select(
                            Appointment.user_name,
                            Appointment.phone_number,
                            Appointment.appointment_time,
                            Appointment.appointment_date,
                            Appointment.id
                        ).where(
                            Appointment.appointment_date.in_(matching_dates)
                        )

                        # result = await self.session.execute(
                        #     stmt)
                        result = await execute_with_retry(self.session.execute, stmt)
                        result = result.all()  # Returns list of tuples
                        if not result:
                            return f"No appointments found, to be sure use a date in this format:'12th May 2025'\n or tomorrow, this week, next week, tomorrow, next week, next two weeks, this month, next month.\n If no appointments are stil found you have no appointments for that time period."
                        response_string += f"---{criteria.upper()}---\n"
                        for user_name, phone, time, date, id in result:
                            response_string += f"appointment scheduled with {
                                user_name} |\t {user_name}'s phone number: +{phone} |\t at {date} {time} | appointment_id is {id}\n\n"
                        response = response + response_string
                    except Exception as e:
                        logger.error(
                            f"Error occured while trying to read appointment: {e}")
                        return f"There was an error in trying to read appointments: {e}"
                return response
            else:
                return "Please mention a date or dates you would like to read the appointments for"
        else:
            return "You are not allowed to carry out this action"


if __name__ == "__main__":
    async def main():
        a = time.time()
        async with AsyncSession(engine) as session:
            read_app = ReadAppointments(
                "Show me my appointments for this week", session)
            result = await read_app.get_appointments_by_criteria('2349094540644')
            print(result)
        b = time.time()
        print(b-a)

    asyncio.run(main())
