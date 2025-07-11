
import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# from database import SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db_logic.database import engine
from app.db_logic.models import User
import dateparser
from datetime import datetime, timedelta
import re
import httpx
import logging
from app.settings import settings
import asyncio
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHATSAPP_API_URL = settings.WHATSAPP_API_URL
WHATSAPP_API_AUTHORIZATION = settings.WHATSAPP_API_AUTHORIZATION
DEFAULT_ADMIN = settings.DEFAULT_ADMIN


class BookAppointment:
    def __init__(self, user_input: str) -> None:
        self.input = user_input
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["user_message"],
            template="""
You are an assistant that extracts date and time from a user's message for booking an appointment. The user's message is: {user_message}

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

3. **Output:**
   - Return the result in JSON format with the keys 'date' and 'time'.
   - Use double quotes for the JSON keys and string values.
   - Provide only the JSON output without any additional text.

**Examples:**
- Input: 'I want to book an appointment on 12/12/2024 at 2:30 PM'
  Output: {{"date": "12th December 2024", "time": "14:30"}}
- Input: 'Let’s meet tomorrow'
  Output: {{"date": null, "time": null}}
- Input: 'Appointment on October 12th 2024'
  Output: {{"date": "12th October 2024", "time": null}}
- Input: 'At 15:00'
  Output: {{"date": null, "time": "15:00"}}
- Input: 'See you on February 30th'
  Output: {{"date": null, "time": null}}
- Input: 'Book for 31st December 2024 at 11:59 PM'
  Output: {{"date": "31st December 2024", "time": "23:59"}}

Process the user's message according to these rules and provide only the JSON output with double quotes.
""")
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
        if results['date'] and results['time']:
            return results
        else:
            return False

    def process_dates(self, query: str) -> str:
        """
        Use this for date-related queries like:
        - Relative dates: "today", "next week"
        - Specific dates: "2023-10-15", "October 15th", "15/10/2023".
        """
        # Extract date phrases from the query (e.g., "on October 15th")
        date_phrase = re.search(
            r"(today|tomorrow|next week|\b\d{4}-\d{2}-\d{2}\b|[\w\s]+ \d{1,2}(st|nd|rd|th)?)", query, re.IGNORECASE)
        if not date_phrase:
            return "No date detected in the query."

        date_str = date_phrase.group(0).strip()

        # Parse the date (handles both relative and absolute formats)
        parsed_date = dateparser.parse(
            date_str,
            settings={
                'RELATIVE_BASE': datetime.now().timestamp(),  # Anchor for relative dates
                'PREFER_DAY_OF_MONTH': 'first',  # Resolve ambiguous formats like "10/11/12"
                'DATE_ORDER': 'YMD'  # Prefer year-month-day for numeric dates
            }
        )

        if not parsed_date:
            return "Date format not recognized. Try formats like '2023-10-15' or 'October 15th'."

        # Convert parsed date to YYYY-MM-DD format
        formatted_date = parsed_date.strftime("%Y-%m-%d")

        return formatted_date

    def normalize_time(self, time_str: str) -> str:
        """
        Convert natural language time expressions or 24-hour times to standardized format.
        Handles cases like:
        - "10 minutes past 3pm" → 15:10
        - "quarter to 8" → 07:45
        - "15:30" → 15:30
        - "18:00" → 6:00p.m
        - "13:00" → 1:00 p.m
        - "noon" → 12:00
        - "midnight" → 00:00
        """
        # Clean input and handle case-insensitivity
        time_str = time_str.lower().replace("a.m.", "am").replace("p.m.", "pm")

        # Handle special cases
        special_cases = {
            "noon": "12:00",
            "midnight": "00:00",
            "midday": "12:00"
        }
        if time_str in special_cases:
            return special_cases[time_str]

        # Handle 24-hour time format (e.g., 18:00, 13:00)
        match_24hr = re.match(r"^([0-1]?[0-9]|2[0-3]):([0-5][0-9])$", time_str)
        if match_24hr:
            hour = int(match_24hr.group(1))
            minute = int(match_24hr.group(2))

            # Convert to 12-hour format
            period = "a.m" if hour < 12 else "p.m"
            if hour == 0:
                hour = 12  # Midnight
            elif hour > 12:
                hour -= 12
            elif hour == 12:
                period = "p.m"  # Noon

            # Format output (e.g., "6:00p.m", "1:00 p.m")
            formatted_time = f"{hour}:{minute:02d}{period}" if hour == minute == 0 else f"{hour}:{minute:02d} {period}"
            return formatted_time

        # Handle "X minutes past/to Y" format
        match = re.match(
            r"(?:(\d+)\s*(?:mins?|minutes?)\s+)?(past|to|after|before)\s+(.+)$",
            time_str,
            re.IGNORECASE
        )

        if match:
            minutes = int(match.group(1)) if match.group(1) else 0
            direction = match.group(2).lower()
            base_time_str = match.group(3)

            # Parse base time
            base_time = dateparser.parse(
                base_time_str,
                settings={
                    'PREFER_DAY_OF_MONTH': 'first',
                    'RELATIVE_BASE': datetime.now().replace(hour=0, minute=0, second=0)
                }
            )

            if not base_time:
                raise ValueError(
                    f"Could not parse time expression: {time_str}")

            if direction in ("past", "after"):
                target_time = base_time + timedelta(minutes=minutes)
            else:  # "to", "before"
                target_time = base_time - timedelta(minutes=minutes)

            return target_time.strftime("%H:%M")

        # Handle other formats using dateparser
        parsed = dateparser.parse(
            time_str,
            settings={
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': datetime.now().replace(hour=0, minute=0, second=0),
                'PREFER_DAY_OF_MONTH': 'first',
                'RETURN_AS_TIMEZONE_AWARE': False
            }
        )

        if parsed:
            # Handle 12-hour format without AM/PM
            if "am" not in time_str and "pm" not in time_str and parsed.hour > 12:
                parsed = parsed.replace(hour=parsed.hour % 12)
            return parsed.strftime("%H:%M")

        raise ValueError(
            f"Could not parse time expression: '{time_str}' this time is invalid")

    def get_ordinal(self, n: int) -> str:
        """Convert integer to ordinal string (1st, 2nd, 3rd, etc.)"""
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    def validate_and_format_date(self, date_str: str) -> str:
        """
        Strictly convert date expressions to "10th October 2024" format.
        Rejects ambiguous inputs and requires explicit year specification.
        """
        original_input = date_str
        date_str = date_str.strip().lower()

        # Check for explicit year presence using multiple patterns
        year_pattern = r'(?:20\d{2}|\b\d{2}\b)(?![^\s.,/-]|\d)'
        year_in_input = re.search(year_pattern, date_str) is not None

        # Parse with strict settings
        parsed = dateparser.parse(
            date_str,
            settings={
                'STRICT_PARSING': True,
                'PREFER_DAY_OF_MONTH': 'first',
                'DATE_ORDER': 'DMY',  # Force day-first interpretation
                'REQUIRE_PARTS': ['day', 'month', 'year']
            }
        )

        if not parsed:
            raise ValueError(f"Cannot parse date: {
                original_input}. Use explicit format like '10th October 2024'")

        # Check numeric date ambiguity
        if re.match(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$', original_input):
            parts = re.split(r'[^0-9]', original_input)
            if len(parts) >= 2:
                try:
                    part1, part2 = int(parts[0]), int(parts[1])
                    if part1 <= 12 and part2 <= 12:
                        raise ValueError(f"Ambiguous date: {
                            original_input}. Use unambiguous format such as 10th October 2024")
                except (ValueError, IndexError):
                    pass

        # Verify year explicitly appears in input
        parsed_year = parsed.year
        year_strs_to_check = [str(parsed_year), str(parsed_year)[2:]]
        if not any(ys in original_input for ys in year_strs_to_check):
            return (f"Year not explicitly specified in: {original_input} please use date in this format 10th October 2024")

        # Format validation

        day_ordinal = self.get_ordinal(parsed.day)
        formatted = f"{day_ordinal} {parsed.strftime('%B')} {parsed.year}"

        # Verify formatting matches original input structure
        if not re.search(r'\d{1,2}(st|nd|rd|th)', original_input, re.IGNORECASE):
            raise Exception(
                "Missing ordinal suffix (e.g., 10th, 3rd) please use this format '10th October 2024'  you will have to enter the entire query again to be fully understood")

        if parsed.strftime('%B').lower() not in original_input.lower():
            raise Exception(
                "Month name mismatch please use this format '10th October 2024' you will have to enter the entire query again to be fully understood")

        return formatted

    async def send_admin_message(self, user_name, apostle, date_and_time, user_id):
        async with httpx.AsyncClient() as client:
            message_response = await client.post(str(WHATSAPP_API_URL),
                                                 headers={
                                                     "Content-Type": "application/json", "Authorization": str(WHATSAPP_API_AUTHORIZATION)},
                                                 json={"messaging_product": "whatsapp",
                                                       "recipient_type": "individual",
                                                       "to": str(DEFAULT_ADMIN),
                                                       "type": "template",
                                                       "template": {
                                                           "name": "appointment_requested",
                                                           "language": {
                                                               "code": "en_US"
                                                           },
                                                           "components": [
                                                               {
                                                                   "type": "body",
                                                                   "parameters": [
                                                                       {
                                                                           "type": "text",
                                                                           "parameter_name": "user_name",
                                                                           "text": str(user_name)
                                                                       },
                                                                       {
                                                                           "type": "text",
                                                                           "parameter_name": "apostle",
                                                                           "text": str(apostle)
                                                                       },
                                                                       {
                                                                           "type": "text",
                                                                           "parameter_name": "date_and_time",
                                                                           "text": str(date_and_time)
                                                                       },
                                                                       {
                                                                           "type": "text",
                                                                           "parameter_name": "user_id",
                                                                           "text": str(user_id)
                                                                       }
                                                                   ]
                                                               }
                                                           ]}})
            message_response.raise_for_status()
            response = message_response.json()
            return response

    async def book_appointment(self, name: str, phone_no: str) -> str | None:
        try:
            date_and_time = await self.validate_params()
            # date_and_time = {'date': "5th May 2025", "time": "23:20"}
            if date_and_time:
                async with AsyncSession(engine) as session:
                    try:
                        date = date_and_time['date']
                        time = date_and_time['time']
                        formatted_date = self.validate_and_format_date(date)
                        formatted_time = self.normalize_time(time)
                    except Exception as e:
                        user_message = f"Failed to book an appointment please use this format, '10th October 2024' for dates and '18:00' for time"
                        return user_message

                    user_id = (await session.execute(select(User.id).where(User.phone_number == str(phone_no)))).scalar_one_or_none()
                    user_message = f"Appointment request sent, proposed time is {
                        formatted_date} at {formatted_time}. You will be notified when your appointment has been approved"

                    await self.send_admin_message(
                        name, "Apostle Uche Raymond", f"{formatted_date} at {formatted_time}", str(user_id))
                    return user_message
            else:
                return "Failed to book appointment, All needed inputs were not provided please make sure you provide a date and time in these formats (10th October 2024, 11:24)"
        except Exception as e:
            logger.error(
                f"An error occured while a user was trying to book an appointment: {e}")
            return None


if __name__ == "__main__":
    book_app = BookAppointment(
        user_input="I want to book an appointment for 10th July 2025 at 18:00")
    print(asyncio.run(book_app.book_appointment("Akachi", "2349094540644")))
