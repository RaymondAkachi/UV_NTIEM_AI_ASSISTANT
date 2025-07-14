from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text
from sqlalchemy import select
from .database import Base, engine
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String, nullable=False, unique=False)
    created_at = Column(TIMESTAMP(timezone=True),
                        server_default=text('now()'), nullable=False)
    phone_number = Column(String, nullable=False, unique=False)


class Appointment(Base):
    __tablename__ = 'appointments'
    id = Column(Integer, primary_key=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    user_name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)
    appointment_date = Column(String, nullable=False)
    appointment_time = Column(String, nullable=False)
    is_confirmed = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP(timezone=True),
                        server_default=text('now()'), nullable=False)


class AdminUser(Base):
    __tablename__ = 'admin_users'
    id = Column(Integer, primary_key=True, nullable=False)
    # user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    user_name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("All tables needed for applications confirmed to exist")


# Function to automatically add my number and dad's number to Admin table when created to make sure when the application is started the number's are always there
async def check_insert_admin_users():
    """
    Check if phone numbers 2349094540644 and 2348032235209 exist in admin_users table.
    If not, insert them with user_name Akachi and Dad 1.
    """

    # Admin users to check/insert
    admin_users = [
        {"phone_number": "2349094540644", "user_name": "Akachi"},
        {"phone_number": "2348032235209", "user_name": "Dad 1"}
    ]

    async with AsyncSession(engine) as session:
        try:
            for admin in admin_users:
                # Check if phone_number exists
                result = await session.execute(
                    select(AdminUser).where(
                        AdminUser.phone_number == admin["phone_number"])
                )
                existing_user = result.scalars().first()

                if existing_user:
                    print(
                        f"Phone number {admin['phone_number']} already exists with user_name {existing_user.user_name}")
                else:
                    # Insert new admin user
                    new_user = AdminUser(
                        user_name=admin["user_name"],
                        phone_number=admin["phone_number"]
                    )
                    session.add(new_user)
                    await session.commit()
                    print(
                        f"Inserted {admin['user_name']} with phone number {admin['phone_number']}")

        except Exception as e:
            print(f"Error: {e}")
            await session.rollback()


if __name__ == "__main__":
    # asyncio.run(create_tables())
    asyncio.run(check_insert_admin_users())


# async def add_admin_users():
#     try:
#         async with AsyncSession(engine) as session:
#             for user in [{'user_id': 2, 'user_name': "*******", 'phone_number': "*******"}]:
#                 new_admin_user = AdminUser(**user)
#                 session.add(new_admin_user)
#             await session.commit()
#             await session.refresh(new_admin_user)
#     except BaseException as e:
#         print(e)
# if __name__ == "__main__":
#     asyncio.run(add_admin_users())
