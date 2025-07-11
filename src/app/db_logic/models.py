from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text
from .database import Base, engine
# from sqlalchemy.ext.asyncio import AsyncSession
# import asyncio


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
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    user_name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# asyncio.run(create_tables())


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
