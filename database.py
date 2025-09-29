from sqlmodel import SQLModel, create_engine, Field
from typing import Optional

DATABASE_URL = "postgresql://postgres:postgres@db:5432/housing"

engine = create_engine(DATABASE_URL, echo=True)

class Payment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tenant: str
    amount: float
    date: str
