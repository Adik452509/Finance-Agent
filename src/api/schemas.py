"""Request and response shapes of the backend API (validated by FastAPI / pydantic)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Me(BaseModel):
    email: str
    name: str | None
    is_admin: bool
    questions_left_today: int
    daily_quota: int


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    status: str
    citations: dict | None = None
    trace: dict | None = None
    usage: dict | None = None
    tokens: int = 0
    from_cache: bool = False
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class NewConversation(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class FeedbackIn(BaseModel):
    rating: int = Field(description="+1 or -1")
    comment: str | None = Field(default=None, max_length=2000)


class PdfSource(BaseModel):
    url: str = Field(pattern=r"^https://")
    company: str
    market: str
    doc_type: str
    fiscal_year_end: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    currency: str = Field(min_length=3, max_length=3)
    reporting_unit: str = ""


class AddCompany(BaseModel):
    ticker: str = Field(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9.\-]+$")
    pdf: PdfSource | None = None
    max_chunks: int = Field(default=250, ge=10, le=600)
    force: bool = False


class CompanyOut(BaseModel):
    ticker: str
    company: str
    doc_type: str
    fiscal_year_end: str
    filed: str = ""
    source: str = ""


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    args: dict
    status: str
    progress: str
    result: dict | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
