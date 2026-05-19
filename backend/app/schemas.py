from datetime import datetime
from typing import Literal

from pydantic import BaseModel


RunState = Literal["queued", "processing_uploads", "extracting_ids", "verifying", "completed", "failed"]


class RunConfig(BaseModel):
    regTransactionIdColumn: str | None = None
    useFallback: bool = True
    rrnColumn: str | None = None
    amountColumn: str | None = None


class RunSummary(BaseModel):
    totalRecords: int
    verified: int
    notVerified: int
    noIdExtracted: int
    duplicates: int
    registrationDuplicates: int
    amountMismatch: int


class RunStatus(BaseModel):
    runId: str
    status: RunState
    stage: str
    message: str
    progress: int
    createdAt: datetime
    completedAt: datetime | None = None
    summary: RunSummary | None = None
    error: str | None = None


class RunCreateResponse(BaseModel):
    runId: str
    status: RunState


class ResultsResponse(BaseModel):
    runId: str
    total: int
    page: int
    pageSize: int
    rows: list[dict]

