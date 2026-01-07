from typing import Optional
from pydantic import BaseModel
from enum import Enum


class ReviewStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class ReviewRequest(BaseModel):
    status: ReviewStatus
    note: Optional[str] = ""


class ReviewResponse(BaseModel):
    name: str
    status: ReviewStatus
    note: Optional[str] = ""
