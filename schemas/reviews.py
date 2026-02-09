from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class SentimentLabel(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class TopicScore(BaseModel):
    topic: str
    score: float = Field(..., ge=0.0, le=1.0)


class ReviewAnalysisItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    city: Optional[str]
    text: str
    sentiment: SentimentLabel
    score: float = Field(..., ge=0.0, le=1.0)
    topics: List[TopicScore]
    createdAt: Optional[datetime]


class ReviewsAnalysisStats(BaseModel):
    total: int
    positive: int
    neutral: int
    negative: int


class ReviewsAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stats: ReviewsAnalysisStats
    detailed: List[ReviewAnalysisItem]
