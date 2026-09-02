from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ResearchSubTask(BaseModel):
    id: str = Field(description="Unique identifier for the research task, e.g. task_1")
    question: str = Field(description="Sub-question to answer")
    rationale: str = Field(description="Why this sub-question is critical to the topic")
    search_queries: List[str] = Field(description="Search queries formulated to find answers")
    expected_information: str = Field(description="What specific data or facts are required")


class ResearchPlan(BaseModel):
    topic: str
    objective: str
    target_audience: str = "Informed General & Professional"
    depth: str = "deep"
    tasks: List[ResearchSubTask] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ResearchFinding(BaseModel):
    task_id: str
    query: str
    source_title: str
    source_url: str
    snippet: str
    relevance_score: float = 1.0
    key_facts: List[str] = Field(default_factory=list)


class ResearchCorpus(BaseModel):
    topic: str
    total_findings: int = 0
    findings: List[ResearchFinding] = Field(default_factory=list)
    deduplicated_count: int = 0


class ContentSection(BaseModel):
    heading: str
    content: str
    citations: List[str] = Field(default_factory=list)


class ContentDraft(BaseModel):
    version: int = 1
    title: str
    executive_summary: str
    sections: List[ContentSection] = Field(default_factory=list)
    conclusion: str
    raw_markdown: str
    word_count: int = 0


class ReflectionCritique(BaseModel):
    completeness_score: int = Field(ge=0, le=100, description="Score 0-100 for whether all sub-questions are answered")
    relevance_score: int = Field(ge=0, le=100, description="Score 0-100 for topical relevance and signal-to-noise")
    logical_score: int = Field(ge=0, le=100, description="Score 0-100 for coherence, transitions, and argument structure")
    consistency_score: int = Field(ge=0, le=100, description="Score 0-100 for factual consistency and absence of contradictions")
    overall_score: int = Field(ge=0, le=100, description="Overall draft quality score")
    strengths: List[str] = Field(default_factory=list, description="What the initial draft does well")
    weak_or_missing_points: List[str] = Field(default_factory=list, description="Specific gaps or shallow areas")
    actionable_suggestions: List[str] = Field(default_factory=list, description="Concrete revision directives")
    requires_targeted_research: bool = Field(default=False, description="True if vital facts/data are missing")
    follow_up_queries: List[str] = Field(default_factory=list, description="Search queries to fill the gaps")


class RevisionResult(BaseModel):
    critiques_addressed: List[str] = Field(default_factory=list)
    new_findings_incorporated: int = 0
    revised_markdown: str
    changes_summary: str
    comparison_notes: List[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    topic: str
    plan: ResearchPlan
    corpus_summary: Dict[str, Any]
    draft_v1: ContentDraft
    reflection: ReflectionCritique
    revision: RevisionResult
    final_content: str
    sources: List[Dict[str, str]] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class WorkflowEvent(BaseModel):
    stage: str  # planning, research, generation, reflection, revision, completed, error
    status: str  # started, progress, completed, error
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
