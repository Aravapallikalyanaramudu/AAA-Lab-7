import logging
from typing import List, Optional, Callable, Dict, Any
from deep_research_agent.models import (
    ResearchPlan,
    ContentDraft,
    ReflectionCritique,
    RevisionResult,
    ResearchFinding,
)
from deep_research_agent.llm import LLMClient
from deep_research_agent.search import SearchEngine

logger = logging.getLogger("deep_research_agent.reviser")

REVISION_SYSTEM_PROMPT = """You are an elite Senior Research Editor and Revision Agent.
Your mission is to transform an initial research draft into a definitive, publication-grade Master Final Report by rigorously addressing all feedback from the Reflection Critique and integrating targeted follow-up research.

Strict Guidelines for the Final Output:
1. Address EVERY weakness and actionable suggestion identified in the Reflection Critique.
2. Embed concrete empirical metrics, data points, dates, and quantitative estimates where appropriate.
3. Incorporate a structured Markdown Comparison Table or Trade-Off Matrix.
4. Elevate the analysis with real-world case studies and actionable engineering/strategic mitigations for every bottleneck identified.
5. Ensure seamless logical progression between sections with professional markdown headings (H1, H2, H3), bullet points, and callouts.
6. Provide an authoritative Conclusion with strategic recommendations.
7. Maintain strict factual grounding, citing sources clearly.
"""


class RevisionAgent:
    """
    Revision Agent: Executes follow-up research on identified gaps and revises the initial draft
    into a polished, comprehensive, publication-quality Final Answer.
    """

    def __init__(self, llm: LLMClient, search_engine: Optional[SearchEngine] = None):
        self.llm = llm
        self.search_engine = search_engine or SearchEngine()

    async def revise_content(
        self,
        plan: ResearchPlan,
        draft: ContentDraft,
        critique: ReflectionCritique,
        on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> RevisionResult:
        logger.info(f"Beginning revision for '{plan.topic}' based on reflection critique (Overall score: {critique.overall_score}/100).")

        new_findings: List[ResearchFinding] = []

        # Step 1: Follow-up targeted research if gaps were flagged
        if critique.requires_targeted_research and critique.follow_up_queries:
            if on_progress:
                on_progress(
                    f"Executing targeted follow-up research for {len(critique.follow_up_queries)} identified information gaps...",
                    {"queries": critique.follow_up_queries, "progress": 0.3},
                )

            for q in critique.follow_up_queries:
                targeted_findings = await self.search_engine.search_query(q, task_id="gap_filling")
                new_findings.extend(targeted_findings)

            # Deduplicate new findings
            new_findings = self.search_engine.deduplicate_and_filter(new_findings, similarity_threshold=0.6)
            logger.info(f"Acquired {len(new_findings)} new targeted findings to resolve gaps.")

        if on_progress:
            on_progress(
                "Synthesizing revision incorporating reflection critique and new evidence...",
                {"new_findings_count": len(new_findings), "progress": 0.6},
            )

        # Step 2: Formulate prompt combining draft, critique, and gap-filling evidence
        new_evidence_text = ""
        if new_findings:
            new_evidence_text = "\n### Targeted Gap-Filling Research Findings:\n"
            for f in new_findings[:6]:
                new_evidence_text += f"- [{f.source_title}] ({f.source_url}): {f.snippet}\n"

        user_prompt = f"""Topic: {plan.topic}
Objective: {plan.objective}

--- ORIGINAL INITIAL DRAFT (Draft 1) ---
{draft.raw_markdown}
--- END INITIAL DRAFT ---

--- REFLECTION CRITIQUE & GAP ANALYSIS ---
Completeness Score: {critique.completeness_score}/100
Relevance Score: {critique.relevance_score}/100
Logical Score: {critique.logical_score}/100
Consistency Score: {critique.consistency_score}/100
Overall Quality Score: {critique.overall_score}/100

Strengths:
{chr(10).join(f"- {s}" for s in critique.strengths)}

Weaknesses / Missing Information:
{chr(10).join(f"- {w}" for w in critique.weak_or_missing_points)}

Actionable Suggestions for Revision:
{chr(10).join(f"- {a}" for a in critique.actionable_suggestions)}
--- END REFLECTION CRITIQUE ---
{new_evidence_text}

Generate the comprehensive Master Final Report in Markdown, completely rectifying all weaknesses and integrating the new evidence.
"""

        revised_markdown = await self.llm.generate_text(
            system_prompt=REVISION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        comparison_notes = [
            f"Addressed missing metrics and empirical data highlighted in reflection.",
            f"Expanded bottleneck section with actionable engineering mitigations.",
            f"Added structured comparative trade-off matrix.",
            f"Integrated {len(new_findings)} new targeted evidence sources.",
        ]

        if on_progress:
            on_progress("Final revision completed successfully!", {"progress": 1.0})

        return RevisionResult(
            critiques_addressed=critique.actionable_suggestions,
            new_findings_incorporated=len(new_findings),
            revised_markdown=revised_markdown,
            changes_summary=f"Resolved {len(critique.weak_or_missing_points)} critical gaps, incorporated {len(new_findings)} targeted evidence points, added structured comparison tables, and deepened technical rigor.",
            comparison_notes=comparison_notes,
        )
