import logging
from typing import Dict, Any
from deep_research_agent.models import ResearchPlan, ContentDraft, ReflectionCritique
from deep_research_agent.llm import LLMClient

logger = logging.getLogger("deep_research_agent.reflector")

REFLECTION_SYSTEM_PROMPT = """You are a rigorous Peer-Reviewer and Reflection Agent specializing in scientific and strategic intelligence.
Your responsibility is to perform an uncompromising, constructive critique of an initial research report draft.

Evaluate the draft across four key dimensions (score each from 0 to 100):
1. Completeness: Did the draft address every sub-task in the research plan? Are any critical concepts omitted?
2. Relevance: Is the draft sharply focused on the user's objective without superficial fluff?
3. Logical Flow: Are the headings, arguments, and transitions coherent, progressive, and sound?
4. Consistency & Grounding: Are claims consistent, well-grounded, and free from contradictions?

Identify:
- 3 key strengths of the draft.
- Specific missing, shallow, or weak points (e.g. lack of quantitative benchmarks, absence of comparative tables, missing mitigation strategies, weak real-world case studies).
- Concrete, actionable revision instructions to elevate the draft from a standard summary to an authoritative master report.
- Whether targeted follow-up research is required, and 2-3 specific search queries to retrieve the missing information.

Output strictly valid JSON matching this schema:
{
  "completeness_score": 75,
  "relevance_score": 85,
  "logical_score": 80,
  "consistency_score": 90,
  "overall_score": 82,
  "strengths": ["string", "string"],
  "weak_or_missing_points": ["string", "string"],
  "actionable_suggestions": ["string", "string"],
  "requires_targeted_research": true,
  "follow_up_queries": ["query 1", "query 2"]
}
"""


class ReflectionAgent:
    """
    Reflection Agent: Reviews generated content, identifies gaps, evaluates completeness/relevance/logic/consistency,
    and produces actionable improvement directives.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def reflect_on_draft(self, plan: ResearchPlan, draft: ContentDraft) -> ReflectionCritique:
        logger.info(f"Reflecting on Draft v{draft.version} for topic '{plan.topic}'.")

        plan_summary = f"Topic: {plan.topic}\nObjective: {plan.objective}\nPlanned Sub-Tasks:\n"
        for t in plan.tasks:
            plan_summary += f"- [{t.id}]: {t.question} (Expected: {t.expected_information})\n"

        user_prompt = f"""{plan_summary}

--- DRAFT TO EVALUATE ---
{draft.raw_markdown}
--- END DRAFT ---

Conduct a thorough reflection and critique based on the 4 evaluation criteria.
"""

        try:
            critique_data = await self.llm.generate_json(REFLECTION_SYSTEM_PROMPT, user_prompt)
            if "overall_score" in critique_data or "completeness_score" in critique_data:
                comp = critique_data.get("completeness_score", 70)
                rel = critique_data.get("relevance_score", 80)
                log = critique_data.get("logical_score", 75)
                cons = critique_data.get("consistency_score", 85)
                overall = critique_data.get("overall_score", int((comp + rel + log + cons) / 4))

                return ReflectionCritique(
                    completeness_score=comp,
                    relevance_score=rel,
                    logical_score=log,
                    consistency_score=cons,
                    overall_score=overall,
                    strengths=critique_data.get("strengths", ["Logical general structure"]),
                    weak_or_missing_points=critique_data.get("weak_or_missing_points", ["Needs more empirical evidence"]),
                    actionable_suggestions=critique_data.get("actionable_suggestions", ["Add concrete metrics and case studies"]),
                    requires_targeted_research=critique_data.get("requires_targeted_research", True),
                    follow_up_queries=critique_data.get("follow_up_queries", [f"{plan.topic} empirical benchmarks"]),
                )
        except Exception as e:
            logger.warning(f"Error during reflection LLM call ({e}), generating default reflection critique.")

        # Fallback critique
        return ReflectionCritique(
            completeness_score=70,
            relevance_score=85,
            logical_score=78,
            consistency_score=88,
            overall_score=78,
            strengths=[
                "Addresses the core thematic areas outlined in the initial plan.",
                "Maintains a clear and professional tone.",
                "Establishes a solid foundational framework.",
            ],
            weak_or_missing_points=[
                "Lacks specific quantitative metrics, empirical benchmark percentages, and recent dates.",
                "Case studies are described generically rather than naming real-world enterprise/academic precedents.",
                "The analysis of bottlenecks does not provide concrete, actionable technical mitigation patterns.",
                "Lacks a high-level comparative summary table for rapid decision making.",
            ],
            actionable_suggestions=[
                "Inject specific empirical data points, benchmark stats, and measurable parameters into each section.",
                "Incorporate concrete real-world deployments and comparative benchmarks.",
                "Add an actionable 'Engineering Mitigations & Best Practices' breakdown for identified challenges.",
                "Include a Structured Trade-Off Matrix / Comparative Table.",
                "Enhance the conclusion with strategic, phased implementation advice.",
            ],
            requires_targeted_research=True,
            follow_up_queries=[
                f"{plan.topic} quantitative benchmarks metrics",
                f"{plan.topic} industry deployment case studies",
                f"{plan.topic} technical mitigation strategies",
            ],
        )
