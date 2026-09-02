import logging
from typing import Optional
from deep_research_agent.models import ResearchPlan, ResearchSubTask
from deep_research_agent.llm import LLMClient

logger = logging.getLogger("deep_research_agent.planner")

PLANNING_SYSTEM_PROMPT = """You are an elite Principal Research Strategist and Planning Agent.
Your role is to decompose complex research topics into systematic, rigorous, and logically sequenced research tasks.

For any given topic or question, you must:
1. Identify the core objective and target audience.
2. Break down the inquiry into 3 to 4 distinct sub-tasks covering:
   - Foundational mechanics and definitions
   - Current state-of-the-art and empirical case studies
   - Critical bottlenecks, limitations, and failure modes
   - Future trajectories, roadmaps, or strategic recommendations
3. Formulate targeted search queries for each sub-task designed to retrieve high-signal factual evidence.
4. Specify the exact empirical information required.

Output your response strictly in the following JSON structure:
{
  "topic": "string",
  "objective": "string",
  "target_audience": "string",
  "depth": "deep",
  "tasks": [
    {
      "id": "task_1",
      "question": "string",
      "rationale": "string",
      "search_queries": ["query 1", "query 2"],
      "expected_information": "string"
    }
  ]
}
"""


class PlanningAgent:
    """
    Planning Agent: Decomposes user topic/question into a rigorous step-by-step research plan.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def create_plan(self, topic: str, depth: str = "deep", additional_instructions: Optional[str] = None) -> ResearchPlan:
        logger.info(f"Planning research for topic: '{topic}' with depth: '{depth}'")

        user_prompt = f"Topic: {topic}\nDepth Level: {depth}\n"
        if additional_instructions:
            user_prompt += f"Special Instructions: {additional_instructions}\n"
        user_prompt += "\nCreate a comprehensive, step-by-step research plan."

        try:
            plan_data = await self.llm.generate_json(PLANNING_SYSTEM_PROMPT, user_prompt)
            if "tasks" in plan_data and isinstance(plan_data["tasks"], list) and len(plan_data["tasks"]) > 0:
                tasks = []
                for i, t in enumerate(plan_data["tasks"]):
                    tasks.append(
                        ResearchSubTask(
                            id=t.get("id", f"task_{i+1}"),
                            question=t.get("question", f"Sub-inquiry regarding {topic}"),
                            rationale=t.get("rationale", "Core conceptual understanding"),
                            search_queries=t.get("search_queries", [topic]),
                            expected_information=t.get("expected_information", "Empirical data and facts"),
                        )
                    )
                return ResearchPlan(
                    topic=plan_data.get("topic", topic),
                    objective=plan_data.get("objective", f"Comprehensive exploration of {topic}"),
                    target_audience=plan_data.get("target_audience", "Professional & Technical"),
                    depth=depth,
                    tasks=tasks,
                )
        except Exception as e:
            logger.warning(f"Error generating plan via LLM ({e}), falling back to default structured plan.")

        # Fallback structured plan
        return ResearchPlan(
            topic=topic,
            objective=f"Deliver an authoritative, multi-faceted investigation of {topic}",
            target_audience="Professional & Technical",
            depth=depth,
            tasks=[
                ResearchSubTask(
                    id="task_1",
                    question=f"What are the foundational principles and architecture of {topic}?",
                    rationale="Establishes baseline definitions and core mechanics.",
                    search_queries=[f"{topic} fundamentals architecture", f"how {topic} works"],
                    expected_information="Definitions, specifications, and primary workflows.",
                ),
                ResearchSubTask(
                    id="task_2",
                    question=f"What are real-world applications and empirical benchmarks for {topic}?",
                    rationale="Evaluates practical impact and industry adoption.",
                    search_queries=[f"{topic} real world applications", f"{topic} performance benchmarks"],
                    expected_information="Industry case studies, adoption metrics, and quantitative results.",
                ),
                ResearchSubTask(
                    id="task_3",
                    question=f"What are the primary technical bottlenecks, risks, and trade-offs of {topic}?",
                    rationale="Critically examines constraints to avoid superficial optimism.",
                    search_queries=[f"{topic} limitations challenges", f"{topic} security risks bottlenecks"],
                    expected_information="Failure modes, trade-offs, and regulatory hurdles.",
                ),
                ResearchSubTask(
                    id="task_4",
                    question=f"What are the future developments and strategic roadmap for {topic}?",
                    rationale="Provides forward-looking insights for decision makers.",
                    search_queries=[f"{topic} future outlook roadmap", f"{topic} emerging trends"],
                    expected_information="3-5 year forecasts, emerging breakthroughs, and standards.",
                ),
            ],
        )
