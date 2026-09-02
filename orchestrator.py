import logging
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
from deep_research_agent.models import (
    ResearchPlan,
    ResearchCorpus,
    ContentDraft,
    ReflectionCritique,
    RevisionResult,
    FinalReport,
    WorkflowEvent,
)
from deep_research_agent.llm import LLMClient
from deep_research_agent.search import SearchEngine
from deep_research_agent.agents.planner import PlanningAgent
from deep_research_agent.agents.researcher import ResearchAgent
from deep_research_agent.agents.generator import ContentGenerator
from deep_research_agent.agents.reflector import ReflectionAgent
from deep_research_agent.agents.reviser import RevisionAgent

logger = logging.getLogger("deep_research_agent.orchestrator")


class DeepResearchOrchestrator:
    """
    Coordinates the multi-agent Deep Research workflow:
    User Input -> Planning -> Research -> Content Generation -> Reflection -> Revision -> Final Answer
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        search_engine: Optional[SearchEngine] = None,
    ):
        self.llm = llm or LLMClient()
        self.search_engine = search_engine or SearchEngine()
        self.planner = PlanningAgent(self.llm)
        self.researcher = ResearchAgent(self.search_engine)
        self.generator = ContentGenerator(self.llm)
        self.reflector = ReflectionAgent(self.llm)
        self.reviser = RevisionAgent(self.llm, self.search_engine)

    async def run(
        self,
        topic: str,
        depth: str = "deep",
        additional_instructions: Optional[str] = None,
    ) -> FinalReport:
        """Executes the complete workflow synchronously returning the final report."""
        report = None
        async for event in self.stream_workflow(topic, depth, additional_instructions):
            if event.stage == "completed" and event.data and "final_report" in event.data:
                report = FinalReport.model_validate(event.data["final_report"])
        if not report:
            raise RuntimeError("Workflow failed to produce a final report.")
        return report

    async def stream_workflow(
        self,
        topic: str,
        depth: str = "deep",
        additional_instructions: Optional[str] = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Executes the workflow while yielding real-time events at each agent stage.
        """
        logger.info(f"Initiating Deep Research workflow for: '{topic}'")

        # 1. User Input Received
        yield WorkflowEvent(
            stage="input",
            status="completed",
            message=f"Received topic: '{topic}' with target depth: '{depth}'",
            data={"topic": topic, "depth": depth},
        )

        try:
            # 2. Planning Stage
            yield WorkflowEvent(
                stage="planning",
                status="started",
                message="Planning Agent is analyzing the objective and breaking topic into sub-tasks...",
            )
            plan = await self.planner.create_plan(topic, depth, additional_instructions)
            yield WorkflowEvent(
                stage="planning",
                status="completed",
                message=f"Planning complete: Created {len(plan.tasks)} structured research sub-tasks.",
                data={"plan": plan.model_dump()},
            )

            # 3. Research Stage
            yield WorkflowEvent(
                stage="research",
                status="started",
                message="Research Agent is querying multi-source knowledge bases and extracting evidence...",
            )

            # Define sub-progress event queue
            corpus = await self.researcher.execute_research(plan)

            yield WorkflowEvent(
                stage="research",
                status="completed",
                message=f"Research complete: {len(corpus.findings)} verified findings retained ({corpus.deduplicated_count} duplicates/noise removed).",
                data={
                    "corpus_summary": {
                        "total_raw": corpus.total_findings,
                        "retained": len(corpus.findings),
                        "deduplicated": corpus.deduplicated_count,
                    },
                    "findings": [f.model_dump() for f in corpus.findings[:10]],
                },
            )

            # 4. Content Generation Stage (Draft 1)
            yield WorkflowEvent(
                stage="generation",
                status="started",
                message="Content Generator is synthesizing research corpus into Initial Draft (Draft 1)...",
            )
            draft = await self.generator.generate_draft(plan, corpus)
            yield WorkflowEvent(
                stage="generation",
                status="completed",
                message=f"Draft 1 generated ({draft.word_count} words across {len(draft.sections)} sections).",
                data={"draft_v1": draft.model_dump()},
            )

            # 5. Reflection Stage
            yield WorkflowEvent(
                stage="reflection",
                status="started",
                message="Reflection Agent is critiquing Draft 1 for completeness, relevance, logic, and gaps...",
            )
            critique = await self.reflector.reflect_on_draft(plan, draft)
            yield WorkflowEvent(
                stage="reflection",
                status="completed",
                message=f"Reflection critique complete: Quality Score {critique.overall_score}/100 with {len(critique.weak_or_missing_points)} improvement directives.",
                data={"reflection": critique.model_dump()},
            )

            # 6. Revision Stage
            yield WorkflowEvent(
                stage="revision",
                status="started",
                message="Revision Agent is incorporating reflection critique and conducting gap-filling research...",
            )
            revision = await self.reviser.revise_content(plan, draft, critique)
            yield WorkflowEvent(
                stage="revision",
                status="completed",
                message=f"Revision complete: Integrated {revision.new_findings_incorporated} new evidence points and addressed all critiques.",
                data={"revision": revision.model_dump()},
            )

            # 7. Assemble Final Output
            sources = []
            seen_urls = set()
            for f in corpus.findings:
                if f.source_url not in seen_urls and f.source_url.startswith("http"):
                    seen_urls.add(f.source_url)
                    sources.append({"title": f.source_title, "url": f.source_url})

            final_report = FinalReport(
                topic=topic,
                plan=plan,
                corpus_summary={
                    "total_retrieved": corpus.total_findings,
                    "retained_findings": len(corpus.findings),
                    "duplicates_filtered": corpus.deduplicated_count,
                },
                draft_v1=draft,
                reflection=critique,
                revision=revision,
                final_content=revision.revised_markdown,
                sources=sources,
            )

            yield WorkflowEvent(
                stage="completed",
                status="completed",
                message="Deep Research Workflow completed successfully! Publication-ready master report generated.",
                data={"final_report": final_report.model_dump()},
            )

        except Exception as e:
            logger.exception("Error during Deep Research execution")
            yield WorkflowEvent(
                stage="error",
                status="error",
                message=f"Workflow error: {str(e)}",
                data={"error": str(e)},
            )
