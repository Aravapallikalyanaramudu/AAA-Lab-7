import logging
from typing import List, Callable, Optional, Dict, Any
from deep_research_agent.models import ResearchPlan, ResearchFinding, ResearchCorpus
from deep_research_agent.search import SearchEngine

logger = logging.getLogger("deep_research_agent.researcher")


class ResearchAgent:
    """
    Research Agent: Executes the research plan by querying multi-source knowledge bases,
    extracting key facts, deduplicating findings, and filtering out noise.
    """

    def __init__(self, search_engine: Optional[SearchEngine] = None):
        self.search_engine = search_engine or SearchEngine()

    async def execute_research(
        self,
        plan: ResearchPlan,
        on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> ResearchCorpus:
        logger.info(f"Starting research execution across {len(plan.tasks)} sub-tasks.")
        all_raw_findings: List[ResearchFinding] = []

        for idx, task in enumerate(plan.tasks, 1):
            if on_progress:
                on_progress(
                    f"Executing Sub-Task {idx}/{len(plan.tasks)}: {task.question}",
                    {"task_id": task.id, "question": task.question, "progress": (idx / len(plan.tasks)) * 0.7},
                )

            # Gather findings for each query within this sub-task
            task_findings: List[ResearchFinding] = []
            for query in task.search_queries:
                findings = await self.search_engine.search_query(query=query, task_id=task.id)
                task_findings.extend(findings)

            all_raw_findings.extend(task_findings)

        total_raw = len(all_raw_findings)
        logger.info(f"Retrieved {total_raw} raw findings. Beginning deduplication and noise filtering.")

        if on_progress:
            on_progress(
                f"Deduplicating {total_raw} findings and ranking by relevance score...",
                {"total_raw": total_raw, "progress": 0.85},
            )

        # Apply deduplication and relevance thresholding
        curated_findings = self.search_engine.deduplicate_and_filter(
            all_raw_findings,
            similarity_threshold=0.60,
            min_relevance=0.20,
        )

        dedup_count = total_raw - len(curated_findings)
        logger.info(f"Deduplication complete. Retained {len(curated_findings)} high-signal findings (removed {dedup_count} duplicates/noise).")

        if on_progress:
            on_progress(
                f"Research complete: {len(curated_findings)} curated findings retained across {len(plan.tasks)} tasks.",
                {"retained": len(curated_findings), "duplicates_removed": dedup_count, "progress": 1.0},
            )

        return ResearchCorpus(
            topic=plan.topic,
            total_findings=total_raw,
            findings=curated_findings,
            deduplicated_count=dedup_count,
        )
