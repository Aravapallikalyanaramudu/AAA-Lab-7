import unittest
import asyncio
from deep_research_agent.models import (
    ResearchPlan,
    ResearchSubTask,
    ResearchFinding,
    ContentDraft,
    ReflectionCritique,
)
from deep_research_agent.llm import LLMClient
from deep_research_agent.search import SearchEngine, calculate_jaccard_similarity, calculate_relevance_score
from deep_research_agent.agents.planner import PlanningAgent
from deep_research_agent.agents.researcher import ResearchAgent
from deep_research_agent.agents.generator import ContentGenerator
from deep_research_agent.agents.reflector import ReflectionAgent
from deep_research_agent.agents.reviser import RevisionAgent
from deep_research_agent.orchestrator import DeepResearchOrchestrator


class TestDeepResearchAgentWorkflow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = LLMClient(provider="smart_heuristic")
        self.search = SearchEngine()
        self.orchestrator = DeepResearchOrchestrator(llm=self.llm, search_engine=self.search)

    def test_search_similarity_and_relevance(self):
        s1 = "Quantum computing uses quantum bits or qubits to perform calculations."
        s2 = "Quantum computing relies on quantum bits or qubits to execute computations."
        sim = calculate_jaccard_similarity(s1, s2)
        self.assertGreater(sim, 0.4)

        relevance = calculate_relevance_score("quantum computing qubits", s1)
        self.assertGreater(relevance, 0.5)

    def test_search_deduplication(self):
        f1 = ResearchFinding(
            task_id="t1",
            query="quantum",
            source_title="Source 1",
            source_url="http://a.org",
            snippet="Quantum key distribution provides information-theoretic security based on quantum mechanics principles.",
            relevance_score=0.9,
        )
        f2 = ResearchFinding(
            task_id="t1",
            query="quantum",
            source_title="Source 2",
            source_url="http://b.org",
            snippet="Quantum key distribution provides information theoretic security based on quantum mechanics principles.",
            relevance_score=0.9,
        )
        f3 = ResearchFinding(
            task_id="t1",
            query="quantum",
            source_title="Source 3",
            source_url="http://c.org",
            snippet="Post-quantum cryptography focuses on public-key algorithms secure against quantum attacks.",
            relevance_score=0.85,
        )

        deduped = self.search.deduplicate_and_filter([f1, f2, f3], similarity_threshold=0.7)
        self.assertEqual(len(deduped), 2)

    async def test_planning_agent(self):
        planner = PlanningAgent(self.llm)
        plan = await planner.create_plan(topic="Solid-State Batteries", depth="standard")
        self.assertIsInstance(plan, ResearchPlan)
        self.assertGreaterEqual(len(plan.tasks), 3)
        self.assertTrue(any("fundamentals" in t.question.lower() or "architecture" in t.question.lower() for t in plan.tasks))

    async def test_content_generator(self):
        planner = PlanningAgent(self.llm)
        plan = await planner.create_plan(topic="Autonomous Drone Delivery", depth="brief")
        researcher = ResearchAgent(self.search)
        corpus = await researcher.execute_research(plan)

        generator = ContentGenerator(self.llm)
        draft = await generator.generate_draft(plan, corpus)
        self.assertIsInstance(draft, ContentDraft)
        self.assertEqual(draft.version, 1)
        self.assertGreater(draft.word_count, 50)
        self.assertGreater(len(draft.sections), 0)

    async def test_reflection_agent(self):
        planner = PlanningAgent(self.llm)
        plan = await planner.create_plan(topic="Synthetic Biology", depth="brief")
        draft = ContentDraft(
            version=1,
            title="Initial Overview: Synthetic Biology",
            executive_summary="Summary of synthetic biology.",
            sections=[],
            conclusion="Preliminary conclusion.",
            raw_markdown="# Synthetic Biology\nBrief summary without quantitative metrics.",
            word_count=40,
        )

        reflector = ReflectionAgent(self.llm)
        critique = await reflector.reflect_on_draft(plan, draft)
        self.assertIsInstance(critique, ReflectionCritique)
        self.assertGreaterEqual(critique.overall_score, 0)
        self.assertLessEqual(critique.overall_score, 100)
        self.assertGreater(len(critique.weak_or_missing_points), 0)
        self.assertGreater(len(critique.actionable_suggestions), 0)

    async def test_full_orchestrator_stream(self):
        events = []
        async for event in self.orchestrator.stream_workflow("Next-Gen Solar Cells", depth="brief"):
            events.append(event)

        stages = [e.stage for e in events]
        self.assertIn("input", stages)
        self.assertIn("planning", stages)
        self.assertIn("research", stages)
        self.assertIn("generation", stages)
        self.assertIn("reflection", stages)
        self.assertIn("revision", stages)
        self.assertIn("completed", stages)

        completed_event = [e for e in events if e.stage == "completed"][0]
        final_report = completed_event.data["final_report"]
        self.assertIn("final_content", final_report)
        self.assertIn("Master Research Report", final_report["final_content"])


if __name__ == "__main__":
    unittest.main()
