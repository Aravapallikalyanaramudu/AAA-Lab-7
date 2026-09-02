import logging
from typing import List
from deep_research_agent.models import ResearchPlan, ResearchCorpus, ContentDraft, ContentSection
from deep_research_agent.llm import LLMClient

logger = logging.getLogger("deep_research_agent.generator")

GENERATION_SYSTEM_PROMPT = """You are an expert Research Synthesizer and Content Generation Agent.
Your job is to generate an initial comprehensive report (Draft 1) based on the supplied Research Plan and curated Research Findings.

Requirements for Draft 1:
1. Include an engaging Title and an Executive Summary summarizing key takeaways.
2. Structure the body into distinct sections corresponding directly to the planned research sub-tasks.
3. Synthesize the factual findings into coherent explanations, highlighting key concepts, real-world examples, and known challenges.
4. Reference research sources wherever applicable using [Source: <title>](<url>) or standard Markdown citations.
5. Conclude with a Preliminary Conclusion.
6. Present the output in clean, professional Markdown.
"""


class ContentGenerator:
    """
    Content Generator: Synthesizes researched facts into an initial structured draft report.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate_draft(self, plan: ResearchPlan, corpus: ResearchCorpus) -> ContentDraft:
        logger.info(f"Generating initial content draft for topic: '{plan.topic}' with {len(corpus.findings)} findings.")

        # Organize findings by task
        findings_by_task = {}
        for f in corpus.findings:
            findings_by_task.setdefault(f.task_id, []).append(f)

        findings_summary_blocks = []
        for task in plan.tasks:
            task_findings = findings_by_task.get(task.id, [])
            block = f"### Sub-Task [{task.id}]: {task.question}\n"
            block += f"Expected: {task.expected_information}\n"
            block += "Collected Evidence:\n"
            if not task_findings:
                block += "- (No direct findings retrieved; synthesize from related general context)\n"
            for f in task_findings[:4]:
                block += f"- [{f.source_title}] ({f.source_url}): {f.snippet}\n"
            findings_summary_blocks.append(block)

        user_prompt = f"""Topic: {plan.topic}
Objective: {plan.objective}
Target Audience: {plan.target_audience}

Research Corpus by Task:
{chr(10).join(findings_summary_blocks)}

Generate the complete initial research report (Draft 1) in Markdown format.
"""

        raw_markdown = await self.llm.generate_text(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.35,
        )

        # Parse sections and executive summary from generated markdown
        sections = self._parse_markdown_sections(raw_markdown)
        word_count = len(raw_markdown.split())

        # Extract title
        title = f"Research Report: {plan.topic}"
        lines = [l.strip() for l in raw_markdown.split("\n") if l.strip()]
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        exec_summary = "Executive summary of the preliminary findings."
        for sec in sections:
            if "executive summary" in sec.heading.lower():
                exec_summary = sec.content
                break

        return ContentDraft(
            version=1,
            title=title,
            executive_summary=exec_summary,
            sections=sections,
            conclusion="Preliminary conclusion based on initial research synthesis.",
            raw_markdown=raw_markdown,
            word_count=word_count,
        )

    def _parse_markdown_sections(self, markdown: str) -> List[ContentSection]:
        sections: List[ContentSection] = []
        current_heading = "Introduction"
        current_lines: List[str] = []

        for line in markdown.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections.append(
                        ContentSection(
                            heading=current_heading,
                            content="\n".join(current_lines).strip(),
                        )
                    )
                    current_lines = []
                current_heading = line[3:].strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                ContentSection(
                    heading=current_heading,
                    content="\n".join(current_lines).strip(),
                )
            )

        return sections
