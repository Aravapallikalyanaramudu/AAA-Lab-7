import argparse
import asyncio
import sys
import os

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from deep_research_agent.orchestrator import DeepResearchOrchestrator
from deep_research_agent.llm import LLMClient


def format_bar(score: int, length: int = 20) -> str:
    filled = int((score / 100) * length)
    return "[" + "=" * filled + " " * (length - filled) + f"] {score}%"


async def run_cli():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent: Planning + Research + Content Generation + Reflection + Revision Workflow"
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        help="Research topic or question to investigate",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=str,
        choices=["brief", "standard", "deep"],
        default="deep",
        help="Research depth level (default: deep)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="auto",
        choices=["auto", "gemini", "openai", "ollama", "smart_heuristic"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Optional path to save the final markdown report",
    )

    args = parser.parse_args()

    topic = args.topic
    if not topic:
        print("\n" + "=" * 70)
        print(">> DEEP RESEARCH AGENT: Planning + Research + Reflection")
        print("=" * 70)
        try:
            topic = input("\nEnter research topic or question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not topic:
        print("Error: A research topic is required.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f">> Objective: {topic}")
    print(f">> Depth: {args.depth.upper()} | Provider: {args.provider}")
    print("=" * 70)

    llm = LLMClient(provider=args.provider)
    orchestrator = DeepResearchOrchestrator(llm=llm)

    final_report = None

    async for event in orchestrator.stream_workflow(topic=topic, depth=args.depth):
        stage = event.stage.upper()
        status = event.status

        if stage == "PLANNING" and status == "COMPLETED":
            plan = event.data.get("plan", {})
            print(f"\n[STAGE 1: PLANNING AGENT] Created {len(plan.get('tasks', []))} Sub-Tasks:")
            for t in plan.get("tasks", []):
                print(f"   * [{t['id']}]: {t['question']}")
                print(f"     Queries: {', '.join(t.get('search_queries', []))}")

        elif stage == "RESEARCH" and status == "COMPLETED":
            summary = event.data.get("corpus_summary", {})
            print(f"\n[STAGE 2: RESEARCH AGENT] Evidence Harvested:")
            print(f"   * Total findings fetched: {summary.get('total_raw', 0)}")
            print(f"   * High-signal findings retained: {summary.get('retained', 0)}")
            print(f"   * Duplicates & noise filtered: {summary.get('deduplicated', 0)}")

        elif stage == "GENERATION" and status == "COMPLETED":
            draft = event.data.get("draft_v1", {})
            print(f"\n[STAGE 3: CONTENT GENERATOR] Draft 1 Assembled:")
            print(f"   * Title: {draft.get('title', '')}")
            print(f"   * Word Count: {draft.get('word_count', 0)} words")
            print(f"   * Sections: {len(draft.get('sections', []))}")

        elif stage == "REFLECTION" and status == "COMPLETED":
            critique = event.data.get("reflection", {})
            print(f"\n[STAGE 4: REFLECTION AGENT] Peer Critique & Gap Analysis:")
            print(f"   * Completeness: {format_bar(critique.get('completeness_score', 0))}")
            print(f"   * Relevance:    {format_bar(critique.get('relevance_score', 0))}")
            print(f"   * Logical Flow: {format_bar(critique.get('logical_score', 0))}")
            print(f"   * Consistency:  {format_bar(critique.get('consistency_score', 0))}")
            print(f"   * OVERALL:      {format_bar(critique.get('overall_score', 0))}")
            print("\n   Identified Gaps / Weaknesses:")
            for w in critique.get("weak_or_missing_points", []):
                print(f"   [-] {w}")
            print("\n   Actionable Revision Directives:")
            for a in critique.get("actionable_suggestions", []):
                print(f"   [+] {a}")

        elif stage == "REVISION" and status == "COMPLETED":
            rev = event.data.get("revision", {})
            print(f"\n[STAGE 5: REVISION AGENT] Refinement Complete:")
            print(f"   * New targeted evidence points incorporated: {rev.get('new_findings_incorporated', 0)}")
            print(f"   * Summary of changes: {rev.get('changes_summary', '')}")

        elif stage == "COMPLETED":
            final_report = event.data.get("final_report", {})
            print("\n" + "=" * 70)
            print("[STAGE 6: FINAL MASTER REPORT]")
            print("=" * 70)
            content = final_report.get("final_content", "")
            print("\n" + content[:1200] + ("\n... [truncated preview] ..." if len(content) > 1200 else ""))

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"\n[SUCCESS] Full report saved to: {args.output}")

        elif stage == "ERROR":
            print(f"\n[ERROR] Encountered: {event.message}")


if __name__ == "__main__":
    asyncio.run(run_cli())
