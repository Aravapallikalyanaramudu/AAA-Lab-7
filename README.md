# Deep Research Agent

An AI-powered **Deep Research Agent** implementing a **Planning + Research + Reflection** workflow to generate high-quality, comprehensive, and evidence-grounded content.

---

## 🚀 Workflow Architecture

```
User Input ──► Planning Agent ──► Research Agent ──► Content Generator ──► Reflection Agent ──► Revision Agent ──► Final Master Answer
```

### Stages Breakdown

1. **User Input**
   - Receives topic/question, target audience, and research depth (`brief`, `standard`, `deep`).
   - Formulates primary research objective.

2. **Planning Agent (`deep_research_agent/agents/planner.py`)**
   - Decomposes the central topic into 3–5 structured sub-tasks.
   - For each sub-task, defines the scientific/strategic rationale, expected information, and targeted search queries.

3. **Research Stage (`deep_research_agent/agents/researcher.py` & `search.py`)**
   - Executes multi-source retrieval (Wikipedia API, DuckDuckGo API, and fallback knowledge corpora).
   - Extracts relevant factual snippets.
   - Automatically filters out duplicate snippets using Jaccard word-level similarity.
   - Computes query relevance scores and ranks high-signal findings.

4. **Content Generation (`deep_research_agent/agents/generator.py`)**
   - Synthesizes findings by task into an initial structured report (**Draft 1**).
   - Generates Executive Summary, section breakdowns, citations, and preliminary conclusions.

5. **Reflection Agent (`deep_research_agent/agents/reflector.py`)**
   - Performs a rigorous peer review across four dimensions:
     - **Completeness**: Were all sub-questions answered?
     - **Relevance**: Is the content focused and free of fluff?
     - **Logical Flow**: Are transitions and structural progressions coherent?
     - **Consistency**: Are claims grounded and free of internal contradictions?
   - Identifies specific missing data, ungrounded claims, and weak points.
   - Generates actionable revision directives and determines if follow-up targeted research is required.

6. **Revision Stage (`deep_research_agent/agents/reviser.py`)**
   - Gathers gap-filling research for follow-up queries flagged by reflection.
   - Ingests the initial draft, critique directives, and new evidence.
   - Produces the polished, data-dense **Final Master Report**.

7. **Final Output**
   - Well-structured master report with headings, quantitative metrics, comparative trade-off tables, named case studies, and citations.
   - Provides side-by-side **Draft 1 vs. Final** comparison proving how planning and reflection improve content quality.

---

## 🛠️ Quick Start

### 1. Installation
All core dependencies (`fastapi`, `uvicorn`, `httpx`, `pydantic`) are already installed.

```bash
pip install -r requirements.txt
```

### 2. Command Line Interface (CLI)
Run deep research directly in your terminal:

```bash
# Run with automatic provider detection (works offline out-of-the-box!)
python -m deep_research_agent.cli --topic "Impact of Quantum Computing on Post-Quantum Cryptography"

# Save the final master report to a file
python -m deep_research_agent.cli --topic "Solid-State Batteries for Electric Vehicles" --output report.md

# Interactive mode
python -m deep_research_agent.cli
```

### 3. Interactive Web Dashboard
Launch the real-time reactive Web UI:

```bash
python -m deep_research_agent.web.server
```
Then open your browser at **`http://localhost:8000`**.

#### Features of the Web Dashboard:
- **Live Workflow Stepper**: Real-time visual progress across all 6 agent stages.
- **Server-Sent Events (SSE)**: Streaming agent thoughts, logs, and sub-task progress.
- **Side-by-Side Diff View**: Demonstrates Draft 1 vs. Final Master Report to showcase the impact of reflection.
- **Reflection Scorecard**: Visual ratings for Completeness, Relevance, Logic, Consistency, and Overall Quality.
- **Sources & Citations Drawer**: Direct links to referenced research evidence.
- **Markdown Export**: One-click copy or `.md` file download.

---

## 🧪 Running Automated Tests

```bash
python -m unittest discover -s deep_research_agent/tests -p "test_*.py" -v
```

---

## ⚙️ Supported LLM Providers

The agent supports multiple providers via `llm.py`:
- **Smart Engine (Zero-API)**: High-fidelity built-in synthesizer requiring no API keys.
- **Google Gemini**: Set `GEMINI_API_KEY` (defaults to `gemini-2.5-flash`).
- **OpenAI / Groq / OpenRouter**: Set `OPENAI_API_KEY`.
- **Local Ollama**: Connects to `http://localhost:11434`.
