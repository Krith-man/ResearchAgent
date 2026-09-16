# 🏛️ Autonomous Program Committee Review Simulation

[![Flower AgentApp](https://img.shields.io/badge/Flower-AgentApp_1.35%2B-0080FF.svg)](https://flower.ai)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent_Swarm-792EE5.svg)](#-system-architecture)
[![Concurrency](https://img.shields.io/badge/Reviewers-Parallel_Execution-00A67E.svg)](#-overview--workflow)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An industrial-grade, multi-agent academic peer-review simulation engineered on the **[Flower AgentApp](https://flower.ai)** framework. Given any scientific submission—supplied via arXiv URL, canonical arXiv identifier, or raw paper text—the system autonomously simulates the end-to-end deliberation of a tier-1 conference **Program Committee (PC)** (such as NeurIPS, ICML, or ICLR).

The deliberation pipeline orchestrates **5 specialized autonomous AI agents**: an ingestion **Coordinator Agent**, three blinded **Expert Reviewer Agents** operating with distinct scientific mandates in parallel, and an **Area Chair / PC Chair Agent** responsible for resolving inter-reviewer divergence and rendering the official conference decision letter (**ACCEPT ✅** or **REJECT ❌**).

All outputs, evaluation rubrics, and comparative scorecard matrices are computed in-memory and streamed live to terminal `stdout` in clean, formatted academic markdown with **zero disk clutter**.

---

## 📑 Table of Contents

- [Executive Summary & Motivation](#-executive-summary--motivation)
- [System Architecture](#-system-architecture)
- [Multi-Agent Deliberation Workflow](#-multi-agent-deliberation-workflow)
- [Program Committee Agent Specifications](#-program-committee-agent-specifications)
- [Conference Evaluation Rubric & Dimensions](#-conference-evaluation-rubric--dimensions)
- [Output & Evaluation Scorecard](#-output--evaluation-scorecard)
- [Execution Guide](#-execution-guide)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [Repository Structure](#-repository-structure)

---

## 💡 Executive Summary & Motivation

Top-tier machine learning conferences receive tens of thousands of submissions each year, placing severe operational strain on human peer-review ecosystems. Common failure modes include superficial reviews, reviewer fatigue, conflicting evaluation standards, and inconsistent meta-reviews.

This project addresses these challenges by modeling the conference review process as an adversarial, multi-agent committee:
1. **Committee Diversity**: Rather than querying a single generalist prompt (which inevitably averages out critical flaws), the system instantiates 3 distinct expert reviewer agents with orthogonal, domain-tailored mandates.
2. **Blinded Isolation**: Reviewer agents do not cross-communicate during the review phase, ensuring each evaluation remains completely independent and unpolluted by groupthink.
3. **Calibrated Likert Rubrics**: Reviewers evaluate submissions against standardized 5-point Likert questions with explicit qualitative anchor definitions, enabling strict mathematical comparability.
4. **Hierarchical Adjudication**: A dedicated Area Chair agent reconciles inter-reviewer conflicts, weighs methodological trade-offs, and issues a binding, actionable decision letter.

---

## 🏗️ System Architecture

```
                        ┌────────────────────────┐
                        │   arXiv Paper Input    │
                        │  (URL / ID / Text)     │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   Coordinator Agent    │
                        │ (Paper Summarization)  │
                        └────────────┬───────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
    ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
    │Reviewer Agent 1│       │Reviewer Agent 2│       │Reviewer Agent 3│
    │ (Rigor & Tech) │       │(Novelty/Theory)│       │(Empirical/App) │
    ├────────────────┤       ├────────────────┤       ├────────────────┤
    │• Positives     │       │• Positives     │       │• Positives     │
    │• Negatives     │       │• Negatives     │       │• Negatives     │
    │• Q1 (1-5) MC   │       │• Q1 (1-5) MC   │       │• Q1 (1-5) MC   │
    │• Q2 (1-5) MC   │       │• Q2 (1-5) MC   │       │• Q2 (1-5) MC   │
    │• Q3 (1-5) MC   │       │• Q3 (1-5) MC   │       │• Q3 (1-5) MC   │
    │• Q4 (1-5) MC   │       │• Q4 (1-5) MC   │       │• Q4 (1-5) MC   │
    │• Q5 (1-5) MC   │       │• Q5 (1-5) MC   │       │• Q5 (1-5) MC   │
    │• Final Rec MC  │       │• Final Rec MC  │       │• Final Rec MC  │
    │• Justification │       │• Justification │       │• Justification │
    └───────┬────────┘       └───────┬────────┘       └───────┬────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │  PC Chair (Area Chair) │
                        │   Decision & Verdict   │
                        ├────────────────────────┤
                        │• Scorecard Matrix      │
                        │• Consensus & Conflict  │
                        │• Final Justification   │
                        │• 🏆 ACCEPT / REJECT    │
                        └────────────────────────┘
```

---

## 🔄 Multi-Agent Deliberation Workflow

The deliberation lifecycle executes through three synchronized phases:

```
[Phase 1: Ingestion & Dossier] ──► [Phase 2: Parallel Reviewers] ──► [Phase 3: Meta-Review & Decision]
    • Coordinator Agent               • Reviewer 1 (Rigor)              • Area Chair Synthesis
    • Web Search & Fetch Tools        • Reviewer 2 (Novelty)            • Conflict Adjudication
    • 9-Point Structured Dossier      • Reviewer 3 (Empirical)          • Scorecard Summary Table
```

### Phase 1: Ingestion & Dossier Synthesis (Coordinator Agent)
1. Ingests the paper target via `--run-config 'agent.input="..."'` (accepts canonical arXiv URLs, IDs like `2307.09288`, or direct text).
2. Deploys `web_search` and `web_fetch` tools to retrieve title, authors, submission date, abstract, and core methodology sections.
3. Constructs an objective, neutral **9-point Academic Paper Dossier** (Title/Authors, Domain, Problem Statement, Method, Claims, Theoretical Foundations, Empirical Setup, Headline Results, Limitations).
4. Streams the complete dossier to the terminal.

### Phase 2: Parallel Independent Peer Reviews (Reviewers 1, 2, 3)
1. Spawns Reviewers 1, 2, and 3 concurrently using Python's `ThreadPoolExecutor`.
2. Each agent evaluates the paper independently through its specialized scientific persona with zero cross-visibility.
3. Reviewers answer 5 standardized conference evaluation questions (1–5 scale), provide 3 key strengths, 3 key weaknesses, a single-line rationale per rating, and assign a final recommendation (`Strong Accept` to `Strong Reject`).
4. Individual evaluation reports are rendered sequentially to the terminal upon thread completion.

### Phase 3: Meta-Review & Program Committee Adjudication (PC Chair Agent)
1. Ingests all three independent reviewer evaluations and extracts numeric ratings via regex parsing.
2. Synthesizes consensus merits, identifies shared points of critique, and adjudicates conflicting reviewer opinions.
3. Issues a formal, authoritative **Program Committee Decision Letter** containing the binding verdict: **ACCEPT ✅** or **REJECT ❌**, confidence rating, and actionable instructions for authors.
4. Renders a unified ASCII **Scorecard Summary Table** comparing all evaluation dimensions and averages across the committee.

---

## 👥 Program Committee Agent Specifications

| Agent Name | Persona & Perspective | Core Scientific Mandate | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Coordinator Agent** | Neutral Dossier Synthesizer | Gathers metadata, parses abstract and core methodology via tools, and constructs an objective 9-point technical brief. | 9-Point Academic Paper Dossier |
| **Reviewer 1 Agent** | Technical Soundness & Rigour | Scrutinizes mathematical proofs, theorem validity, experimental controls, baseline fairness, threat models, and error bounds. | Rigour & Soundness Evaluation Report |
| **Reviewer 2 Agent** | Novelty & Conceptual Innovation | Audits prior art, contextualizes methodological leaps, and evaluates whether the work presents genuine originality or routine derivation. | Novelty & Originality Evaluation Report |
| **Reviewer 3 Agent** | Empirical Validation & Relevance | Evaluates dataset scale, benchmark fidelity, hyperparameter ablations, statistical significance, compute costs, and real-world applicability. | Empirical & Timeliness Evaluation Report |
| **PC Chair Agent** | Program Committee Chair (Area Chair) | Aggregates individual scores, resolves inter-reviewer divergence, balances trade-offs, and authors the official conference decision letter. | Meta-Review & Decision Letter (ACCEPT / REJECT) |

---

## 📋 Conference Evaluation Rubric & Dimensions

Reviewers evaluate submissions against **5 standardized multiple-choice dimensions** calibrated to tier-1 conference criteria:

### Question 1: Technical Content & Correctness
*Rate the technical contribution of the paper, its soundness, and scientific rigour.*
* `[1] Poor`: Major technical flaws, unsound claims, or unvalidated methodology.
* `[2] Marginal`: Questionable soundness, weak theoretical support, or incomplete validation.
* `[3] Sound`: Technically correct, sound methodology with adequate validation.
* `[4] Solid`: High technical quality, rigorous methodology, and robust validation.
* `[5] Outstanding`: Exceptional technical depth, rigorous theoretical proofs, or groundbreaking scientific validity.

### Question 2: Novelty & Originality
*Rate the novelty and originality of the work presented in the paper.*
* `[1] None`: Known techniques applied without novelty or distinction.
* `[2] Minor`: Incremental modification of existing techniques or routine extension.
* `[3] Moderate`: Noticeable novelty in formulation, architecture, or application.
* `[4] Significant`: Highly novel formulation, creative methodology, or original concepts.
* `[5] Breakthrough`: Paradigm-shifting conceptual advance or disruptive contribution.

### Question 3: Clarity & Presentation
*Rate the clarity of the presentation, the organization of the paper, and the quality of writing.*
* `[1] Unacceptable`: Incomprehensible writing, severe structural chaos.
* `[2] Poor`: Difficult to read, ambiguous formulations, major structural issues.
* `[3] Adequate`: Understandable, reasonably organized, standard academic writing.
* `[4] Clear`: Well-written, logical flow, easy to follow arguments.
* `[5] Exemplary`: Exceptionally lucid, beautifully structured, masterclass in scientific communication.

### Question 4: Quality of Presentation
*Rate the paper organization, the clearness of text and figures, the completeness and accuracy of references.*
* `[1] Substandard`: Misleading figures, unreadable tables, missing critical citations.
* `[2] Below Average`: Poorly formatted figures, careless citations, messy tables.
* `[3] Acceptable`: Readable figures and tables, adequate bibliography.
* `[4] High Quality`: Crisp figures, informative captions, thorough and accurate references.
* `[5] Flawless`: Publication-ready visual design, exemplary references, flawless formatting.

### Question 5: Relevance & Timeliness
*Rate the importance of the topic addressed in the paper and its timeliness within its area of research.*
* `[1] Obsolete`: Outdated problem with negligible current interest.
* `[2] Marginal`: Niche or peripheral interest to the research community.
* `[3] Relevant`: Solid topic addressing ongoing challenges in the field.
* `[4] Timely & Important`: Highly relevant, tackles an urgent active problem.
* `[5] Critical & Pivotal`: Core open problem of immense importance to the community right now.

### Recommendation Scale
`[Strong Accept]` | `[Accept]` | `[Weak Accept]` | `[Borderline]` | `[Weak Reject]` | `[Reject]` | `[Strong Reject]`

---

## 📊 Output & Evaluation Scorecard

All agent deliberations, individual peer reviews, and the final PC Chair decision letter stream live to terminal `stdout` in formatted markdown with zero disk footprint. Upon deliberation completion, the system renders a unified comparative scorecard:

```text
===========================================================================
 📊 PROGRAM COMMITTEE SCORECARD SUMMARY
===========================================================================
Evaluation Dimension                | Rev 1 (Rigor)      | Rev 2 (Novelty)    | Rev 3 (Empirical) 
---------------------------------------------------------------------------
Q1: Technical Content & Correctness | 4 (Solid)          | 5 (Breakthrough)   | 4 (Solid)         
Q2: Novelty & Originality           | 3 (Moderate)       | 5 (Breakthrough)   | 4 (Significant)   
Q3: Clarity & Presentation          | 4 (Clear)          | 4 (Clear)          | 4 (Clear)         
Q4: Quality of Presentation         | 4 (High Quality)   | 4 (High Quality)   | 4 (High Quality)  
Q5: Relevance & Timeliness          | 5 (Critical)       | 5 (Critical)       | 5 (Critical)      
---------------------------------------------------------------------------
Overall Average Score               | 4.0 / 5.0          | 4.6 / 5.0          | 4.2 / 5.0         
Recommendation                      | Accept             | Strong Accept      | Accept            
===========================================================================

 🏆 OFFICIAL FINAL DECISION: ACCEPT ✅ (Consensus: Solid Accept)
```

---

## 🚀 Execution Guide

All runs are executed and monitored using the Flower CLI (`uv run flwr`):

### 1. Build the Flower App Bundle (FAB)
```bash
uv run flwr build
```

### 2. Login to Flower SuperGrid
```bash
uv run flwr login supergrid
```

### 3. Run Deliberation on Flower SuperGrid
Pass any arXiv paper URL, arXiv ID, or raw paper text via `--run-config`:

```bash
# Run with default paper (Llama 2):
uv run flwr run . supergrid --stream

# Run using an arXiv URL:
uv run flwr run . supergrid \
  --run-config 'agent.input="https://arxiv.org/abs/1706.03762"' \
  --stream

# Run using an arXiv ID:
uv run flwr run . supergrid \
  --run-config 'agent.input="2401.12345"' \
  --stream

# Run using direct paper text:
uv run flwr run . supergrid \
  --run-config 'agent.input="Title: ... Abstract: ... Method: ..."' \
  --stream
```

### 4. Monitor Runs & Logs
```bash
# List all active and completed runs:
uv run flwr list supergrid

# Retrieve status and metadata for a specific run:
uv run flwr list supergrid --run-id <RUN_ID>

# Stream historical terminal logs:
uv run flwr log <RUN_ID> supergrid --show
```

---

## 🔧 Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `FLWR_MODEL` | `flower-endeavor` | Model identifier for agent execution. Supported: `flower-endeavor`, `openai/gpt-5.6-sol`, `openai/gpt-4o`. |
| `FLWR_RUNTIME_API_KEY` | *(Automated)* | Injected automatically by the Flower SuperGrid execution runtime. |
| `FLWR_RUNTIME_BASE_URL`| *(Automated)* | Endpoint URL injected by Flower SuperGrid. |
| `OPENAI_API_KEY` | `flower-runtime-key` | API key utilized for local standalone execution or custom OpenAI backends. |
| `OPENAI_BASE_URL` | `None` | Optional base URL override for custom OpenAI-compatible proxies. |

---

## 📁 Repository Structure

```
ResearchAgent/
├── agent/
│   ├── __init__.py           # Package marker
│   ├── agent_app.py          # Flower AgentApp lifecycle, parallel executor & chair logic
│   └── prompts.py            # Calibrated reviewer rubrics, specialist personas & chair prompts
├── LICENSE                   # Apache 2.0 open-source license
├── pyproject.toml            # Project metadata, Flower app bundle specification & dependencies
├── README.md                 # Project documentation and execution guide
└── uv.lock                   # Deterministic, multi-platform dependency lockfile
```

---

## 📄 License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for complete details.
