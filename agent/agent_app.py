"""Multi-Agent Conference Program Committee Review Simulation on Flower Agent.

Simulates an academic conference Program Committee (PC) reviewing a submission:
1. Ingestion: Ingests an arXiv submission (URL, ID, or raw text) via direct API or connectors.
2. Coordinator: Extracts a neutral, factual 9-point Paper Dossier.
3. Reviewers: Three expert PC reviewers evaluate the paper concurrently in parallel
   through distinct scientific lenses, providing Positives, Negatives, Q1–Q5 multiple-choice
   ratings, and final recommendations.
4. PC Chair: Synthesizes reviews, compiles an Evaluation Scorecard Matrix across
   Q1–Q5, adjudicates disagreements, and issues the official conference decision.
5. Output: Every agent outputs clean, human-readable academic text directly to the
   terminal stdout (no log files are saved to disk).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

from agent.prompts import (
    CHAIR_PROMPT,
    COORDINATOR_PROMPT,
    REVIEWER_1_PROMPT,
    REVIEWER_2_PROMPT,
    REVIEWER_3_PROMPT,
)

# ===========================================================================
# 1. Configuration & Constants
# ===========================================================================

# Default model (change to "openai/gpt-5.6-sol" or "flower-endeavor" as needed)
MODEL: str = os.environ.get("FLWR_MODEL", "flower-endeavor")
TOOL_REFS: tuple[str, ...] = ("web_search", "web_fetch")
MAX_TOOL_TURNS: int = 3

# Reviewer specification registry: (short_id, display_title, system_prompt)
REVIEWER_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "reviewer_1",
        "Reviewer 1 (Technical Soundness & Rigour)",
        REVIEWER_1_PROMPT,
    ),
    (
        "reviewer_2",
        "Reviewer 2 (Novelty & Conceptual Innovation)",
        REVIEWER_2_PROMPT,
    ),
    (
        "reviewer_3",
        "Reviewer 3 (Empirical Validation & Practical Relevance)",
        REVIEWER_3_PROMPT,
    ),
)

app = AgentApp()


# ===========================================================================
# 2. Client & Session Helpers
# ===========================================================================


def create_client() -> OpenAI:
    """Instantiate an OpenAI SDK client pointed at Flower runtime or custom endpoint."""
    base_url = os.environ.get("FLWR_RUNTIME_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    api_key = (
        os.environ.get("FLWR_RUNTIME_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "flower-runtime-key"
    )

    kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def message_text(content: Any) -> str:
    """Normalize Responses message content (Pydantic objects, dicts, strings) to plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                val = part.get("text") or part.get("refusal") or ""
                parts.append(str(val))
            elif hasattr(part, "text") and part.text:
                parts.append(str(part.text))
            elif hasattr(part, "refusal") and part.refusal:
                parts.append(str(part.refusal))
            elif hasattr(part, "to_dict"):
                d = part.to_dict()
                val = d.get("text") or d.get("refusal") or ""
                parts.append(str(val))
            else:
                parts.append(str(part))
        return "\n".join(parts)

    return str(content)


def conversation_messages(agent: AgentSession) -> list[dict[str, Any]]:
    """Rebuild completed user and assistant messages from the run-series event trace."""
    run_order: list[int] = []
    turns_by_run: dict[int, list[dict[str, Any]]] = {}
    assistant_parts_by_run: dict[int, list[str]] = {}

    for entry in agent.events.get_trace():
        run_id = entry.get("run_id")
        event_type = entry.get("event")
        data = entry.get("data")
        if not isinstance(run_id, int) or not isinstance(data, dict):
            continue

        if event_type == "message" and data.get("role") == "user":
            assistant_parts_by_run.pop(run_id, None)
            if run_id not in turns_by_run:
                run_order.append(run_id)
            turns_by_run[run_id] = [
                {
                    "type": "message",
                    "role": "user",
                    "content": message_text(data.get("content")),
                }
            ]
        elif event_type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            delta = data.get("delta")
            if isinstance(delta, str):
                assistant_parts_by_run.setdefault(run_id, []).append(delta)
        elif event_type == "response.completed":
            parts = assistant_parts_by_run.pop(run_id, [])
            turn = turns_by_run.get(run_id)
            if parts and turn is not None:
                turn.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "".join(parts),
                    }
                )
        elif event_type in {"error", "response.failed", "response.incomplete"}:
            assistant_parts_by_run.pop(run_id, None)

    return [
        message for run_id in run_order for message in turns_by_run[run_id]
    ]


def connector_error_output(
    tool_call: dict[str, Any], exc: Exception
) -> dict[str, Any]:
    """Return an error item formatted for the model's next tool-loop turn."""
    return {
        "type": "function_call_output",
        "call_id": tool_call.get("call_id", "unknown"),
        "output": json.dumps({"error": str(exc)}),
    }


# ===========================================================================
# 3. Model Interaction Functions
# ===========================================================================


def call_model_no_stream(
    client: OpenAI,
    input_items: list[dict[str, Any]],
    instructions: str,
) -> str:
    """Execute a non-streaming model call and return the generated text."""
    response = client.responses.create(
        model=MODEL,
        input=input_items,
        instructions=instructions,
        stream=False,
    )
    parts: list[str] = []
    for item in response.output:
        if hasattr(item, "text") and item.text:
            parts.append(str(item.text))
        elif hasattr(item, "content") and item.content:
            parts.append(message_text(item.content))
    return "".join(parts) if parts else str(response.output)


def stream_and_collect(
    client: OpenAI,
    agent: AgentSession,
    input_items: list[dict[str, Any]],
    instructions: str,
) -> str:
    """Stream model response, publish events to frontend client, and return complete text."""
    stream = client.responses.create(
        model=MODEL,
        input=input_items,
        instructions=instructions,
        stream=True,
    )

    output_text: list[str] = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed", "response.incomplete"}:
            raise RuntimeError(f"Model response did not complete: {event}")
        if event.type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            output_text.append(event.delta)

    return "".join(output_text)


# ===========================================================================
# 4. Review Scorecard & Parsing Helpers
# ===========================================================================


def parse_review_to_json(
    reviewer_id: str,
    reviewer_title: str,
    raw_review: str,
) -> dict[str, Any]:
    """Parse raw review text into structured Q1-Q5 ratings, scores, and recommendations."""
    scores: dict[str, Any] = {}
    for i in range(1, 6):
        pattern = rf"\*\*Question {i}:.*?\n(?:.*?\n)*?\*Selected Rating\*:\s*\[?(\d)\]?\s*[-–:]?\s*(.*?)(?:\n|$)"
        match = re.search(pattern, raw_review, re.IGNORECASE)
        if match:
            scores[f"q{i}"] = {
                "question_num": i,
                "score": int(match.group(1)),
                "label": match.group(2).strip(),
            }
        else:
            fallback = re.search(
                rf"Question {i}.*?\[?(\d)\]", raw_review, re.IGNORECASE
            )
            score_val = int(fallback.group(1)) if fallback else None
            scores[f"q{i}"] = {
                "question_num": i,
                "score": score_val,
                "label": "Selected" if score_val else "Unparsed",
            }

    valid_scores = [
        v["score"] for v in scores.values() if v["score"] is not None
    ]
    avg_score = (
        round(sum(valid_scores) / len(valid_scores), 2)
        if valid_scores
        else None
    )

    rec_match = re.search(
        r"\*Selected Recommendation\*:\s*\[?(Strong Accept|Accept|Weak Accept|Borderline|Weak Reject|Reject|Strong Reject)\]?",
        raw_review,
        re.IGNORECASE,
    )
    recommendation = rec_match.group(1).title() if rec_match else "Unspecified"

    return {
        "reviewer_id": reviewer_id,
        "reviewer_title": reviewer_title,
        "scores": scores,
        "average_score": avg_score,
        "recommendation": recommendation,
        "full_review_text": raw_review,
    }


# ===========================================================================
# 5. Paper Ingestion & ArXiv Extraction
# ===========================================================================


def extract_arxiv_id(text: str) -> str | None:
    """Extract an arXiv ID from a URL or raw string."""
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", text)
    return match.group(1) if match else None


def is_url(text: str) -> bool:
    """Check if the string is an HTTP/HTTPS URL."""
    return text.strip().startswith(("http://", "https://", "www."))


def fetch_paper_content(
    client: OpenAI,
    agent: AgentSession,
    paper_input: str,
) -> str:
    """Fetch academic paper metadata & abstract via connectors or direct fallback."""
    arxiv_id = extract_arxiv_id(paper_input)
    target_url = (
        f"https://arxiv.org/abs/{arxiv_id}"
        if arxiv_id
        else (paper_input if is_url(paper_input) else None)
    )

    if not target_url:
        return paper_input

    try:
        tools = agent.connectors.tools(TOOL_REFS)
    except Exception as exc:
        print(f"   ⚠️ Connectors unavailable: {exc}. Using paper input directly.")
        return paper_input

    try:
        input_items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "user",
                "content": (
                    f"Please retrieve the title, authors, and abstract of the paper from: {target_url}\n"
                    "Use web_fetch to read the page content."
                ),
            }
        ]

        for _ in range(MAX_TOOL_TURNS):
            response = client.responses.create(
                model=MODEL,
                input=input_items,
                tools=tools,
                stream=False,
            )

            tool_calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]
            if not tool_calls:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        return message_text(item.content)
                    if hasattr(item, "text") and item.text:
                        return str(item.text)
                break

            response_output = [item.to_dict() for item in response.output]
            function_outputs: list[dict[str, Any]] = []
            for tc in tool_calls:
                tc_dict = tc.to_dict()
                try:
                    call_result = agent.connectors.call(tc_dict)
                    if isinstance(call_result, dict) and "output" in call_result:
                        out_str = str(call_result["output"])
                        if len(out_str) > 15000:
                            call_result["output"] = out_str[:15000] + "... [truncated]"
                    function_outputs.append(call_result)
                except (RuntimeError, ValueError) as exc:
                    function_outputs.append(connector_error_output(tc_dict, exc))

            input_items.extend(response_output)
            input_items.extend(function_outputs)

        extracted_content = call_model_no_stream(
            client,
            input_items,
            instructions="Extract the paper Title, Authors, and Abstract concisely from the fetched content.",
        )

        if extracted_content and extracted_content.strip():
            return extracted_content.strip()

    except Exception as exc:
        print(f"   ⚠️ Web fetch notice ({exc}). Proceeding with paper input as-is.")

    return paper_input


# ===========================================================================
# 6. Pipeline Phase Execution
# ===========================================================================


def run_coordinator_phase(
    client: OpenAI,
    agent: AgentSession,
    paper_input: str,
) -> tuple[str, str]:
    """Execute Coordinator phase: Ingest submission and produce the Paper Dossier."""
    print("\n" + "=" * 70)
    print(" 🎯 PHASE 1: COORDINATOR AGENT — Ingestion & Paper Dossier")
    print("=" * 70)

    # 1. Ingestion
    if is_url(paper_input) or extract_arxiv_id(paper_input):
        print(f"📥 Ingesting academic paper from input: {paper_input} ...")
        paper_content = fetch_paper_content(client, agent, paper_input)
    else:
        print("📄 Direct text input detected.")
        paper_content = paper_input

    # 2. Dossier Generation
    print("📝 Generating objective, neutral Paper Dossier...")
    coordinator_input: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": (
                f"Please analyze the following academic submission and produce a "
                f"comprehensive, neutral Paper Dossier for the Program Committee reviewers:\n\n"
                f"{paper_content}"
            ),
        }
    ]
    paper_dossier = call_model_no_stream(
        client, coordinator_input, COORDINATOR_PROMPT
    )

    # 3. Output to terminal
    print("\n" + "-" * 70)
    print(" 📄 COORDINATOR PAPER DOSSIER OUTPUT:")
    print("-" * 70)
    print(paper_dossier)
    print("-" * 70 + "\n")

    return paper_dossier, paper_content


def _evaluate_single_reviewer(
    client: OpenAI,
    r_id: str,
    r_title: str,
    r_prompt: str,
    paper_dossier: str,
    paper_content: str,
) -> tuple[str, str, str, dict[str, Any]]:
    """Worker function executed in parallel thread for an expert reviewer."""
    input_items: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": (
                f"# Academic Paper Dossier\n\n{paper_dossier}\n\n"
                f"# Full Available Paper Content\n\n{paper_content}\n\n"
                "---\n\n"
                f"You are {r_title}. Conduct a rigorous, critical conference "
                "peer-review according to the required 5-question multiple choice "
                "evaluation rubric. Be concise and punchy: provide exactly 3 key strengths, "
                "3 key weaknesses, strict 1-line rationales for Q1–Q5, and a concise "
                "1-paragraph justification instead of lengthy essays."
            ),
        }
    ]

    review_text = call_model_no_stream(client, input_items, r_prompt)
    review_data = parse_review_to_json(r_id, r_title, review_text)
    return r_id, r_title, review_text, review_data


def run_reviewers_phase(
    client: OpenAI,
    paper_dossier: str,
    paper_content: str,
) -> list[dict[str, Any]]:
    """Execute all three expert Program Committee reviewers concurrently in parallel."""
    print("=" * 70)
    print(" 👥 PHASE 2: PROGRAM COMMITTEE EXPERT REVIEWERS (PARALLEL)")
    print("=" * 70)
    print("⚡ Launching Reviewer 1, Reviewer 2, and Reviewer 3 concurrently...")

    reviewer_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=len(REVIEWER_SPECS)) as executor:
        # Submit all 3 reviewers to run concurrently
        futures = [
            executor.submit(
                _evaluate_single_reviewer,
                client,
                r_id,
                r_title,
                r_prompt,
                paper_dossier,
                paper_content,
            )
            for r_id, r_title, r_prompt in REVIEWER_SPECS
        ]

        # Collect and display results in deterministic order (Reviewer 1 -> 2 -> 3)
        for future in futures:
            r_id, r_title, review_text, review_data = future.result()
            reviewer_results.append(review_data)

            # Output human-readable report to terminal
            print("\n" + "-" * 70)
            print(f" 📋 {r_title.upper()} EVALUATION REPORT:")
            print("-" * 70)
            print(review_text)
            print("-" * 70)
            print(
                f"⭐ Average Score: {review_data.get('average_score', 'N/A')} / 5.0 | "
                f"Recommendation: {review_data.get('recommendation', 'N/A')}\n"
            )

    return reviewer_results


def run_chair_phase(
    client: OpenAI,
    agent: AgentSession,
    input_items: list[dict[str, Any]],
    paper_dossier: str,
    reviewer_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute Program Committee Chair: Synthesize reviews and issue official decision."""
    print("\n" + "=" * 70)
    print(" ⚖️  PHASE 3: PROGRAM COMMITTEE CHAIR — Decision & Meta-Review")
    print("=" * 70)

    # Format reviews compilation
    formatted_reviews = "\n\n" + "=" * 50 + "\n\n".join(
        f"### {r['reviewer_title']}\n"
        f"- **Parsed Average Score**: {r.get('average_score', 'N/A')} / 5.0\n"
        f"- **Recommendation**: {r.get('recommendation', 'N/A')}\n\n"
        f"{r['full_review_text']}"
        for r in reviewer_results
    )

    chair_input: list[dict[str, Any]] = input_items + [
        {
            "type": "message",
            "role": "user",
            "content": (
                f"# Academic Paper Dossier\n\n{paper_dossier}\n\n"
                f"# Program Committee Individual Reviews\n\n{formatted_reviews}\n\n"
                "---\n\n"
                "As the Program Committee Chair, synthesize the reviews, compile the "
                "scorecard matrix across Q1-Q5, evaluate consensus positives and negatives, "
                "adjudicate disagreements, and issue the authoritative conference DECISION: "
                "ACCEPT or REJECT with a formal decision letter."
            ),
        }
    ]

    # Stream the Chair's decision report live
    chair_verdict = stream_and_collect(client, agent, chair_input, CHAIR_PROMPT)

    decision = "ACCEPT" if "ACCEPT" in chair_verdict.upper() else "REJECT"
    match_dec = re.search(r"DECISION:\s*\[?(ACCEPT|REJECT)", chair_verdict, re.IGNORECASE)
    if match_dec:
        decision = match_dec.group(1).upper()

    chair_output = {
        "final_decision": decision,
        "decision_report": chair_verdict,
    }

    # Output to terminal
    print("\n" + "=" * 70)
    print(" 📜 OFFICIAL PROGRAM COMMITTEE DECISION LETTER")
    print("=" * 70)
    print(chair_verdict)
    print("=" * 70)
    print(f"\n🏆 Official Chair Verdict: {decision}\n")

    return chair_output


def render_scorecard_summary(
    reviewer_results: list[dict[str, Any]],
    chair_output: dict[str, Any],
) -> None:
    """Render a clean ASCII Scorecard Matrix summarizing Q1-Q5 across all reviewers."""
    print("=" * 75)
    print(" 📊 PROGRAM COMMITTEE SCORECARD SUMMARY")
    print("=" * 75)

    def get_score_str(r_dict: dict[str, Any], q: str) -> str:
        scores = r_dict.get("scores", {})
        q_data = scores.get(q, {})
        score = q_data.get("score")
        label = q_data.get("label", "")
        return f"{score} ({label})" if score is not None else "N/A"

    rev1 = reviewer_results[0] if len(reviewer_results) > 0 else {}
    rev2 = reviewer_results[1] if len(reviewer_results) > 1 else {}
    rev3 = reviewer_results[2] if len(reviewer_results) > 2 else {}

    header = f"{'Evaluation Dimension':<35} | {'Rev 1 (Rigor)':<18} | {'Rev 2 (Novelty)':<18} | {'Rev 3 (Empirical)':<18}"
    print(header)
    print("-" * len(header))

    dimensions = [
        ("q1", "Q1: Technical Content & Correctness"),
        ("q2", "Q2: Novelty & Originality"),
        ("q3", "Q3: Clarity & Presentation"),
        ("q4", "Q4: Quality of Presentation"),
        ("q5", "Q5: Relevance & Timeliness"),
    ]
    for q_key, dim_name in dimensions:
        s1 = get_score_str(rev1, q_key)
        s2 = get_score_str(rev2, q_key)
        s3 = get_score_str(rev3, q_key)
        print(f"{dim_name:<35} | {s1:<18} | {s2:<18} | {s3:<18}")

    print("-" * len(header))
    r1_avg = f"{rev1.get('average_score', 'N/A')} / 5.0"
    r2_avg = f"{rev2.get('average_score', 'N/A')} / 5.0"
    r3_avg = f"{rev3.get('average_score', 'N/A')} / 5.0"
    print(f"{'Overall Average Score':<35} | {r1_avg:<18} | {r2_avg:<18} | {r3_avg:<18}")

    r1_rec = rev1.get("recommendation", "N/A")
    r2_rec = rev2.get("recommendation", "N/A")
    r3_rec = rev3.get("recommendation", "N/A")
    print(f"{'Recommendation':<35} | {r1_rec:<18} | {r2_rec:<18} | {r3_rec:<18}")
    print("=" * len(header))

    decision = chair_output.get("final_decision", "UNKNOWN")
    icon = "✅" if decision == "ACCEPT" else "❌"
    print(f"\n 🏆 OFFICIAL FINAL DECISION: {decision} {icon}\n")


# ===========================================================================
# 7. Main Application Entrypoint
# ===========================================================================


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Orchestrate the complete academic conference Program Committee review pipeline."""
    raw_prompt = context.run_config.get("agent.input")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError("agent.input must be a non-empty string or arXiv URL")

    paper_input = raw_prompt.strip()
    client = create_client()
    input_items = conversation_messages(agent)

    print("\n" + "=" * 75)
    print(" 🏛️  ACADEMIC CONFERENCE PROGRAM COMMITTEE REVIEW SIMULATION")
    print(f" 📄 Submission: {paper_input}")
    print("=" * 75)

    # Step 1: Coordinator Agent (Ingest & Dossier)
    paper_dossier, paper_content = run_coordinator_phase(
        client, agent, paper_input
    )

    # Step 2: Three Expert Reviewers (Evaluate, Score & Log to Terminal)
    reviewer_results = run_reviewers_phase(
        client, paper_dossier, paper_content
    )

    # Step 3: PC Chair (Synthesize, Scorecard & Final Decision Letter)
    chair_output = run_chair_phase(
        client, agent, input_items, paper_dossier, reviewer_results
    )

    # Step 4: Summary Table
    render_scorecard_summary(reviewer_results, chair_output)

    print("=" * 75)
    print(" 🏁 PROGRAM COMMITTEE SIMULATION COMPLETED")
    print("=" * 75 + "\n")


# ===========================================================================
# 8. Standalone Local Execution
# ===========================================================================

if __name__ == "__main__":
    import argparse

    class _StandaloneEvents:
        def __init__(self) -> None:
            self.trace: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.trace.append(event)
            if event.get("type") == "response.output_text.delta":
                print(event.get("delta", ""), end="", flush=True)

        def get_trace(self) -> list[dict[str, Any]]:
            return self.trace

    class _StandaloneConnectors:
        def tools(self, refs: tuple[str, ...]) -> list[dict[str, Any]]:
            return []

        def call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
            return {}

    class _StandaloneAgentSession:
        def __init__(self) -> None:
            self.events = _StandaloneEvents()
            self.connectors = _StandaloneConnectors()

    class _StandaloneContext:
        def __init__(self, run_config: dict[str, Any] | None = None) -> None:
            self.run_config = run_config or {}

    parser = argparse.ArgumentParser(
        description="Run Academic Conference Program Committee Review Simulation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="https://arxiv.org/abs/2307.09288",
        help="arXiv paper URL, ID, or raw text",
    )
    cli_args = parser.parse_args()

    standalone_ctx = _StandaloneContext(
        run_config={
            "agent.input": cli_args.input,
        }
    )
    standalone_sess = _StandaloneAgentSession()
    main(standalone_sess, standalone_ctx)
