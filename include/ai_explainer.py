"""
include/ai_explainer.py
-----------------------
LangGraph agent that reads a GX validation summary and generates:
  1. A plain-English root cause explanation
  2. A stakeholder-friendly Slack alert message

Uses NVIDIA's OpenAI-compatible API via Airflow Variables:
  NVIDIA_API_KEY
  NVIDIA_MODEL (optional; defaults to meta/llama-3.3-70b-instruct)

Falls back to rule-based template if keys are not set (safe for demo).
"""

import json
from importlib import import_module
from typing import TypedDict
from langgraph.graph import StateGraph, END


# ── State shared across LangGraph nodes ───────────────────────────────────────

class AlertState(TypedDict):
    summary: dict
    failed_count: int
    passed_count: int
    checked_at: str
    root_cause: str
    slack_message: str


def _get_llm(temperature: float = 0):
    """Returns the NVIDIA-hosted LLM if credentials are available, else None."""
    try:
        Variable = import_module("airflow.sdk").Variable
        api_key = Variable.get("NVIDIA_API_KEY", default=None)
        model = Variable.get(
            "NVIDIA_MODEL",
            default="meta/llama-3.3-70b-instruct",
        )
    except Exception:
        api_key = model = None

    if not api_key:
        return None

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        temperature=temperature,
    )


# ── Node 1: Analyse root cause ────────────────────────────────────────────────

def analyse_node(state: AlertState) -> AlertState:
    summary      = state["summary"]
    failed_count = state["failed_count"]
    checked_at   = state["checked_at"]

    llm = _get_llm(temperature=0)

    if llm:
        from langchain_core.messages import HumanMessage
        prompt = f"""
You are a data quality analyst reviewing a pipeline validation report.

Validation ran at: {checked_at}
Failed rows: {failed_count}
Issues found per column:
{json.dumps(summary, indent=2)}

Write a short root cause analysis (3-4 sentences) explaining:
- What went wrong
- Which columns are affected
- Likely business impact
Keep it factual and non-technical.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        root_cause = response.content
    else:
        # ── Fallback: rule-based template ─────────────────────────────────
        lines = []
        for col, issues in summary.items():
            for issue in issues:
                lines.append(f"• {col}: {issue}")
        root_cause = (
            f"Validation at {checked_at} found {failed_count} failed rows.\n"
            + "\n".join(lines)
            + "\nThese rows have been quarantined and excluded from the warehouse load."
        )

    state["root_cause"] = root_cause
    return state


# ── Node 2: Compose Slack message ─────────────────────────────────────────────

def compose_node(state: AlertState) -> AlertState:
    failed_count = state["failed_count"]
    passed_count = state["passed_count"]
    root_cause   = state["root_cause"]
    checked_at   = state["checked_at"]

    llm = _get_llm(temperature=0.3)

    if llm:
        from langchain_core.messages import HumanMessage
        prompt = f"""
Convert this data quality report into a friendly Slack alert for a business stakeholder.
Keep it under 5 lines. Use emojis. Start with a status emoji.

Root cause:
{root_cause}

Passed rows: {passed_count}
Failed rows: {failed_count}
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        slack_message = response.content
    else:
        # ── Fallback Slack message ─────────────────────────────────────────
        slack_message = (
            f"🚨 *Meridian Pipeline — Data Quality Alert*\n"
            f"🕐 Run time: {checked_at}\n"
            f"✅ Clean rows loaded to warehouse: *{passed_count}*\n"
            f"❌ Rows quarantined (failed validation): *{failed_count}*\n"
            f"📋 Details:\n{state['root_cause']}\n"
            f"👉 Action: Check the quarantine folder and review the failed rows."
        )

    state["slack_message"] = slack_message
    return state


# ── Build the LangGraph ───────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AlertState)
    graph.add_node("analyse", analyse_node)
    graph.add_node("compose", compose_node)
    graph.set_entry_point("analyse")
    graph.add_edge("analyse", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


# ── Public function called from Airflow task ──────────────────────────────────

def explain_failures(gx_result: dict) -> str:
    if gx_result["failed_count"] == 0:
        return ""

    agent = build_graph()
    final_state = agent.invoke({
        "summary":       gx_result["summary"],
        "failed_count":  gx_result["failed_count"],
        "passed_count":  gx_result["passed_count"],
        "checked_at":    gx_result["checked_at"],
        "root_cause":    "",
        "slack_message": "",
    })

    print(f"[AI] Root cause:\n{final_state['root_cause']}")
    print(f"[AI] Slack message:\n{final_state['slack_message']}")

    return final_state["slack_message"]
