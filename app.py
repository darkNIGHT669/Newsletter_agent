"""
Streamlit dashboard for the Autonomous Newsletter Agent.

Lets you:
  - enter the goal
  - toggle Fully Autonomous vs Human-in-the-Loop mode
  - watch the agent's multi-step reasoning STREAM live, node by node,
    including the three critics visibly completing as they finish
    (they genuinely run concurrently — see run_agent_streaming below)
  - in HITL mode, approve or request a revision on the draft before send
  - view/download the final newsletter

Run with:  streamlit run app.py
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

import streamlit as st
from langgraph.types import Command

from agent.graph import get_graph, load_thread_state
from agent.state import initial_state

st.set_page_config(page_title="Newsletter Agent", page_icon="❖", layout="wide")

# --- Custom Styling & Premium Theme CSS ----------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* Global fonts and background styling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0f172a !important; /* Dark Slate background */
        color: #e2e8f0 !important;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
    }
    
    .stTitle h1 {
        background: linear-gradient(135deg, #6366f1, #d946ef) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        font-size: 2.85rem !important;
        margin-bottom: 0.2rem !important;
        text-shadow: 0 4px 20px rgba(99, 102, 241, 0.15) !important;
    }

    /* Subheader & Captions styling */
    .stCaption {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        margin-bottom: 1.5rem !important;
    }

    /* Custom styling for Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #6366f1 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }

    /* Main Container Cards (Expander, Metrics, Dataframes) */
    div[data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 1.25rem !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: #f8fafc !important;
        padding: 0.75rem 1rem !important;
        background-color: #1e293b !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #6366f1 !important;
    }
    
    /* Metrics panel custom layout */
    div[data-testid="metric-container"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 1.25rem !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px) !important;
        border-color: #4f46e5 !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.15) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #6366f1 !important;
        font-size: 1.85rem !important;
        margin-top: 0.25rem !important;
    }

    /* Buttons styling - Outlined & Filled gradients */
    div[data-testid="stButton"] button {
        border-radius: 10px !important;
        padding: 0.6rem 1.75rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }
    
    /* Primary buttons */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #d946ef 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #c026d3 100%) !important;
    }
    
    /* Secondary buttons */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid #475569 !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: #6366f1 !important;
        color: #f8fafc !important;
        background-color: rgba(99, 102, 241, 0.05) !important;
        transform: translateY(-1px) !important;
    }

    /* Text areas and Input containers */
    div[data-baseweb="textarea"], div[data-baseweb="input"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* Logs reasoning container */
    pre code, div.stText {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        background-color: #090d16 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
    }

    /* Newsletter Card preview container */
    .newsletter-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Pipeline badge bar layout styling */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        flex-wrap: wrap;
        gap: 0.65rem;
        padding: 0.75rem;
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.95rem;
        border-radius: 9999px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .pipeline-step.completed {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #10b981;
    }
    .pipeline-step.active {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(217, 70, 239, 0.18));
        border: 1px solid rgba(99, 102, 241, 0.45);
        color: #c084fc;
        box-shadow: 0 0 12px rgba(168, 85, 247, 0.25);
        animation: activePulse 2s infinite ease-in-out;
    }
    .pipeline-step.pending {
        background-color: rgba(148, 163, 184, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.12);
        color: #64748b;
    }
    .pipeline-arrow {
        color: #475569;
        font-weight: 700;
    }
    .pipeline-arrow.completed {
        color: #10b981;
    }
    @keyframes activePulse {
        0% { transform: scale(1); box-shadow: 0 0 10px rgba(168, 85, 247, 0.2); }
        50% { transform: scale(1.02); box-shadow: 0 0 16px rgba(168, 85, 247, 0.45); }
        100% { transform: scale(1); box-shadow: 0 0 10px rgba(168, 85, 247, 0.2); }
    }
</style>
""", unsafe_allow_html=True)

st.title("Autonomous Newsletter Agent")
st.caption("A multi-agent LangGraph system for automated research, writing, and editorial review.")

# --- session state bootstrap -------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{os.urandom(4).hex()}"
if "state" not in st.session_state:
    st.session_state.state = None
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

graph = get_graph()

# --- sidebar: configuration ---------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # Allow user/reviewer to input custom API keys for the selected provider
    if provider == "gemini":
        gemini_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
    elif provider == "openai":
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    elif provider == "anthropic":
        anthropic_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    st.caption(f"LLM provider: `{provider}` (set LLM_PROVIDER in .env to change)")

    mode = st.radio(
        "Mode",
        ["autonomous", "hitl"],
        format_func=lambda m: "Fully Autonomous" if m == "autonomous" else "Human-in-the-Loop",
    )

    goal = st.text_area(
        "Goal",
        value="Create a weekly newsletter on latest AI agent news and send it to our subscribers.",
        height=100,
    )

    run_clicked = st.button("Run Agent", type="primary", width="stretch")
    reset_clicked = st.button("Reset Session", width="stretch")

    st.divider()
    st.caption("Session persists across app restarts (checkpoints.sqlite).")
    st.text_input("Current thread ID", value=st.session_state.thread_id, disabled=True)
    with st.expander("Resume a paused session"):
        resume_id = st.text_input("Thread ID to resume", key="resume_id_input")
        resume_clicked = st.button("Load", width="stretch")

if reset_clicked:
    st.session_state.thread_id = f"session-{os.urandom(4).hex()}"
    st.session_state.state = None
    st.session_state.pending_interrupt = None
    st.rerun()

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if resume_clicked:
    if not resume_id.strip():
        st.sidebar.error("Enter a thread ID to resume.")
    else:
        values, interrupt_payload = load_thread_state(resume_id.strip())
        if values is None:
            st.sidebar.error(f"No saved session found for thread ID '{resume_id.strip()}'.")
        else:
            st.session_state.thread_id = resume_id.strip()
            st.session_state.state = values
            st.session_state.pending_interrupt = interrupt_payload
            st.rerun()


# ---------------------------------------------------------------------------
# Pipeline stage tracking — maps individual node names to a small set of
# human-facing stages so the "critique_factual / _tone / _structure /
# _aggregate" nodes all light up a single "Critique" chip rather than
# looking like four separate confusing stages.
# ---------------------------------------------------------------------------
STAGES = ["planner", "research", "draft", "critique", "review", "send"]
STAGE_LABELS = {
    "planner": "Plan",
    "research": "Research",
    "draft": "Draft",
    "critique": "Critique",
    "review": "Human Review",
    "send": "Send",
}
CRITIQUE_NODES = {"critique_factual", "critique_tone", "critique_structure", "critique_aggregate"}


def stage_for_node(node_name: str):
    if node_name in CRITIQUE_NODES:
        return "critique"
    if node_name == "human_review":
        return "review"
    if node_name in STAGES:
        return node_name
    return None  # "revise" has no chip of its own; it just loops draft back


def render_pipeline(completed_stages: set, current_stage: str, run_mode: str):
    html_parts = ['<div class="pipeline-container">']
    first = True
    
    stages_to_show = STAGES.copy()
    if run_mode == "autonomous" and "review" in stages_to_show:
        stages_to_show.remove("review")
        
    for index, s in enumerate(stages_to_show):
        label = STAGE_LABELS[s]
        
        # Add arrow if not first element
        if not first:
            arrow_class = "completed" if (s in completed_stages or s == current_stage or stages_to_show[index - 1] in completed_stages) else ""
            html_parts.append(f'<span class="pipeline-arrow {arrow_class}">➔</span>')
        first = False
        
        if s == current_stage:
            html_parts.append(f'''
                <div class="pipeline-step active">
                    <span class="step-icon">●</span>
                    <span class="step-label">{label}</span>
                </div>
            ''')
        elif s in completed_stages:
            html_parts.append(f'''
                <div class="pipeline-step completed">
                    <span class="step-icon">✔</span>
                    <span class="step-label">{label}</span>
                </div>
            ''')
        else:
            html_parts.append(f'''
                <div class="pipeline-step pending">
                    <span class="step-icon">○</span>
                    <span class="step-label">{label}</span>
                </div>
            ''')
            
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# --- rendering helpers ---------------------------------------------------
def render_logs(logs, expanded=True):
    with st.expander("Agent Reasoning Log", expanded=expanded):
        for line in logs:
            st.text(line)


def render_critic_breakdown(factual, tone, structure):
    c1, c2, c3 = st.columns(3)
    for col, label, result in [
        (c1, "Factual/Relevance", factual),
        (c2, "Tone/Clarity", tone),
        (c3, "Structure/Format", structure),
    ]:
        if result:
            col.metric(label, f"{result['score']}/10")
            col.caption(result["feedback"])
        else:
            col.metric(label, "—")
            col.caption("waiting...")


def render_trace_summary(trace: list):
    if not trace:
        return

    total_ms = sum(e["duration_ms"] for e in trace)
    known_tokens = any(e["input_tokens"] is not None for e in trace)
    total_tokens = sum((e["input_tokens"] or 0) + (e["output_tokens"] or 0) for e in trace)

    with st.expander("Cost & latency trace", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("LLM calls so far", len(trace))
        c1.metric("Total compute time", f"{total_ms:.0f} ms")
        c2.metric("Total tokens", f"{total_tokens:,}" if known_tokens else "not reported")

        critic_nodes = {"critique_factual", "critique_tone", "critique_structure"}
        critic_entries = [e for e in trace if e["node"] in critic_nodes]
        if len(critic_entries) >= 2:
            critic_sum = sum(e["duration_ms"] for e in critic_entries)
            critic_max = max(e["duration_ms"] for e in critic_entries)
            c3.metric("Saved by running critics in parallel", f"~{critic_sum - critic_max:.0f} ms")
            st.caption(
                f"The critics ran concurrently: {critic_sum:.0f} ms of combined LLM compute "
                f"time only added ~{critic_max:.0f} ms to the pipeline's wall clock, instead "
                f"of the full sum if they'd run one after another."
            )

        st.markdown("**Per-call breakdown**")
        st.dataframe(
            [
                {
                    "Node": e["node"],
                    "Duration (ms)": e["duration_ms"],
                    "Input tokens": e["input_tokens"] if e["input_tokens"] is not None else "—",
                    "Output tokens": e["output_tokens"] if e["output_tokens"] is not None else "—",
                }
                for e in trace
            ],
            width="stretch",
            hide_index=True,
        )


def render_snapshot(state: dict, run_mode: str, completed_stages: set, current_stage: str = None):
    """Renders a single frame of agent progress — used both live (during
    streaming) and for the final resting state after a run completes."""
    render_pipeline(completed_stages, current_stage, run_mode)

    if state.get("logs"):
        render_logs(state["logs"], expanded=(current_stage is not None))

    col1, col2 = st.columns([2, 1])
    with col1:
        if state.get("subject") or state.get("draft_markdown"):
            st.markdown('<div class="newsletter-card">', unsafe_allow_html=True)
            if state.get("subject"):
                st.subheader(state['subject'])
            if state.get("draft_markdown"):
                st.markdown(state["draft_markdown"])
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if state.get("critique_feedback"):
            st.metric("Blended Score", f"{state.get('critique_score', '-')}/10")
            st.info(state["critique_feedback"])
        if state.get("final_path"):
            st.success(f"Sent (simulated) — saved to `{state['final_path']}`")
            try:
                with open(state["final_path"], "r", encoding="utf-8") as f:
                    st.download_button(
                        "Download HTML", f.read(), file_name="newsletter.html", mime="text/html"
                    )
            except FileNotFoundError:
                pass

    if state.get("critique_factual") or "critique" in completed_stages or current_stage == "critique":
        st.markdown("**Critic ensemble breakdown**")
        render_critic_breakdown(
            state.get("critique_factual"), state.get("critique_tone"), state.get("critique_structure")
        )

    render_trace_summary(state.get("trace", []))


# ---------------------------------------------------------------------------
# Streaming execution core.
#
# graph.stream(..., stream_mode="updates") yields one chunk per completed
# node: {node_name: partial_state_update}. Because the three critics have no
# data dependency on each other, LangGraph actually runs them concurrently —
# their chunks arrive in real completion order (not declaration order), so
# this loop reflects genuine parallel execution, not a simulated one.
#
# We keep a locally accumulated "display_state" purely for progressive
# rendering. The canonical, reducer-correct state (needed for the download
# button, the HITL resume flow, etc.) is always re-fetched afterwards via
# graph.get_state(config).values rather than reconstructed by hand here.
# ---------------------------------------------------------------------------
def run_agent_streaming(stream_input, run_mode: str, placeholder):
    display_state = {"logs": [], "trace": []}
    completed_stages = set()

    for chunk in graph.stream(stream_input, config=config, stream_mode="updates"):
        node_name, update = next(iter(chunk.items()))

        if node_name == "__interrupt__":
            interrupt_payload = update[0].value
            return None, interrupt_payload

        update = dict(update)
        new_logs = update.pop("logs", [])
        new_trace = update.pop("trace", [])
        display_state["logs"] = display_state["logs"] + new_logs
        display_state["trace"] = display_state["trace"] + new_trace
        display_state.update(update)

        stage = stage_for_node(node_name)
        if stage:
            completed_stages.add(stage)

        with placeholder.container():
            render_snapshot(display_state, run_mode, completed_stages, current_stage=None)

    final_state = graph.get_state(config).values
    return final_state, None


# --- run agent ---------------------------------------------------
# Helper to format and display execution errors cleanly without python tracebacks
def handle_execution_error(e):
    error_msg = str(e)
    if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
        st.error("🛑 **API Rate Limit Exceeded (429)**: The model's free-tier rate limit has been reached. Please wait a moment before trying again, or upgrade your key.")
    elif "UNAVAILABLE" in error_msg or "503" in error_msg:
        st.error("🛑 **Model Unavailable (503)**: Google's API is currently experiencing high demand. Please try running the agent again in a few seconds.")
    elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower() or "credentials" in error_msg.lower():
        st.error("🛑 **Authentication Error**: Could not resolve authentication method. Please check that you entered a valid API Key in the sidebar.")
    else:
        st.error(f"🛑 **Execution Error**: {error_msg}")

if run_clicked:
    # Validate API Key before execution
    key_error = False
    provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider_name == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key or api_key == "your_google_key_here":
            st.error("🔑 **API Key Error**: Google API Key is missing. Please enter your Google API Key in the sidebar to run the agent.")
            key_error = True
    elif provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_key_here":
            st.error("🔑 **API Key Error**: OpenAI API Key is missing. Please enter your OpenAI API Key in the sidebar to run the agent.")
            key_error = True
    elif provider_name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key or api_key == "your_anthropic_key_here":
            st.error("🔑 **API Key Error**: Anthropic API Key is missing. Please enter your Anthropic API Key in the sidebar to run the agent.")
            key_error = True

    if not key_error:
        placeholder = st.empty()
        try:
            final_state, interrupt_payload = run_agent_streaming(
                initial_state(goal, mode), mode, placeholder
            )
            if interrupt_payload is not None:
                st.session_state.pending_interrupt = interrupt_payload
                st.session_state.state = graph.get_state(config).values
            else:
                st.session_state.state = final_state
                st.session_state.pending_interrupt = None
            st.rerun()
        except Exception as e:
            handle_execution_error(e)

# --- human-in-the-loop review gate ---------------------------------------------------
if st.session_state.pending_interrupt:
    st.warning("Agent paused for Human-in-the-Loop review")
    payload = st.session_state.pending_interrupt

    st.subheader(f"Draft subject: {payload['subject']}")
    st.markdown(payload["draft_markdown"])
    st.info(f"Blended score: {payload['critique_score']}/10 — {payload['critique_feedback']}")
    breakdown = payload.get("critique_breakdown")
    if breakdown:
        st.markdown("Critic ensemble breakdown")
        render_critic_breakdown(breakdown.get("factual"), breakdown.get("tone"), breakdown.get("structure"))

    feedback = st.text_area("Feedback for revision (leave blank if approving)")
    c1, c2 = st.columns(2)

    if c1.button("Approve & Send", type="primary", width="stretch"):
        placeholder = st.empty()
        try:
            final_state, interrupt_payload = run_agent_streaming(
                Command(resume={"decision": "approve"}), mode, placeholder
            )
            st.session_state.state = final_state if final_state else graph.get_state(config).values
            st.session_state.pending_interrupt = interrupt_payload
            st.rerun()
        except Exception as e:
            handle_execution_error(e)

    if c2.button("Request Revision", width="stretch"):
        placeholder = st.empty()
        try:
            final_state, interrupt_payload = run_agent_streaming(
                Command(resume={"decision": "revise", "feedback": feedback}), mode, placeholder
            )
            if interrupt_payload is not None:
                st.session_state.pending_interrupt = interrupt_payload
                st.session_state.state = graph.get_state(config).values
            else:
                st.session_state.state = final_state
                st.session_state.pending_interrupt = None
            st.rerun()
        except Exception as e:
            handle_execution_error(e)

if not st.session_state.pending_interrupt and st.session_state.state:
    final = st.session_state.state
    completed = {stage_for_node(n) for n in ["planner", "research", "draft"]} | {"critique", "send"}
    if final.get("human_decision"):
        completed.add("review")
    render_snapshot(final, mode, completed, current_stage=None)
