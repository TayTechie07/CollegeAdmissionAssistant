import time
from pathlib import Path

import streamlit as st

AGENT_FILE = "agent.py"          # <- your original, unmodified file
CLI_MARKER = 'print("WElcome'    # everything from here down is the CLI loop (skipped)

st.set_page_config(
    page_title="College Assistant | RAG Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    #MainMenu, footer {visibility: hidden;}

    .hero {
        padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #4f46e5 100%);
        color: #fff; box-shadow: 0 8px 24px rgba(30,58,138,.25);
    }
    .hero h1 {margin: 0; font-size: 1.8rem; font-weight: 700; color: #fff;}
    .hero p  {margin: .3rem 0 0; opacity: .85; font-size: .95rem;}

    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: .72rem; font-weight: 600; letter-spacing: .04em;
        text-transform: uppercase; margin-right: 6px;
    }
    .b-academic {background:#dbeafe; color:#1d4ed8;}
    .b-fee      {background:#dcfce7; color:#15803d;}
    .b-general  {background:#f3e8ff; color:#7e22ce;}
    .b-time     {background:#f1f5f9; color:#475569;}

    .pipe {display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin:.3rem 0 .8rem;}
    .node {
        padding: 4px 12px; border-radius: 8px; font-size: .78rem; font-weight: 600;
        border: 1px solid #cbd5e1; color: #64748b; background: #f8fafc;
    }
    .node.on {background:#4f46e5; color:#fff; border-color:#4f46e5;}
    .arrow {color:#94a3b8; font-size:.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ load the agent
@st.cache_resource(show_spinner="Initialising agent: loading PDFs, embeddings and graph...")
def load_agent():
    source = Path(AGENT_FILE).read_text(encoding="utf-8")
    graph_part = source.split(CLI_MARKER)[0]      # keep graph, drop CLI loop
    namespace = {"__name__": "agent_module"}
    exec(compile(graph_part, AGENT_FILE, "exec"), namespace)
    return namespace["app"]


try:
    agent = load_agent()
except Exception as e:
    st.error(f"Could not load the agent from `{AGENT_FILE}`.")
    st.exception(e)
    st.stop()

# ------------------------------------------------------------------- state
if "history" not in st.session_state:
    st.session_state.history = []     # list of dicts: role, content, meta
if "stats" not in st.session_state:
    st.session_state.stats = {"academic": 0, "fee": 0, "general": 0}

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎓 College Assistant")
    st.caption("Multi-route RAG agent built with LangGraph")
    st.divider()

    programme = st.selectbox("Student programme", ["BCA", "BBA", "B.COM"])

    show_context = st.toggle("Show retrieved context", value=True)
    show_pipeline = st.toggle("Show pipeline trace", value=True)

    st.divider()
    st.markdown("**Session analytics**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acad.", st.session_state.stats["academic"])
    c2.metric("Fee", st.session_state.stats["fee"])
    c3.metric("Gen.", st.session_state.stats["general"])

    st.divider()
    st.markdown("**Architecture**")
    st.markdown(
        """
        - **LLM:** `gpt-oss-120b` via Groq  
        - **Embeddings:** `all-MiniLM-L6-v2`  
        - **Vector store:** FAISS (k=4)  
        - **Chunking:** 800 / 100 overlap  
        - **Orchestration:** LangGraph
        """
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.stats = {"academic": 0, "fee": 0, "general": 0}
        st.rerun()

# -------------------------------------------------------------------- hero
st.markdown(
    f"""
    <div class="hero">
        <h1>🎓 College Admission Assistant</h1>
        <p>Ask about admissions, eligibility, courses, fees and scholarships.
        Answers are grounded in the official handbook and fee structure PDFs.
        &nbsp;·&nbsp; Programme: <b>{programme}</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)


def pipeline_html(route=None):
    mid = {"academic": "academic_rag", "fee": "fee_rag", "general": "general"}.get(route)
    nodes = ["classifier", mid or "retrieval", "response"]
    parts = []
    for i, n in enumerate(nodes):
        on = "on" if route else ""
        parts.append(f'<span class="node {on}">{n}</span>')
        if i < len(nodes) - 1:
            parts.append('<span class="arrow">➜</span>')
    return f'<div class="pipe">{"".join(parts)}</div>'


def render_meta(meta):
    if not meta:
        return
    st.markdown(
        f'<span class="badge b-{meta["route"]}">{meta["route"]}</span>'
        f'<span class="badge b-time">⏱ {meta["latency"]:.2f}s</span>',
        unsafe_allow_html=True,
    )
    if show_pipeline:
        st.markdown(pipeline_html(meta["route"]), unsafe_allow_html=True)
    if show_context and meta["context"] and meta["context"] != "NO_RETRIEVAL_NEEDED":
        with st.expander("📄 Retrieved context (source chunks)"):
            st.code(meta["context"], language=None)


# ----------------------------------------------------------- empty-state tips
if not st.session_state.history:
    st.markdown("##### Try asking")
    suggestions = [
        "What are the eligibility criteria for admission?",
        "What is the total tuition fee for my programme?",
        "Are there any scholarships available?",
        "What are the university timings?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug{i}", use_container_width=True):
            st.session_state.pending = s
            st.rerun()

# --------------------------------------------------------------- chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(msg.get("meta"))

# ---------------------------------------------------------------- user input
user_query = st.chat_input("Ask your question here...")
if "pending" in st.session_state:
    user_query = st.session_state.pop("pending")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        route, context, answer = None, "", ""
        start = time.time()
        trace = st.empty()

        with st.status("Running agent pipeline...", expanded=False) as status:
            try:
                stream = agent.stream(
                    {"programme": programme, "messages": [("human", user_query)]},
                    stream_mode="updates",
                )
                for chunk in stream:
                    for node, update in chunk.items():
                        if not update:
                            continue
                        if node == "classifier":
                            route = update["query_type"]
                            status.update(label=f"Classified as **{route}**, retrieving...")
                            if show_pipeline:
                                trace.markdown(pipeline_html(route), unsafe_allow_html=True)
                        elif node in ("academic_rag", "fee_rag", "general"):
                            context = update["retrieved_context"]
                            status.update(label="Generating answer...")
                        elif node == "response":
                            last = update["messages"][-1]
                            answer = last[1] if isinstance(last, tuple) else last.content
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Pipeline failed", state="error")
                answer = f"⚠️ Something went wrong: `{e}`"

        trace.empty()
        latency = time.time() - start
        st.markdown(answer)

        meta = None
        if route:
            meta = {"route": route, "latency": latency, "context": context}
            st.session_state.stats[route] += 1
            render_meta(meta)

    st.session_state.history.append({"role": "assistant", "content": answer, "meta": meta})
    st.rerun()