"""Streamlit UI for the Quang Ninh travel-guide RAG chatbot."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Quảng Ninh Explorer | Travel Guide",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject CSS via st.html (bypasses markdown renderer) ─────────────────────
# Read the external CSS file and inject it
_css_path = PROJECT_ROOT / ".streamlit" / "custom.css"
if _css_path.exists():
    _css = _css_path.read_text(encoding="utf-8")
    st.html(f"<style>{_css}</style>")

# ─── Also inject Google Font link ─────────────────────────────────────────────
st.html(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">'
)

# ─── Constants ────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    "🛥️  Gợi ý lịch trình Hạ Long 2 ngày 1 đêm cho người đi lần đầu.",
    "🏔️  Đi Yên Tử trong ngày cần chuẩn bị những gì?",
    "🍜  Đến Hạ Long nhất định phải thử những món nào?",
    "🏝️  Lịch trình Cô Tô 3 ngày 2 đêm tiết kiệm.",
    "💡  Mẹo di chuyển và tiết kiệm chi phí ở Quảng Ninh.",
]


# ─── Source renderer ──────────────────────────────────────────────────────────
def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📚 Nguồn tham khảo  ·  {len(sources)} đoạn trích"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            name = meta.get("source", meta.get("section", "Không rõ nguồn"))
            doc_type = meta.get("type", "knowledge base")
            score = src.get("score", 0.0)
            excerpt = src.get("content", "").strip()
            short = excerpt[:480] + ("…" if len(excerpt) > 480 else "")
            st.html(
                f"""
                <div class="source-card">
                  <div class="source-title">{i}. {name}
                    <span class="score-badge">độ liên quan {score:.3f}</span>
                  </div>
                  <div class="source-meta">{doc_type}</div>
                  <div class="source-excerpt">{short}</div>
                </div>
                """
            )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.html(
        """
        <div class="sb-brand">
          <div class="sb-brand-icon">⛰️</div>
          <div class="sb-brand-text">
            <h2>Quảng Ninh Explorer</h2>
            <span>Trợ lý du lịch thông minh</span>
          </div>
        </div>
        """
    )

    st.html('<p class="sb-label">Thiết lập hành trình</p>')
    top_k = st.slider(
        "Số chunks lấy về", min_value=3, max_value=10, value=5,
        help="Nhiều chunks hơn sẽ lấy thêm nguồn tham khảo cho hành trình."
    )
    st.html("<br>")

    st.html('<p class="sb-label">Khám phá Quảng Ninh</p>')
    for idx, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, key=f"sug_{idx}", use_container_width=True):
            st.session_state.pending_query = suggestion.split("  ", 1)[-1].strip()

    st.html("<br>")
    st.html(
        """
        <div class="pipeline-badge">
          <span>Travel RAG</span> · lịch trình · ẩm thực
          · văn hoá · mẹo tiết kiệm
        </div>
        """
    )


# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ─── Hero section ─────────────────────────────────────────────────────────────
st.html(
    """
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-tag">✦ QUẢNG NINH · VIỆT NAM</div>
        <h1>Chuyến đi Quảng Ninh, được lên kế hoạch thông minh</h1>
        <p>Hỏi về Hạ Long, Yên Tử, Cô Tô, Móng Cái hay Bình Liêu — nhận gợi ý hành trình,
           ẩm thực và mẹo chi phí dựa trên nguồn tham khảo rõ ràng.</p>
      </div>
    </div>
    """
)

# ─── Stat cards ───────────────────────────────────────────────────────────────
st.html(
    """
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-icon">📚</div>
        <div class="stat-value">Quảng Ninh</div>
        <div class="stat-label">Điểm đến trọng tâm</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🗺️</div>
        <div class="stat-value">Lịch trình</div>
        <div class="stat-label">Gợi ý theo nhu cầu</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🍤</div>
        <div class="stat-value">Ẩm thực</div>
        <div class="stat-label">Đặc sản địa phương</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🔖</div>
        <div class="stat-value">Có nguồn</div>
        <div class="stat-label">Thông tin kiểm chứng</div>
      </div>
    </div>
    """
)

# ─── Chat area header ─────────────────────────────────────────────────────────
st.html(
    """
    <div class="chat-header">
      <div class="chat-header-dot"></div>
      <span class="chat-header-title">Cuộc trò chuyện</span>
    </div>
    """
)

# ─── Render history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

# ─── Input handling ───────────────────────────────────────────────────────────
user_input = st.chat_input("Ví dụ: Lên lịch trình Hạ Long 2 ngày 1 đêm cho tôi")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm cẩm nang du lịch và kiểm tra nguồn…"):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Tôi chưa thể trả lời câu hỏi này.")
                sources = response.get("sources", [])
            except Exception as err:
                answer = f"⚠️ Không thể chạy pipeline: `{err}`"
                sources = []

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
