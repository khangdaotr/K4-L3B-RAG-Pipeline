"""Streamlit UI for the grounded EU web-accessibility RAG assistant."""

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation

load_dotenv()

st.set_page_config(page_title="EU Accessibility RAG", page_icon="♿", layout="wide")


def render_sources(sources: list[dict], retrieval_source: str) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn đã dùng ({len(sources)}) — {retrieval_source}", expanded=True):
        for index, source in enumerate(sources, start=1):
            metadata = source["metadata"]
            st.markdown(f"**[S{index}] {metadata['title']}**")
            st.caption(
                f"{metadata['source']} · {source['retrieval_method']} · "
                f"score {float(source['score']):.4f} · chunk {metadata['chunk_index']}"
            )
            if metadata.get("url"):
                st.markdown(f"[Mở nguồn công khai]({metadata['url']})")
            st.code(source["id"], language=None)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("EU Accessibility RAG")
    st.caption("Khả năng tiếp cận website và ứng dụng di động khu vực công tại EU")
    top_k = st.slider("Số chunks", 3, 10, 5)
    if st.button("Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()

st.title("Trợ lý nghiên cứu khả năng tiếp cận số EU")
st.caption("Câu trả lời chỉ dựa trên corpus; citation [S…] ánh xạ tới nguồn hiển thị bên dưới.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []), message.get("retrieval_source", "none"))

query = st.chat_input("Nhập câu hỏi về khả năng tiếp cận số tại EU…")
if query:
    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và kiểm chứng nguồn…"):
            result = generate_with_citation(query, top_k)
        st.markdown(result["answer"])
        render_sources(result["sources"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
