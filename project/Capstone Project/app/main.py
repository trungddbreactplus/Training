import json
import re
from typing import TypedDict, List, Optional, Literal
import streamlit as st
from langgraph.graph import StateGraph, END
from config import AgentState, client, co_here
from node import node_define_task, node_router, node_cls, node_retriever


# ------------------ 6. Xây dựng đồ thị ------------------
graph = StateGraph(AgentState)
graph.add_node("define_task", node_define_task)
graph.add_node("router", node_router)
graph.add_node("ask_user", ask_user_e)
workflow.add_node("final_answer", final_answer_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {
    "retrieve": "retrieve",
    "ask_user": "ask_user",
    "final_answer": "final_answer"
})
workflow.add_edge("retrieve", "agent")
workflow.add_edge("ask_user", END)  # tạm dừng ở đây, sẽ resume sau
workflow.add_edge("final_answer", END)
app_graph = workflow.compile()


# ------------------ 7. Hàm chạy từng bước (resumable) ------------------
def run_agent_step(current_state: AgentState):
    """Chạy một bước của agent, trả về state mới và có thể cần dừng."""
    # Đặc biệt: nếu đang trong trạng thái chờ user, ta resume bằng cách set lại next_action và chạy tiếp agent
    if current_state.get("next_action") == "wait_for_user":
        # Đã có user_response, chuyển sang retrieve và chạy agent
        if current_state.get("user_response"):
            current_state["next_action"] = "retrieve"
            current_state["user_context"]["last_answer"] = current_state["user_response"]
            current_state["user_response"] = None
        else:
            raise ValueError("Waiting for user but no user_response")
    # Invoke một bước (LangGraph sẽ chạy cho đến khi gặp END hoặc ask_user node)
    # Nhưng do ask_user node trả về END, ta cần xử lý special
    for event in app_graph.stream(current_state):
        # event là dict {node_name: state}
        for node_name, node_state in event.items():
            if node_name == "ask_user":
                # Dừng lại, yêu cầu user nhập
                return node_state, True  # cần user input
            elif node_name == "final_answer":
                return node_state, False
            elif node_name == "end":
                return node_state, False
    return current_state, False


# ------------------ 8. Streamlit UI ------------------
st.set_page_config(page_title="Agent tư vấn chính sách trường học", layout="wide")
st.title("🎓 Agent tư vấn chính sách, quy định nhà trường")
st.markdown("Hỏi về học bổng, chính sách hỗ trợ, quy chế thi cử,...")

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # lưu lịch sử chat để hiển thị
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "waiting_for_user" not in st.session_state:
    st.session_state.waiting_for_user = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# Hàm reset agent
def reset_agent():
    st.session_state.agent_state = None
    st.session_state.waiting_for_user = False
    st.session_state.pending_question = ""


# Hiển thị lịch sử hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input từ user
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Thêm tin nhắn user vào lịch sử
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Nếu đang chờ user trả lời câu hỏi từ agent, thì đây là câu trả lời
    if st.session_state.waiting_for_user:
        # Gán user_response vào agent_state
        if st.session_state.agent_state:
            st.session_state.agent_state["user_response"] = user_input
            st.session_state.agent_state["next_action"] = "wait_for_user"  # để resume biết
            # Chạy tiếp agent
            with st.spinner("Đang xử lý..."):
                new_state, need_input = run_agent_step(st.session_state.agent_state)
                st.session_state.agent_state = new_state
                st.session_state.waiting_for_user = need_input
                # Nếu cần input thêm thì cập nhật pending_question
                if need_input:
                    st.session_state.pending_question = new_state.get("ask_user_question", "")
                else:
                    # Agent kết thúc, lấy câu trả lời cuối
                    final_ans = new_state.get("final_answer", "Không tìm thấy câu trả lời.")
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
                    with st.chat_message("assistant"):
                        st.markdown(final_ans)
                    reset_agent()
        else:
            reset_agent()
        st.rerun()
    else:
        # Bắt đầu agent mới
        initial_state = AgentState(
            task=user_input,
            user_id="streamlit_user",
            current_query="",
            retrieved_chunks=[],
            previous_queries=[],
            iteration=0,
            max_iterations=5,
            user_context={},
            next_action="retrieve",
            ask_user_question=None,
            final_answer=None,
            user_response=None
        )
        with st.spinner("Agent đang suy nghĩ..."):
            new_state, need_input = run_agent_step(initial_state)
            st.session_state.agent_state = new_state
            st.session_state.waiting_for_user = need_input
            if need_input:
                st.session_state.pending_question = new_state.get("ask_user_question", "")
            else:
                final_ans = new_state.get("final_answer", "Không tìm thấy câu trả lời.")
                st.session_state.messages.append({"role": "assistant", "content": final_ans})
                with st.chat_message("assistant"):
                    st.markdown(final_ans)
                reset_agent()
        st.rerun()

# Nếu đang chờ user trả lời, hiển thị câu hỏi của agent
if st.session_state.waiting_for_user and st.session_state.pending_question:
    # Hiển thị câu hỏi nếu chưa có trong messages (tránh trùng)
    already = any(msg.get("role") == "assistant" and msg.get("content") == st.session_state.pending_question for msg in
                  st.session_state.messages)
    if not already:
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.pending_question})
        with st.chat_message("assistant"):
            st.markdown(st.session_state.pending_question)
        st.rerun()
