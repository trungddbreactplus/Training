# app.py
import json
import re
from typing import TypedDict, List, Optional, Literal
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API = os.getenv("GROQ_API")


# ------------------ 1. Định nghĩa state ------------------
class AgentState(TypedDict):
    # === NHIỆM VỤ CHÍNH ===
    task: str
    """Câu hỏi gốc của sinh viên. Đây là mục tiêu xuyên suốt mà Agent cần trả lời.
       Không thay đổi trong suốt vòng lặp, dùng để đối chiếu khi Agent đi chệch hướng."""

    user_id: str
    """Định danh duy nhất của người dùng (sinh viên). Dùng để truy xuất bộ nhớ dài hạn từ Mem0,
       giúp Agent nhớ được hoàn cảnh, lịch sử hỏi đáp của sinh viên đó qua nhiều phiên."""

    # === BỘ NHỚ LÀM VIỆC (WORKING MEMORY) ===
    current_query: str
    """Truy vấn sẽ được dùng cho lần gọi retrieve sắp tới. Agent có thể viết lại câu hỏi gốc
       thành query ngắn gọn, chính xác hơn trước khi tìm kiếm trong vector DB."""

    retrieved_chunks: List[str]
    """Danh sách các đoạn văn bản (chunk) vừa được truy xuất từ vector DB ở bước retrieve gần nhất."""

    previous_queries: List[str]
    """Lưu lại tất cả các query đã từng dùng để retrieve. Tác dụng: tránh việc Agent lặp lại
       cùng một truy vấn vô ích, đồng thời giúp quan sát xu hướng tìm kiếm."""

    # === BỘ NHỚ ĐIỀU KHIỂN (CONTROL MEMORY) ===
    iteration: int
    """Số lần retrieve đã thực hiện. Dùng để giới hạn vòng lặp, tránh Agent chạy mãi
       khi không tìm thấy câu trả lời. Mỗi lần retrieve xong iteration tăng lên 1."""

    max_iterations: int
    """Giới hạn trên của số lần retrieve (thường đặt 5-7). Khi iteration >= max_iterations,
       Agent bắt buộc phải chuyển sang final_answer, dù chưa đủ thông tin, để kết thúc đúng lúc."""

    # === BỘ NHỚ DÀI HẠN CACHE (MEM0) ===
    user_context: dict
    """Lưu các thông tin bền vững về sinh viên trong phiên hiện tại, ví dụ:
       {"ho_canh": "ngheo", "khoa": "CNTT", "nam_hoc": 3}. Dữ liệu này được đồng bộ từ Mem0
       khi bắt đầu và sẽ được ghi ngược lại sau phiên để dùng cho lần sau."""

    # === ĐIỀU KHIỂN HÀNH ĐỘNG (ACTION & ROUTING) ===
    next_action: Literal["retrieve", "ask_user", "final_answer"]
    """Hành động tiếp theo mà Agent sẽ thực hiện. Do LLM quyết định dựa trên phân tích
       ở agent_node. Router sẽ căn cứ vào trường này để chuyển đến node phù hợp."""

    ask_user_question: Optional[str]
    """Nếu next_action là 'ask_user', nội dung ở đây chính là câu hỏi mà Agent muốn gửi
       đến sinh viên để làm rõ thông tin (ví dụ: 'Bạn đang là sinh viên năm mấy?').
       Node ask_user sẽ đọc và hiển thị câu hỏi này."""

    final_answer: Optional[str]
    """Khi next_action là 'final_answer', đây là câu trả lời hoàn chỉnh mà Agent tổng hợp
       được. final_answer_node sẽ in nội dung này ra cho người dùng và kết thúc."""

    # === TẠM GIỮ PHẢN HỒI CỦA USER ===
    user_response: Optional[str]
    """Dùng để lưu câu trả lời từ sinh viên khi Agent đã hỏi (ask_user). Sau khi user nhập,
       giá trị này được gán vào user_response, sau đó agent_node sẽ xử lý nó như một
       nguồn thông tin bổ sung (ví dụ cập nhật user_context hoặc sửa query)."""


# ------------------ 2. Giả lập vector DB ------------------
POLICY_DB = {
    "học bổng vượt khó": "Điều kiện: hộ nghèo hoặc cận nghèo, GPA >= 3.0, điểm rèn luyện >= 80, tham gia ít nhất 20h tình nguyện.",
    "học bổng khuyến khích học tập": "Điều kiện: GPA >= 3.5, điểm rèn luyện >= 85, không bị kỷ luật.",
    "học bổng tài năng trẻ": "Dành cho sinh viên có thành tích nghiên cứu khoa học hoặc giải thưởng cấp trường trở lên. GPA >= 3.2.",
    "chính sách hỗ trợ hộ nghèo": "Giảm 50% học phí, hỗ trợ 1 triệu đồng/tháng chi phí sinh hoạt.",
    "chính sách hỗ trợ hộ cận nghèo": "Giảm 30% học phí, hỗ trợ 500k/tháng.",
    "chính sách tín dụng sinh viên": "Vay tối đa 30 triệu/năm, lãi suất 4%/năm, ân hạn trả nợ 12 tháng sau khi tốt nghiệp.",
    "quy định đào tạo tín chỉ": "Mỗi học kỳ đăng ký tối thiểu 12 tín, tối đa 25 tín. Cảnh báo học vụ nếu GPA < 2.0.",
    "quy chế thi cử": "Sinh viên vi phạm quy chế thi bị đình chỉ, hủy kết quả và có thể bị kỷ luật."
}


def retrieve_from_db(query: str, k: int = 3) -> List[str]:
    query_lower = query.lower()
    scored = []
    for key, value in POLICY_DB.items():
        score = sum(1 for w in query_lower.split() if w in key or w in value.lower())
        if score > 0:
            scored.append((score, f"**{key}**: {value}"))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:k]]


# ------------------ 3. LLM ------------------
llm = ChatGroq(
    model='openai/gpt-oss-120b',  # giữ nguyên theo yêu cầu
    temperature=0,
    api_key=GROQ_API
)


# ------------------ 4. Prompt & Parse ------------------
def build_agent_prompt(state: AgentState) -> str:
    chunks_text = "\n\n".join(state["retrieved_chunks"]) if state["retrieved_chunks"] else "(Chưa có thông tin nào)"
    queries_text = ", ".join(state["previous_queries"]) if state["previous_queries"] else "chưa có"
    return f"""Bạn là trợ lý tư vấn chính sách nhà trường. Hãy dựa vào câu hỏi và thông tin đã tra cứu để quyết định bước tiếp theo.

### Câu hỏi của sinh viên:
{state["task"]}

### Thông tin về sinh viên (từ bộ nhớ dài hạn):
{json.dumps(state.get("user_context", {}), ensure_ascii=False)}

### Các bước đã thực hiện:
- Số lần tra cứu: {state["iteration"]} / {state["max_iterations"]}
- Các từ khóa đã tìm: {queries_text}

### Kết quả tra cứu gần nhất:
{chunks_text}

### Yêu cầu:
Trả lời bằng JSON duy nhất: {{"action": "retrieve/ask_user/final_answer", "content": "nội dung"}}
Không thêm text ngoài JSON.
"""


def parse_llm_response(response_text: str):
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("action"), data.get("content")
    except:
        pass
    return "ask_user", "Xin lỗi, tôi chưa hiểu rõ. Bạn có thể nói rõ hơn?"


# ------------------ 5. Các node ------------------
def agent_node(state: AgentState):
    prompt = build_agent_prompt(state)
    response = llm.invoke(prompt)
    action, content = parse_llm_response(response.content)
    new_state = state.copy()
    new_state["next_action"] = action
    if action == "retrieve":
        new_state["current_query"] = content
    elif action == "ask_user":
        new_state["ask_user_question"] = content
    elif action == "final_answer":
        new_state["final_answer"] = content
    return new_state


def retrieve_node(state: AgentState):
    query = state["current_query"] or state["task"]
    chunks = retrieve_from_db(query, k=3)
    new_state = state.copy()
    new_state["retrieved_chunks"] = chunks
    new_state["previous_queries"].append(query)
    new_state["iteration"] = state["iteration"] + 1
    new_state["current_query"] = ""
    return new_state


def ask_user_node(state: AgentState):
    # Ở đây không dùng input() nữa, sẽ trả về state với flag chờ
    new_state = state.copy()
    new_state["next_action"] = "wait_for_user"  # flag đặc biệt
    return new_state


def final_answer_node(state: AgentState):
    return state


def router(state: AgentState):
    action = state.get("next_action", "final_answer")
    if action == "retrieve":
        return "retrieve"
    elif action == "ask_user":
        return "ask_user"
    elif action == "final_answer":
        return "final_answer"
    else:
        return "end"


# ------------------ 6. Xây dựng đồ thị ------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("ask_user", ask_user_node)
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
