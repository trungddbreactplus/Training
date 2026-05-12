import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from zep_cloud.client import Zep
from zep_cloud.types import Message
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

ZEP_API_KEY = os.getenv("ZEP_API")
GROQ_API_KEY = os.getenv("GROQ_API")


@st.cache_resource
def get_zep_client():
    return Zep(api_key=ZEP_API_KEY)


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)


zep_client = get_zep_client()
llm = get_llm()

# Khởi tạo session
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Tạo user và thread (nếu chưa tồn tại) ---
# ... (giữ nguyên code tạo user và thread hiện tại của em)
try:
    zep_client.user.get(user_id=st.session_state.user_id)
except:
    zep_client.user.add(
        user_id=st.session_state.user_id,
        first_name="User",
        email=f"{st.session_state.user_id}@example.com"
    )

try:
    zep_client.thread.get(thread_id=st.session_state.thread_id)
except:
    zep_client.thread.create(
        thread_id=st.session_state.thread_id,
        user_id=st.session_state.user_id
    )


def get_memory_context() -> str:
    """
    Cách này dùng memory.get(), Zep tự xử lý:
    - Dùng 4 tin nhắn gần nhất trong thread để tìm fact.
    - Lấy danh sách fact, đánh giá rating và trả về context string.
    - Tối ưu cho đa số các tình huống chat thông thường.
    """
    try:
        memory_result = zep_client.memory.get(
            session_id=st.session_state.thread_id,
            last_n=10,
            min_rating=0.7  # Chỉ lấy fact có độ liên quan cao
        )
        # Trả về context string đã được format sẵn
        return memory_result.context if memory_result.context else ""
    except Exception as e:
        return ""


def get_graph_context(query: str) -> str:
    """
    Cách này dùng graph.search(), chủ động tìm fact theo query.
    - Phù hợp khi cần tìm thông tin rất cụ thể, hoặc khi không muốn phụ thuộc vào lịch sử gần đây.
    """
    try:
        search_results = zep_client.graph.search(
            user_id=st.session_state.user_id,
            query=query,
            scope="edges",
            limit=5
        )
        facts = []
        for result in search_results:
            if hasattr(result, 'data') and hasattr(result.data, 'fact'):
                facts.append(result.data.fact)
        return "\n".join(facts) if facts else ""
    except Exception as e:
        return ""


# Giao diện Streamlit
st.set_page_config(page_title="Zep v3 Memory Chatbot")
st.title("🧠 Chatbot Đồ Thị Tri Thức Với Zep (Hybrid Memory)")
st.caption(f"User: `{st.session_state.user_id}` | Thread: `{st.session_state.thread_id}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Lưu tin nhắn user vào Zep
    user_msg = Message(role="user", content=prompt)
    zep_client.thread.add_messages(thread_id=st.session_state.thread_id, messages=[user_msg])

    # --- 🚀 CHIẾN LƯỢC LAI GHÉP (HYBRID) ---
    # 1. Lấy context nhanh từ memory.get (dựa trên lịch sử gần đây)
    base_context = get_memory_context()

    # 2. Nếu không có context từ memory.get, hoặc câu hỏi rất cụ thể, thì tìm kiếm sâu trong graph
    # Ở đây, nếu base_context rỗng, ta sẽ fallback sang graph.search
    if not base_context:
        detailed_context = get_graph_context(prompt)
    else:
        detailed_context = ""

    # Kết hợp cả hai nguồn context
    final_context = base_context + "\n" + detailed_context if detailed_context else base_context
    # ---

    # Lấy lịch sử chat gần đây
    thread = zep_client.thread.get(thread_id=st.session_state.thread_id)
    recent_history = []
    for msg in thread.messages[-10:]:
        if msg.role == "user":
            recent_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            recent_history.append(AIMessage(content=msg.content))

    system_prompt = f"""Bạn là một trợ lý AI thông minh, có khả năng nhớ lâu dài.

    Dưới đây là các thông tin Zep tổng hợp được về người dùng từ các cuộc trò chuyện trước:

    {final_context if final_context else "Không có thông tin liên quan nào từ quá khứ."}

    Hãy dùng những thông tin trên để trả lời một cách cá nhân hóa, tự nhiên.
    """

    messages = [SystemMessage(content=system_prompt)] + recent_history + [HumanMessage(content=prompt)]

    with st.spinner("Đang suy nghĩ..."):
        response = llm.invoke(messages)
        bot_reply = response.content

    # Lưu phản hồi bot
    assistant_msg = Message(role="assistant", content=bot_reply)
    zep_client.thread.add_messages(thread_id=st.session_state.thread_id, messages=[assistant_msg])

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    st.rerun()