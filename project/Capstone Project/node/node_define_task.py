from pathlib import Path
from app.config import AgentState, llm
from typing import TypedDict, Literal
import json

BASE_DIR = Path(__file__).parent.parent  # parent của node/
json_path = BASE_DIR / "describe" / "describe_pdf.json"

with open(json_path, 'r', encoding='utf-8') as f:
    policies = json.load(f)

full_desc_lines = []
for p in policies:
    for k, v in p.items():
        json_file = f"{k}.json"
        full_desc_lines.append(f"{json_file}: {v}")
FULL_DESCRIPTION = "\n\n".join(full_desc_lines)


def prompt_define_task(state: AgentState) -> str:
    chunks_text = "\n\n".join(state.get('retrieved_chunks', 'Chưa có thông tin'))
    queries_text = ", ".join(state.get('previous_queries', 'Chưa có'))
    return f"""
    Bạn là trợ lý tư vấn về quy định, quy chế chính sách nhà trường.
    Hãy dựa vào câu hỏi và thông tin đã tra cứu để quyết định bước tiếp theo.
    
    Các domain tài liệu hiện có:
    {FULL_DESCRIPTION}
    
    Nếu câu hỏi liên quan ngữ nghĩa tới bất kỳ domain nào phía trên:
    - action phải là "retrieve"
    - content phải là danh sách tên file json liên quan
    - không được tự tạo tên file mới
    - không được dùng kiến thức nền để thay retrieval
    
    Chỉ chọn "final_answer" nếu:
    - câu hỏi không liên quan tới các domain trên
    - hoặc retrieved_chunks đã đủ để trả lời
    
    ### Câu hỏi của sinh viên:
    {state["task"]}
    
    ### Thông tin về sinh viên (từ bộ nhớ dài hạn):
    {json.dumps(state.get("user_context", {}), ensure_ascii=False)}
    
    ### Các bước đã thực hiện:
    - Số lần tra cứu: {state.get("iteration", 0)} / {state.get("max_iterations", 5)}
    - Các từ khóa đã tìm: {queries_text}
    
    ### Kết quả tra cứu gần nhất:
    {chunks_text}
    
    ### Yêu cầu:
    Trả lời bằng JSON duy nhất: {{"action": "retrieve/final_answer", "content": ["nội dung"]}}
    Không thêm text ngoài JSON.
    """


class DefineTaskOutput(TypedDict):
    action: Literal["retrieve", "final_answer"]
    content: list[str]


def node_define_task(state: AgentState):
    prompt = prompt_define_task(state)
    structured_llm = llm.with_structured_output(DefineTaskOutput, method='json_mode')
    response = structured_llm.invoke(prompt)
    state['next_action'] = response
    return state

# for _ in range(10):
#     input_state = AgentState(task="Điều kiện nhận học bổng là gì")
#     res = node_define_task(input_state)['next_action']
#     print(res)
