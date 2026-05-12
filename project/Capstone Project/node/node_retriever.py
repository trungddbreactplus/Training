from typing import TypedDict, List, Optional, Literal
from app.config import client, co_here
from qdrant_client.models import MatchAny, Filter, FieldCondition
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
import os
from pprintpp import pprint
load_dotenv()



model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices='cpu')

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
    """Danh sách các đoạn văn bản (chunk) vừa được truy xuất từ vector DB ở bước retrieve gần nhất.
       Chỉ giữ tối đa 3-5 chunk để tránh quá tải token, giúp Agent dễ dàng đánh giá thông tin hiện có."""

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
    category_query: List[str]

def node_retriever(state: AgentState):
    embeddings = model.encode(
        [state['current_query']],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    query_vec = embeddings['dense_vecs'][0].tolist()

    filter_points = client.query_points(
        collection_name='RAG_ChatBot_HAUI_v1',
        query=query_vec,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key='source',
                    match=MatchAny(any=state['category_query'])
                )
            ]
        ),
        limit=30
    )

    docs = [p.payload.get('raw_text') for p in filter_points.points]

    if docs:
        results = co_here.rerank(
            model='rerank-v3.5',  # model rerank của Cohere
            query=state['current_query'],
            documents=docs, top_n=10
        )
        sorted_results = sorted(
            results.results,
            key=lambda x: x.relevance_score,  # sắp xếp theo score
            reverse=True  # giảm dần
        )
        state['retrieved_chunks'] = [docs[r.index] for r in sorted_results]

    else:
        state['retrieved_chunks'] = ['']
    return state

# input_query = AgentState(current_query="Điều kiện nhận học bổng", category_query=["HocBong.json"])
# res = node_retriever(input_query)['retrieved_chunks']
# pprint(res)