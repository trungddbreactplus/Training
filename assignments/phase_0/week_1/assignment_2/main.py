from typing import TypeVar

T = TypeVar("T")


def chunk_tokens(tokens: list[T], chunk_size: int, overlap: int = 0) -> list[list[T]]:
    """Chia danh sách thành các chunk có kích thước cố định.
    
    Args:
        tokens: danh sách token cần chia
        chunk_size: kích thước mỗi chunk
        overlap: số phần tử trùng lặp giữa các chunk liên tiếp
    
    Returns:
        Danh sách các chunk (bao gồm cả chunk cuối dù chưa đầy đủ)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks: list[list[T]] = []
    start = 0
    step = chunk_size - overlap

    # Lấy các chunk đầy đủ kích thước
    while start + chunk_size <= len(tokens):
        chunks.append(tokens[start : start + chunk_size])
        start += step

    # [FIX] Luôn giữ chunk cuối dù chưa đầy đủ kích thước để không mất dữ liệu
    if start < len(tokens):
        chunks.append(tokens[start:])

    return chunks


if __name__ == "__main__":
    tokens = ["a", "b", "c", "d", "e", "f"]
    print("Test 1 - chia hết:")
    print(chunk_tokens(tokens, chunk_size=3, overlap=2))
    
    # [FIX] Demo xử lý phần dư - luôn giữ chunk cuối
    tokens_odd = ["a", "b", "c", "d", "e", "f", "g"]
    print("\nTest 2 - chia không hết (tự động giữ chunk cuối):")
    print(chunk_tokens(tokens_odd, chunk_size=3, overlap=0))  # Giữ "g"
    
    print("\nTest 3 - overlap=1:")
    print(chunk_tokens(tokens_odd, chunk_size=3, overlap=1))
