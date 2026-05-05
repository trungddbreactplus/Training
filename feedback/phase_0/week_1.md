# Nhận xét Chung

- Cấu trúc còn chia thành nhiều script đơn lẻ, nhưng hướng tách nhỏ là hợp lý.
- `pyproject.toml` đang dùng sai tên dependency: `bs4` nên đổi thành `beautifulsoup4`.

# Bài tập

## Bài 1 (`Bai1.py`)

- Ý tưởng đúng.
- Có thể dùng `Counter` để gọn hơn.

## Bài 2 (`Bai2.py`)

- Đúng với trường hợp chia hết.
- Chưa xử lý phần dư vì vòng `while` đang dừng sớm, nên có thể mất dữ liệu.

## Bài 3 (`Bai3.py`)

- Logic BFS/DFS cơ bản đúng.
- Cần chuẩn hóa URL từ relative sang absolute.

## Bài 4 (`Bai4.py`, `test.py`)

- Ý tưởng retry/backoff ổn.

# Tổng kết

- Nắm được khái niệm cơ bản.
- Cần chỉnh nhẹ ở URL normalization, xử lý partial chunk
