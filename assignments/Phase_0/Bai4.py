import time
import random
import functools


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (TimeoutError, ConnectionError),
):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):  # +1 vì lần đầu không phải retry
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        print(f"[OK] Thành công sau {attempt} lần retry.")
                    return result

                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break  # hết lượt, thoát vòng lặp

                    delay = base_delay * (backoff ** attempt)
                    print(
                        f"[Retry {attempt + 1}/{max_retries}] "
                        f"Lỗi: {type(e).__name__}: {e} — "
                        f"Thử lại sau {delay:.1f}s..."
                    )
                    time.sleep(delay)

            raise RuntimeError(
                f"Thất bại sau {max_retries} lần retry. "
                f"Lỗi cuối: {type(last_exception).__name__}: {last_exception}"
            ) from last_exception

        return wrapper
    return decorator


# ───────────────────────────────────────────────
# Dùng decorator
# ───────────────────────────────────────────────

@retry(max_retries=4, base_delay=1.0, backoff=2.0, exceptions=(TimeoutError, ConnectionError))
def call_api(url: str):
    r = random.randrange(-2, 0)
    if r < 0.4:
        raise TimeoutError("Request timed out")
    elif r < 0.6:
        raise ConnectionError("Connection refused")
    else:
        return {"status": 200, "data": "OK"}




result = call_api("https://api.example.com/llm")
print(f"Kết quả: {result}")
