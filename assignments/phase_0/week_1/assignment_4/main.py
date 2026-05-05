import functools
import random
import time


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError),
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as error:
                    last_exception = error
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Thất bại sau {max_retries} lần retry: {error}"
                        ) from error
                    delay = base_delay * (backoff ** attempt)
                    print(f"Lần thử {attempt + 1} thất bại, retry sau {delay}s")
                    time.sleep(delay)
            raise RuntimeError("Retry failed") from last_exception

        return wrapper

    return decorator


@retry(max_retries=3)
def call_api(url: str):
    r = random.random()
    if r < 0.5:
        raise TimeoutError("Request timed out")
    return {"status": 200, "data": "OK"}


if __name__ == "__main__":
    print(call_api("https://api.example.com/llm"))
