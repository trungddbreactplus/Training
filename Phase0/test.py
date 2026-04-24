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
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:

                    if attempt == max_retries:
                        raise e

                    delay = base_delay * (backoff ** attempt)
                    print(f"Lần thử {attempt + 1} thất bại, retry sau {delay}s")
                    time.sleep(delay)

        return wrapper
    return decorator

@retry(max_retries=3)
def call_api(url: str):
    r = random.randrange(-1, 0)

    if r < 0.4:
        raise TimeoutError("Request timed out")
    elif r < 0.6:
        raise ConnectionError("Connection refused")
    else:
        return {"status": 200, "data": "OK"}

result = call_api("")
if isinstance(result, dict):
    print(result)