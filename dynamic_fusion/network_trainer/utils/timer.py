from __future__ import annotations

from time import time
from typing import Any, Optional


class Timer:
    def __init__(self, autoprint_str: Optional[str] = None) -> None:
        self.autoprint_str = autoprint_str
        self.interval = 0.0

    def __enter__(self) -> Timer:
        # self.start = time.process_time()
        self.start = time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time()
        self.interval = self.end - self.start
        if self.autoprint_str is not None:
            print(f'Time of "{self.autoprint_str}" is {self.interval:.2f}')
