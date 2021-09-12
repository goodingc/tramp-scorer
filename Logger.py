from __future__ import annotations

import functools
import time
from typing import List

current_milli_time = lambda: int(round(time.time() * 1000))


class Logger:
    tags: List[str] = []

    def __init__(self, tags: List[str] = None):
        self.start_time = current_milli_time()
        if tags is None:
            self.tags = []
        else:
            self.tags = tags

    def print(self, status_character: str, message: str):
        lifetime = self.lifetime()
        if len(self.tags) > 0:
            tag_display = functools.reduce(lambda tag_display, tag: "{} [{}]".format(tag_display, tag), self.tags,
                                           "[{}]".format(status_character))
            print("[{}] {} {}".format(lifetime, tag_display, message))
        else:
            print("[{}] [{}] {}".format(lifetime, status_character, message))

    def success(self, message: str):
        self.print("+", message)

    def info(self, message: str):
        self.print("?", message)

    def warning(self, message: str):
        self.print("!", message)

    def error(self, message: str):
        self.print("-", message)

    def tag(self, tag: str) -> Logger:
        child = Logger(self.tags.copy())
        child.tags.append(tag)
        child.start_time = self.start_time
        return child

    def lifetime(self) -> int:
        return current_milli_time() - self.start_time

    def start_timer(self, name: str, silent=False) -> Timer:
        return Timer(self.tag("Timer: " + name), silent)


class Timer:
    def __init__(self, logger: Logger, silent=False):
        self.logger = logger
        self.silent = silent
        if not self.silent: self.logger.info("Started")
        self.start_time = current_milli_time()

    def end(self) -> int:
        total_time = current_milli_time() - self.start_time
        if not self.silent: self.logger.success("Finished in %dms" % total_time)
        return total_time
