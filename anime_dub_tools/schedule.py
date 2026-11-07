"""Keep one inference thread busy while CPU work runs beside it.

Inference stays on the caller. That is the thread that owns the GPU model.
Preparing a later line, and finishing an earlier line with ffmpeg, run on a
pool so those stages do not sit in front of the next inference.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

Item = TypeVar("Item")
Prepared = TypeVar("Prepared")
Inferred = TypeVar("Inferred")
Finished = TypeVar("Finished")


def cpu_workers() -> int:
    count = os.cpu_count() or 2
    return max(2, min(8, count))


def overlap_line_work(
    items: list[Item],
    prepare: Callable[[Item], Prepared],
    infer: Callable[[Prepared], Inferred],
    finish: Callable[[Item, Prepared, Inferred], Finished],
    *,
    ahead: int = 2,
    workers: int | None = None,
    pool: ThreadPoolExecutor | None = None,
) -> list[Finished]:
    """Infer each item on the caller while the pool prepares and finishes lines.

    ``prepare`` of an upcoming item is submitted before ``infer`` of the
    current one. ``finish`` of the current item is submitted before the next
    ``infer``. ``infer`` calls themselves never overlap.
    """
    if not items:
        return []
    if ahead < 1:
        raise ValueError("ahead must be at least 1")
    worker_count = workers or cpu_workers()
    own_pool = pool is None
    if pool is None:
        pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dub-cpu")
    # Finishes (ffmpeg) use their own pool so a backed-up time-fit cannot
    # occupy every worker and leave the next line unprepared.
    finish_pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dub-finish")
    finishes: list[Future[Finished]] = []
    try:
        prefetched: dict[int, Future[Prepared]] = {}
        for index in range(min(ahead, len(items))):
            prefetched[index] = pool.submit(prepare, items[index])
        for index, item in enumerate(items):
            prepared = prefetched.pop(index).result()
            upcoming = index + ahead
            if upcoming < len(items):
                prefetched[upcoming] = pool.submit(prepare, items[upcoming])
            inferred = infer(prepared)
            finishes.append(finish_pool.submit(finish, item, prepared, inferred))
        return [future.result() for future in finishes]
    finally:
        finish_pool.shutdown(wait=True, cancel_futures=False)
        if own_pool:
            pool.shutdown(wait=True, cancel_futures=False)
