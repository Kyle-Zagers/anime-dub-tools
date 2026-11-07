"""The line scheduler overlaps CPU prep with inference. No episode media."""

from __future__ import annotations

import threading
import time

import pytest

from anime_dub_tools.schedule import overlap_line_work


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def test_cpu_prep_and_finish_overlap_inference():
    count = 4
    ahead = 2
    prep_entered = [threading.Event() for _ in range(count)]
    finish_entered = [threading.Event() for _ in range(count)]
    spans: dict[str, dict[int, tuple[float, float]]] = {"prep": {}, "infer": {}, "finish": {}}
    infer_ids: list[int] = []
    finish_ids: list[int] = []
    active_finishes = 0
    peak_finishes = 0
    lock = threading.Lock()

    def prepare(index: int) -> int:
        started = time.perf_counter()
        prep_entered[index].set()
        time.sleep(0.2)
        spans["prep"][index] = (started, time.perf_counter())
        return index

    def infer(index: int) -> int:
        infer_ids.append(threading.get_ident())
        upcoming = index + ahead
        if upcoming < count:
            assert prep_entered[upcoming].wait(1.0)
        if index > 0:
            assert finish_entered[index - 1].wait(1.0)
        started = time.perf_counter()
        time.sleep(0.1)
        spans["infer"][index] = (started, time.perf_counter())
        return index

    def finish(index: int, prepared: int, inferred: int) -> int:
        assert prepared == inferred == index
        finish_ids.append(threading.get_ident())
        started = time.perf_counter()
        finish_entered[index].set()
        nonlocal active_finishes, peak_finishes
        with lock:
            active_finishes += 1
            peak_finishes = max(peak_finishes, active_finishes)
        try:
            time.sleep(0.25)
        finally:
            with lock:
                active_finishes -= 1
        spans["finish"][index] = (started, time.perf_counter())
        return index

    rendered = overlap_line_work(
        list(range(count)), prepare, infer, finish, ahead=ahead, workers=4
    )
    assert rendered == [0, 1, 2, 3]
    assert _overlaps(spans["prep"][2], spans["infer"][0])
    assert _overlaps(spans["finish"][0], spans["infer"][1])
    assert peak_finishes >= 2
    assert len(set(infer_ids)) == 1
    assert set(infer_ids).isdisjoint(finish_ids)
    for index in range(count - 1):
        assert spans["infer"][index][1] <= spans["infer"][index + 1][0]


def test_inference_failure_is_not_swallowed():
    def prepare(index: int) -> int:
        return index

    def infer(index: int) -> int:
        if index == 1:
            raise RuntimeError("gpu failed")
        return index

    def finish(index: int, prepared: int, inferred: int) -> int:
        del prepared, inferred
        return index

    with pytest.raises(RuntimeError, match="gpu failed"):
        overlap_line_work([0, 1, 2], prepare, infer, finish, ahead=1, workers=2)
