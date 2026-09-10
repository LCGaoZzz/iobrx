"""Default thread budgets respect process affinity, without capping explicit choices."""
import pytest
from iobrx import _threads as threads


@pytest.fixture(autouse=True)
def reset_policy(monkeypatch):
    monkeypatch.setattr(threads, "_user_default", None)
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(threads.os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(threads.os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False)


def test_affinity_caps_only_the_default(monkeypatch):
    monkeypatch.setattr(threads.os, "sched_getaffinity", lambda _pid: {3, 5})
    assert threads.available_cpu_count() == threads.get_threads() == 2
    assert threads.resolve_threads(16) == 16
    threads.set_threads(4)
    assert threads.get_threads() == 4
    threads.set_threads(None)
    assert threads.get_threads() == 2


def test_new_python_process_count_and_desktop_cap(monkeypatch):
    monkeypatch.setattr(threads.os, "process_cpu_count", lambda: 3)
    assert threads.get_threads() == 3
    monkeypatch.setattr(threads.os, "process_cpu_count", lambda: 32)
    assert threads.get_threads() == 8


@pytest.mark.parametrize("error", [AttributeError, NotImplementedError, OSError])
def test_portable_cpu_count_fallback(monkeypatch, error):
    def unavailable(*_):
        raise error("not supported")
    monkeypatch.setattr(threads.os, "process_cpu_count", unavailable)
    monkeypatch.setattr(threads.os, "sched_getaffinity", unavailable)
    assert threads.get_threads() == 8
    monkeypatch.setattr(threads.os, "cpu_count", lambda: None)
    assert threads.get_threads() == 1


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_explicit_thread_counts(n):
    with pytest.raises(ValueError):
        threads.resolve_threads(n)
    with pytest.raises(ValueError):
        threads.set_threads(n)
