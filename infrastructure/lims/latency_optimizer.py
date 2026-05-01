"""LIMS latency optimization helpers.

Provides simple instrumentation, adaptive warmup, and lightweight caching
to reduce perceived latency for remote LIMS calls.
"""
from __future__ import annotations

import functools
import time
import threading
from typing import Any, Callable, Dict, Optional


class LimsLatencyOptimizer:
    """Simple latency optimizer for LIMS integrations.

    Features:
    - Record latency EMA (exponential moving average)
    - Adaptive pre-warm callback when latency exceeds thresholds
    - LRU-like in-memory TTL cache for idempotent lookups
    - Decorator to wrap LIMS callables (sync or async supported via wrapper)
    """

    def __init__(self, cache_ttl: float = 30.0, warmup_threshold: float = 0.5):
        self.lock = threading.Lock()
        self.ema_latency = None  # seconds
        self.alpha = 0.2
        self.cache: Dict[str, tuple[float, Any]] = {}
        self.cache_ttl = cache_ttl
        self.warmup_threshold = warmup_threshold

    def _update_ema(self, latency: float) -> None:
        with self.lock:
            if self.ema_latency is None:
                self.ema_latency = latency
            else:
                self.ema_latency = self.alpha * latency + (1 - self.alpha) * self.ema_latency

    def get_ema(self) -> Optional[float]:
        return self.ema_latency

    def cache_get(self, key: str) -> Optional[Any]:
        v = self.cache.get(key)
        if not v:
            return None
        ts, val = v
        if time.time() - ts > self.cache_ttl:
            try:
                del self.cache[key]
            except KeyError:
                pass
            return None
        return val

    def cache_set(self, key: str, value: Any) -> None:
        self.cache[key] = (time.time(), value)

    def pre_warm(self, fn: Callable[..., Any], *args, **kwargs) -> None:
        """Run a background pre-warm call (fire-and-forget)."""
        def runner():
            try:
                fn(*args, **kwargs)
            except Exception:
                pass
        t = threading.Thread(target=runner, daemon=True)
        t.start()

    def instrument(self, key_func: Optional[Callable[..., str]] = None):
        """Decorator to instrument LIMS callables.

        key_func: optional callable to produce cache key from args/kwargs
        """
        def decorator(fn: Callable[..., Any]):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                key = None
                if key_func is not None:
                    try:
                        key = key_func(*args, **kwargs)
                    except Exception:
                        key = None
                if key:
                    cached = self.cache_get(key)
                    if cached is not None:
                        return cached
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                latency = time.perf_counter() - start
                self._update_ema(latency)
                # if latency is high, schedule a background warmup
                if self.ema_latency and self.ema_latency > self.warmup_threshold:
                    try:
                        self.pre_warm(fn, *args, **kwargs)
                    except Exception:
                        pass
                if key:
                    try:
                        self.cache_set(key, result)
                    except Exception:
                        pass
                return result
            return wrapper
        return decorator


# convenience factory
_default_optimizer: Optional[LimsLatencyOptimizer] = None

def get_default_optimizer() -> LimsLatencyOptimizer:
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = LimsLatencyOptimizer()
    return _default_optimizer
