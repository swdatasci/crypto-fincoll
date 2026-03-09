"""
Prometheus metrics for FinColl service.

Exposes metrics for:
- API request rates and latencies
- Provider-specific metrics (requests, errors, rate limits)
- Feature extraction performance
- Model inference latency
"""

import time
from functools import wraps
from typing import Callable, Dict, Optional

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create no-op classes if prometheus_client not installed
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Info:
        def __init__(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

    def generate_latest():
        return b"# Prometheus client not installed\n"

    CONTENT_TYPE_LATEST = "text/plain"


# Service info
service_info = Info("fincoll_service", "FinColl service information")

# API request metrics
api_requests_total = Counter(
    "fincoll_api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)

api_request_duration = Histogram(
    "fincoll_api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint", "method"],
)

# Provider metrics
provider_requests_total = Counter(
    "fincoll_provider_requests_total",
    "Total requests to data providers",
    ["provider", "endpoint", "status"],
)

provider_request_duration = Histogram(
    "fincoll_provider_request_duration_seconds",
    "Provider request duration in seconds",
    ["provider", "endpoint"],
)

provider_errors_total = Counter(
    "fincoll_provider_errors_total", "Total provider errors", ["provider", "error_type"]
)

# Rate limit metrics
rate_limit_capacity_remaining = Gauge(
    "fincoll_rate_limit_capacity_remaining",
    "Remaining rate limit capacity (requests)",
    ["provider"],
)

rate_limit_capacity_percent = Gauge(
    "fincoll_rate_limit_capacity_percent",
    "Rate limit capacity used (percentage)",
    ["provider"],
)

rate_limit_exceeded_total = Counter(
    "fincoll_rate_limit_exceeded_total", "Total rate limit violations", ["provider"]
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "fincoll_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open)",
    ["provider"],
)

circuit_breaker_failures = Gauge(
    "fincoll_circuit_breaker_failures",
    "Current failure count for circuit breaker",
    ["provider"],
)

# Safe mode metrics
safe_mode_state = Gauge(
    "fincoll_safe_mode_state",
    "Safe mode state (0=normal, 1=warning, 2=safe_mode, 3=manual_pause)",
)

safe_mode_active = Gauge(
    "fincoll_safe_mode_active",
    "Whether safe mode is currently active (1=active, 0=inactive)",
)

safe_mode_triggers_total = Counter(
    "fincoll_safe_mode_triggers_total",
    "Total number of safe mode activations",
    ["trigger_type"],  # "rate_limit", "server_error", "manual"
)

safe_mode_events_total = Counter(
    "fincoll_safe_mode_events_total",
    "Total safe mode events recorded",
    ["provider", "event_type"],  # event_type: rate_limit, server_error, timeout
)

safe_mode_active_seconds = Counter(
    "fincoll_safe_mode_active_seconds_total",
    "Total time spent in safe mode (seconds)",
)

# Feature extraction metrics
feature_extraction_duration = Histogram(
    "fincoll_feature_extraction_duration_seconds",
    "Feature extraction duration in seconds",
    ["symbol", "feature_type"],
)

feature_extraction_errors = Counter(
    "fincoll_feature_extraction_errors_total",
    "Total feature extraction errors",
    ["symbol", "feature_type", "error_type"],
)

# Model inference metrics
model_inference_duration = Histogram(
    "fincoll_model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_type"],
)

model_inference_total = Counter(
    "fincoll_model_inference_total",
    "Total model inference requests",
    ["model_type", "status"],
)

# Cache metrics
cache_hits_total = Counter(
    "fincoll_cache_hits_total", "Total cache hits", ["cache_type", "symbol"]
)

cache_misses_total = Counter(
    "fincoll_cache_misses_total", "Total cache misses", ["cache_type", "symbol"]
)

# Database metrics
database_queries_total = Counter(
    "fincoll_database_queries_total", "Total database queries", ["operation", "table"]
)

database_query_duration = Histogram(
    "fincoll_database_query_duration_seconds",
    "Database query duration in seconds",
    ["operation", "table"],
)


def track_api_request(endpoint: str, method: str = "GET"):
    """
    Decorator to track API request metrics.

    Usage:
        @track_api_request("/api/v1/inference/predict")
        async def predict(symbol: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                api_requests_total.labels(
                    endpoint=endpoint, method=method, status=status
                ).inc()
                api_request_duration.labels(endpoint=endpoint, method=method).observe(
                    duration
                )

        return wrapper

    return decorator


def track_provider_request(provider: str, endpoint: str):
    """
    Context manager to track provider request metrics.

    Usage:
        with track_provider_request("tradestation", "/v3/marketdata/barcharts"):
            response = client.get(url)
    """

    class ProviderRequestTracker:
        def __init__(self, provider: str, endpoint: str):
            self.provider = provider
            self.endpoint = endpoint
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            status = "error" if exc_type else "success"

            provider_requests_total.labels(
                provider=self.provider, endpoint=self.endpoint, status=status
            ).inc()

            provider_request_duration.labels(
                provider=self.provider, endpoint=self.endpoint
            ).observe(duration)

            if exc_type:
                error_type = exc_type.__name__
                provider_errors_total.labels(
                    provider=self.provider, error_type=error_type
                ).inc()

    return ProviderRequestTracker(provider, endpoint)


def update_rate_limit_metrics(provider: str, stats: Dict):
    """
    Update rate limit metrics from provider stats.

    Args:
        provider: Provider name (e.g., "tradestation")
        stats: Rate limit stats dict with keys:
            - capacity_remaining: int
            - capacity_percent: float
    """
    if "capacity_remaining" in stats:
        rate_limit_capacity_remaining.labels(provider=provider).set(
            stats["capacity_remaining"]
        )

    if "capacity_percent" in stats:
        rate_limit_capacity_percent.labels(provider=provider).set(
            stats["capacity_percent"]
        )


def update_circuit_breaker_metrics(provider: str, is_open: bool, failures: int):
    """
    Update circuit breaker metrics.

    Args:
        provider: Provider name
        is_open: Whether circuit breaker is open
        failures: Current failure count
    """
    circuit_breaker_state.labels(provider=provider).set(1 if is_open else 0)
    circuit_breaker_failures.labels(provider=provider).set(failures)


def update_safe_mode_metrics(state: str, is_active: bool):
    """
    Update safe mode state metrics.

    Args:
        state: Safe mode state ("normal", "warning", "safe_mode", "manual_pause")
        is_active: Whether safe mode is currently active
    """
    # Map state to numeric value
    state_values = {
        "normal": 0,
        "warning": 1,
        "safe_mode": 2,
        "manual_pause": 3,
    }

    safe_mode_state.set(state_values.get(state, 0))
    safe_mode_active.set(1 if is_active else 0)


def record_safe_mode_trigger(trigger_type: str):
    """
    Record a safe mode trigger event.

    Args:
        trigger_type: Type of trigger ("rate_limit", "server_error", "manual")
    """
    safe_mode_triggers_total.labels(trigger_type=trigger_type).inc()


def record_safe_mode_event(provider: str, event_type: str):
    """
    Record a safe mode event (error).

    Args:
        provider: Provider name
        event_type: Event type ("rate_limit", "server_error", "timeout")
    """
    safe_mode_events_total.labels(provider=provider, event_type=event_type).inc()


def increment_safe_mode_active_time(seconds: float):
    """
    Increment total time spent in safe mode.

    Args:
        seconds: Number of seconds to add
    """
    safe_mode_active_seconds.inc(seconds)


def get_metrics() -> bytes:
    """Get Prometheus metrics in text format."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest()
    else:
        return b"# Prometheus client not installed\n"


def get_content_type() -> str:
    """Get content type for metrics response."""
    return CONTENT_TYPE_LATEST
