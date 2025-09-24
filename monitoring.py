from prometheus_client import Counter, Histogram

REQUESTS = Counter("api_requests_total", "Total API requests", ["route", "method", "status"])
PREDICTIONS = Counter("predictions_total", "Total predictions", ["country"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["route"])

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        route = scope.get("path", "unknown")
        method = scope.get("method", "GET")
        import time
        start = time.perf_counter()

        status_holder = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = str(message["status"])
                REQUESTS.labels(route=route, method=method, status=status).inc()
                status_holder["status"] = status
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            LATENCY.labels(route=route).observe(time.perf_counter() - start)
