from prometheus_client import start_http_server, Counter, Gauge
import time

REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total prediction requests'
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy'
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU Usage'
)

start_http_server(8000)

while True:
    REQUEST_COUNT.inc()
    MODEL_ACCURACY.set(0.78)
    CPU_USAGE.set(50)

    time.sleep(5)
