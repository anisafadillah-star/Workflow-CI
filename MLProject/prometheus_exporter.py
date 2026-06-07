from prometheus_client import start_http_server, Counter
import time

REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total prediction requests'
)

if __name__ == "__main__":
    start_http_server(8000)

    while True:
        REQUEST_COUNT.inc()
        time.sleep(5)
