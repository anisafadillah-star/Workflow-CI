from prometheus_client import start_http_server, Counter, Gauge
import psutil
import time

# Total request prediksi

REQUEST_COUNT = Counter(
"prediction_requests_total",
"Total prediction requests"
)

# Akurasi model

MODEL_ACCURACY = Gauge(
"model_accuracy",
"Current model accuracy"
)

# CPU Usage

CPU_USAGE = Gauge(
"cpu_usage_percent",
"CPU Usage"
)

# Memory Usage

MEMORY_USAGE = Gauge(
"memory_usage_percent",
"Memory Usage"
)

start_http_server(8000)

while True:
REQUEST_COUNT.inc()

```
# Isi dengan akurasi model yang kamu dapat saat training
MODEL_ACCURACY.set(0.7869)

CPU_USAGE.set(psutil.cpu_percent())

MEMORY_USAGE.set(psutil.virtual_memory().percent)

time.sleep(5)
```
