from concurrent.futures import ThreadPoolExecutor
import os

# Queue used for SSE notifications
JOB_QUEUES = {}


EXECUTOR = ThreadPoolExecutor(
    max_workers=min(5, os.cpu_count() or 1)
)