import os
import multiprocessing

# benchmark_parallel.py
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def dummy_template_work(i):
    """Simulates one matched filter iteration with numpy-heavy work"""
    import numpy as np
    x = np.random.randn(8192 * 10).astype(np.float32)
    return i, np.abs(np.fft.rfft(x))

N = 100  # simulate 100 templates
n_cores = multiprocessing.cpu_count()
print(f"Cores available: {n_cores}")

# Sequential baseline
t0 = time.time()
for i in range(N):
    dummy_template_work(i)
t_seq = time.time() - t0
print(f"Sequential:          {t_seq:.2f}s  →  {N/t_seq:.1f} it/s")

# ThreadPool
for w in [2, 4, n_cores]:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=w) as ex:
        list(ex.map(dummy_template_work, range(N)))
    t = time.time() - t0
    print(f"ThreadPool({w:2d} workers): {t:.2f}s  →  {N/t:.1f} it/s")

# ProcessPool
for w in [2, 4, n_cores]:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=w) as ex:
        list(ex.map(dummy_template_work, range(N)))
    t = time.time() - t0
    print(f"ProcessPool({w:2d} workers): {t:.2f}s  →  {N/t:.1f} it/s")
