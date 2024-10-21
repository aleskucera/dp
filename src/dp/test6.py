import time

import warp as wp
import numpy as np

ARRAY_SIZE = 1000
NUM_ITERATIONS = 100000

@wp.kernel
def add_vector(a: wp.array(dtype=wp.float32), 
               b: wp.array(dtype=wp.float32)):
    tid = wp.tid()  # Thread ID
    b[tid] += a[tid]  # Simple vector addition

# Initialize input arrays
random_array = np.random.rand(ARRAY_SIZE).astype(np.float32)
a = wp.array(random_array, device='cuda')
b_graph = wp.zeros(ARRAY_SIZE, dtype=wp.float32, device='cuda')
b_no_graph = wp.zeros(ARRAY_SIZE, dtype=wp.float32, device='cuda')

# =====================
# Without CUDA Graphs
# =====================
start_no_graph = time.time()

with wp.ScopedTimer("No CUDA Graphs", use_nvtx=True, color="red", synchronize=True, cuda_filter=wp.TIMING_ALL):
    for _ in range(NUM_ITERATIONS):
        wp.launch(add_vector, dim=len(a), inputs=[a, b_no_graph])

end_no_graph = time.time()

# =====================
# With CUDA Graphs
# =====================
# Capture CUDA graph

with wp.ScopedTimer("CUDA Graph Capture", use_nvtx=True, color="green", synchronize=True, cuda_filter=wp.TIMING_ALL):
    with wp.ScopedCapture() as capture:
        wp.launch(add_vector, dim=len(a), inputs=[a, b_graph])

add_graph = capture.graph

start_graph = time.time()

with wp.ScopedTimer("With CUDA Graphs", use_nvtx=True, color="blue", synchronize=True, cuda_filter=wp.TIMING_ALL):
    for _ in range(NUM_ITERATIONS):
        wp.capture_launch(add_graph)

end_graph = time.time()

# =====================
# Results
# =====================

print(f"Time without CUDA Graphs: {end_no_graph - start_no_graph:.6f} seconds")
print(f"Time with CUDA Graphs: {end_graph - start_graph:.6f} seconds")
