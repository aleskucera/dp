#!/usr/bin/env python

import warp as wp

@wp.kernel
def inc_loop(a: wp.array(dtype=float), num_iters: int):
    i = wp.tid()
    for j in range(num_iters):
        a[i] += 1.0

n = 10_000_000
devices = wp.get_cuda_devices()

# pre-allocate host arrays for readback
host_arrays = [
    wp.empty(n, dtype=float, device="cpu", pinned=True) for _ in devices
]

# code for profiling
with wp.ScopedTimer("Demo", use_nvtx=True, color="yellow", synchronize=True):
    for i, device in enumerate(devices):
        a = wp.zeros(n, dtype=float, device=device)
        wp.launch(inc_loop, dim=n, inputs=[a, 500], device=device)
        wp.launch(inc_loop, dim=n, inputs=[a, 1000], device=device)
        wp.launch(inc_loop, dim=n, inputs=[a, 1500], device=device)
        wp.copy(host_arrays[i], a)