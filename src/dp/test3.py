import warp as wp
import numpy as np

# Example: compute sin values at intervals of delta_t

# Parameters
n_points = 100  # Number of points
delta_t = 0.1   # Time step

# Define output array
sin_values = wp.zeros(n_points, dtype=wp.float32, device="cuda")

# Kernel to compute sine values
@wp.kernel
def compute_sin(delta_t: float, out: wp.array(dtype=wp.float32)):
    tid = wp.tid()  # Get thread index
    t = wp.float(tid) * delta_t
    out[tid] = wp.sin(t)  # Compute sine for each point

# Launch the kernel
wp.launch(
    kernel=compute_sin,
    dim=n_points,  # Number of threads
    inputs=[delta_t, sin_values]
)

# Copy results to host and print
sin_values_host = sin_values.numpy()  # Transfer from GPU to host
print(sin_values_host)
