import warp as wp

@wp.kernel
def compute1(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    b[tid] = a[tid] + b[tid]

@wp.kernel
def compute2(c: wp.array(dtype=wp.float32), d: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    d[tid] = c[tid] + d[tid]

@wp.kernel
def loss(d: wp.array(dtype=wp.float32), l: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    wp.atomic_add(l, 0, d[tid])

@wp.kernel
def save_kernel(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), idx: wp.int32):
    b[idx] = a[idx]

# Initialize data
a = wp.ones(100, dtype=wp.float32, requires_grad=True)
b = wp.ones(100, dtype=wp.float32, requires_grad=True)
c = wp.ones(100, dtype=wp.float32, requires_grad=True)
d = wp.ones(100, dtype=wp.float32, requires_grad=True)
l = wp.zeros(1, dtype=wp.float32, requires_grad=True)

tape = wp.Tape()

# forward pass
with tape:
    with wp.ScopedTimer("Demo", use_nvtx=True, color="yellow"):
        wp.launch(kernel=compute1, dim=len(a), inputs=[a, b])
        wp.launch(kernel=compute2, dim=len(c), inputs=[b, c])
        wp.launch(kernel=save_kernel, dim=1, inputs=[c, d])
        wp.launch(kernel=loss, dim=len(d), inputs=[d, l])

# reverse pass
tape.backward(l)

print(a)
print(b)

