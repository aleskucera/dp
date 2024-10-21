import warp as wp

@wp.kernel
def save_kernel(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), idx: wp.int32):
    b[idx] = a[idx]

@wp.kernel
def transform_points_kernel(points: wp.array(dtype=wp.vec3f),
                            transforms: wp.array(dtype=wp.transform)):
    tid = wp.tid()
    p = points[tid]
    t = transforms[tid]
    points[tid] = wp.transform_point(t, p)

@wp.kernel
def loss(points: wp.array(dtype=wp.vec3f), l: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    p = points[tid]
    norm = wp.dot(p, p)
    wp.atomic_add(l, 0, norm)

def main():
    l = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    points = wp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=wp.vec3f, requires_grad=True)
    transforms = wp.array([wp.transform_identity(), 
                           wp.transform_identity(),
                           wp.transform_identity()],
                           dtype=wp.transform, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel=transform_points_kernel, dim=len(points), inputs=[points, transforms])
        wp.launch(kernel=loss, dim=len(points), inputs=[points, l])
    tape.backward(grads={transforms: wp.ones((len(transforms), 7), dtype=wp.float32)})
    print(points.grad)
    print(transforms.grad)
    print(points.numpy())

if __name__ == "__main__":
    main()



