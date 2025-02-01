from scipy.spatial.transform import Rotation as R
import numpy as np

# Create two rotations
p = R.from_quat([0, 0, 1, 1])
q = R.from_quat([1, 0, 0, 1])

# Normalize the quaternions (important for comparison)
p_quat = p.as_quat()
q_quat = q.as_quat()
p_quat = p_quat / np.linalg.norm(p_quat)
q_quat = q_quat / np.linalg.norm(q_quat)

# Create test point
test_point = np.array([1, 0, 0])

# Test both orders of multiplication
pq = p * q
qp = q * p

# Apply transformations in two different ways
# Method 1: Using the combined rotation
result_pq = pq.apply(test_point)
result_qp = qp.apply(test_point)

# Method 2: Applying rotations one after another
result_p_then_q = p.apply(q.apply(test_point))
result_q_then_p = q.apply(p.apply(test_point))

print("Test point:", test_point)
print("\nResult of p * q:", result_pq)
print("Result of applying q then p:", result_p_then_q)
print("\nResult of q * p:", result_qp)
print("Result of applying p then q:", result_q_then_p)

# Check which matches
if np.allclose(result_pq, result_p_then_q):
    print("\nConclusion: p * q means apply q first, then p")
elif np.allclose(result_pq, result_q_then_p):
    print("\nConclusion: p * q means apply p first, then q")
else:
    print("\nResults are inconsistent")
