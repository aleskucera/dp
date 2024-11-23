import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytest
from xpbd_pytorch.quat import (
    quaternion_multiply,
    quaternion_conjugate,
    rotate_vector,
    rotate_vector_inverse,
)


@pytest.fixture
def identity_quaternion():
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


@pytest.fixture
def test_vector():
    return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

class TestQuaternionMultiplication:
    @pytest.mark.parametrize(
        "q1, q2, expected",
        [
            (torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, 0.0])),
            (torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, 0.0])),
        ],
    )
    def test_identity_multiplication(self, q1, q2, expected):
        result = quaternion_multiply(q1, q2)
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


    def test_consecutive_rotations(self):
        q_90z = torch.tensor([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=torch.float32)
        q_180z = quaternion_multiply(q_90z, q_90z)
        expected_180z = torch.tensor([np.cos(np.pi / 2), 0.0, 0.0, np.sin(np.pi / 2)], dtype=torch.float32)
        assert torch.allclose(q_180z, expected_180z, atol=1e-6), "90-degree rotation composition failed"

    def test_compare_with_scipy(self):
        q1 = torch.tensor([0.7071, 0.1421, 0.7421, 0.0], dtype=torch.float32)
        q2 = torch.tensor([0.7071, 0.0, 1.0, 0.12421], dtype=torch.float32)
        q1 = q1 / torch.norm(q1)
        q2 = q2 / torch.norm(q2)
        q1_scipy = R.from_quat([q1[1].item(), q1[2].item(), q1[3].item(), q1[0].item()])
        q2_scipy = R.from_quat([q2[1].item(), q2[2].item(), q2[3].item(), q2[0].item()])
        expected = torch.from_numpy((q1_scipy * q2_scipy).as_quat()).type(q1.dtype)
        expected = torch.tensor([expected[3], expected[0], expected[1], expected[2]])
        result = quaternion_multiply(q1, q2)
        assert torch.allclose(result, expected, atol=1e-6), "Scipy comparison failed"

class TestVectorRotation:
    @pytest.mark.parametrize(
        "vector, quaternion, expected",
        [
            (
                torch.tensor([1.0, 0.0, 0.0], ),
                torch.tensor([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
                torch.tensor([0.0, 1.0, 0.0]),
            ),

        ],
    )
    def test_vector_rotation(self, vector, quaternion, expected):
        result = rotate_vector(vector, quaternion)
        assert torch.allclose(result, expected, atol=1e-6), f"Rotation failed: expected {expected}, got {result}"


    def test_compare_with_scipy(self, test_vector):
        angle = np.pi / 3
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        scipy_quat = R.from_rotvec(angle * axis).as_quat()
        our_quat = torch.tensor([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])
        our_result = rotate_vector(test_vector, our_quat)
        scipy_result = R.from_quat(scipy_quat).apply(test_vector.numpy())
        scipy_result = torch.from_numpy(scipy_result).type(our_result.dtype)
        assert torch.allclose(our_result, scipy_result, atol=1e-6), "Scipy comparison failed"


    def test_batch_rotation(self):
        vectors = torch.randn(10, 3)
        angles = torch.randn(10) * np.pi
        quats = torch.stack(
            [torch.cos(angles / 2), torch.zeros_like(angles), torch.zeros_like(angles), torch.sin(angles / 2)], dim=1
        )
        rotated = rotate_vector(vectors, quats)
        assert rotated.shape == vectors.shape, "Batch rotation shape mismatch"

        for i in range(10):
            single_rotated = rotate_vector(vectors[i], quats[i])
            assert torch.allclose(single_rotated, rotated[i]), f"Mismatch in batch item {i}"

class TestRotateVectorInverse:
    def test_identity_rotation(self):
        """Test that rotation with identity quaternion does not change the vector."""
        v = torch.tensor([1.0, 0.0, 0.0])
        q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = rotate_vector_inverse(v, q_identity)
        assert torch.allclose(result, v, atol=1e-6)

    def test_rotation_inverse_property(self):
        """Test that rotating a vector by a quaternion's inverse undoes the rotation."""
        v = torch.tensor([1.0, 0.0, 0.0])

        # 90-degree rotation around Z-axis
        q = torch.tensor([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=torch.float32)
        
        rotated = rotate_vector(v, q)
        inverted = rotate_vector_inverse(rotated, q)
        assert torch.allclose(inverted, v, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare rotation inverse with scipy's implementation."""
        from scipy.spatial.transform import Rotation as R

        # Random vector and quaternion
        v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        angle = np.pi / 3
        axis = np.array([1., 0., 0.])
        axis = axis / np.linalg.norm(axis)

        # Quaternion [w, x, y, z]
        scipy_quat = R.from_rotvec(angle * axis).as_quat()
        q = torch.tensor([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]], dtype=torch.float32)

        # Rotate vector with inverse quaternion
        result = rotate_vector_inverse(v, q)

        # Scipy rotation
        scipy_rot = R.from_quat(scipy_quat).inv()
        scipy_result = scipy_rot.apply(v.numpy())

        assert torch.allclose(result, torch.from_numpy(scipy_result).type(torch.float32), atol=1e-6)

    def test_batch_inverse_rotation(self):
        """Test batch rotation inverse with multiple vectors and quaternions."""
        vectors = torch.randn(10, 3)
        angles = torch.randn(10) * np.pi
        quats = torch.stack([
            torch.cos(angles / 2),
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            torch.sin(angles / 2)
        ], dim=1)

        # Rotate vectors
        rotated = rotate_vector(vectors, quats)

        # Rotate back using inverse
        restored = rotate_vector_inverse(rotated, quats)

        assert torch.allclose(restored, vectors, atol=1e-6)

    def test_zero_rotation(self):
        """Test with a zero rotation quaternion."""
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # Identity quaternion
        result = rotate_vector_inverse(v, q)
        assert torch.allclose(result, v, atol=1e-6)

    def test_small_angles(self):
        """Test rotation inverse for very small rotation angles."""
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        angle = 1e-6  # Small angle
        q = torch.tensor([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], dtype=torch.float32)

        rotated = rotate_vector(v, q)
        inverted = rotate_vector_inverse(rotated, q)

        assert torch.allclose(inverted, v, atol=1e-6)

    def test_large_rotations(self):
        """Test rotation inverse for angles larger than 2π."""
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        angle = 4 * np.pi  # Large angle
        q = torch.tensor([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], dtype=torch.float32)

        rotated = rotate_vector(v, q)
        inverted = rotate_vector_inverse(rotated, q)

        # Should result in the same vector since rotation is effectively 0 mod 2π
        assert torch.allclose(inverted, v, atol=1e-6)


class TestQuaternionConjugate:
    def test_single_conjugate(self):
        """Test quaternion conjugate for a single quaternion."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0])
        result = quaternion_conjugate(q)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare quaternion conjugate with scipy's implementation."""
        from scipy.spatial.transform import Rotation as R

        # Random quaternion
        q = torch.tensor([0.707, 0.0, 0.707, 0.0], dtype=torch.float32)  # [w, x, y, z]
        scipy_quat = [q[1].item(), q[2].item(), q[3].item(), q[0].item()]  # Scipy uses [x, y, z, w]

        # Scipy conjugate
        scipy_conjugate = R.from_quat(scipy_quat).inv().as_quat()
        expected = torch.tensor([scipy_conjugate[3], scipy_conjugate[0], scipy_conjugate[1], scipy_conjugate[2]],
                                dtype=torch.float32)

        # Function result
        result = quaternion_conjugate(q)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_batch_conjugate(self):
        """Test quaternion conjugate for a batch of quaternions."""
        q_batch = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
            [0.0, 1.0, 0.0, 0.0],  # 180-degree rotation around X-axis
            [0.0, 0.0, 1.0, 0.0],  # 180-degree rotation around Y-axis
            [0.0, 0.0, 0.0, 1.0]  # 180-degree rotation around Z-axis
        ])
        expected = torch.tensor([
            [1.0, -0.0, -0.0, -0.0],
            [0.0, -1.0, -0.0, -0.0],
            [0.0, -0.0, -1.0, -0.0],
            [0.0, -0.0, -0.0, -1.0]
        ])
        result = quaternion_conjugate(q_batch)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_batch_with_random_values(self):
        """Test quaternion conjugate for a large batch of random quaternions."""
        q_batch = torch.randn(1000, 4)
        result = quaternion_conjugate(q_batch)
        expected = torch.cat([q_batch[..., :1], -q_batch[..., 1:]], dim=-1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_zero_quaternion(self):
        """Test quaternion conjugate with a zero quaternion."""
        q = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        expected = torch.tensor([0.0, -0.0, -0.0, -0.0], dtype=torch.float32)
        result = quaternion_conjugate(q)
        assert torch.allclose(result, expected, atol=1e-6)
