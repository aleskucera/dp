import pytest
import torch

from xpbd_pytorch.correction import *

class TestCollisionDelta:
    @pytest.mark.parametrize(
        "q, r, n, c, expected_dx, expected_dq",
        [
            # Collision without rotation
            (torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # Identity quaternion
             torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),  # Point of contact
             torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),  # Normal
             0.1,  # Depth
             torch.tensor([0.0, 0.0, 0.1], dtype=torch.float32),  # Change in position
             torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # Change in rotation
            ),
            # Collision mainly with rotation
            (torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # Identity quaternion
             torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),  # Point of contact
             torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),  # Normal
             0.1,  # Depth
             torch.tensor([0.0, 0.0, 0.05], dtype=torch.float32),  # Change in position
             torch.tensor([0.0, 0.0, -0.0250, 0.0], dtype=torch.float32),  # Change in rotation
            ),
        ],
    )
    def test_collision_delta(self, q, r, n, c, expected_dx, expected_dq):
        """
        Test collision delta for various contact points, normals, and depths.
        """

        # Create the inverse inertia tensor
        m_inv = 1.0
        I_inv = torch.eye(3)

        # Compute the collision delta
        dx, dq = collision_delta(q, m_inv, I_inv, r, n, c)

        # Assert the computed change in position and rotation match the expected ones
        assert torch.allclose(dx, expected_dx, atol=1e-2)
        assert torch.allclose(dq, expected_dq, atol=1e-2)
