from typing import List

import warp as wp
import numpy as np
import warp.sim.render
from scipy.spatial.transform import Rotation as R

from dp_utils.trajectory import Trajectory

class Renderer:
    def __init__(self, model: wp.sim.Model, time: np.ndarray, output_file: str):
        self.time = time
        self.model = model
        self.renderer = wp.sim.render.SimRenderer(model, output_file, scaling=1.0)

        self.trajectories = []

    def save(self, states: List[wp.sim.State], fps: int = 30):
        assert len(self.time) <= len(states), "Time and states must have the same length."

        frame_interval = 1.0 / fps  # time interval per frame
        last_rendered_time = 0.0    # tracks the time of the last rendered frame

        print("Creating USD render...")
        for t, state in zip(self.time, states):
            if t >= last_rendered_time:  # render only if enough time has passed
                self.renderer.begin_frame(t)
                self.renderer.render(state)
                for trajectory in self.trajectories:
                    self.render_trajectory(trajectory)

                self.renderer.end_frame()
                last_rendered_time += frame_interval  # update to next frame time

        self.renderer.save()


    def add_trajectory(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)

    def render_trajectory(self, trajectory: Trajectory):
        self.renderer.register_body(trajectory.name)

        positions = trajectory.pos.numpy()

        # Render the trajectory
        current_pos = positions[0]
        for i in range(1, len(positions) - 1):
            next_pos = positions[i]

            # Skip if the current and next positions are the same
            if np.allclose(current_pos, next_pos, atol=1e-6):
                continue
            
            pos, rot, half_height = _compute_segment_xform(current_pos, next_pos)

            self.renderer.render_capsule(name=f"c{i}", 
                                         pos=pos,
                                         rot=rot,
                                         color=trajectory.color.rgb,
                                         half_height=half_height,
                                         parent_body=trajectory.name,
                                         radius=trajectory.render_radius)
            
            current_pos = next_pos



def _compute_segment_xform(p1: np.ndarray, p2: np.ndarray):
    """
    Computes the transformation for a capsule segment between two 3D points.

    Args:
        p1 (np.ndarray): Starting point of the segment.
        p2 (np.ndarray): Ending point of the segment.
    
    Returns:
        tuple: (position (np.ndarray), rotation (np.ndarray), half_height (float)):
            - position: Midpoint of the segment.
            - rotation: Quaternion representing the rotation to align with the segment.
            - half_height: Half the length of the segment.
    """
    position = (p1 + p2) / 2
    height = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / height
    default_axis = np.array([0.0, 1.0, 0.0])
    rotation, _ = R.align_vectors([direction], [default_axis])
    return position, rotation.as_quat(), height / 2