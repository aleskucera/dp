import warp as wp
import warp.sim
from omegaconf import DictConfig

from dp_utils.joints import set_joint_config
from dp_utils.general import get_robot_transform


def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    b = builder.add_body(origin=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()), name="ball")
    builder.add_shape_sphere(body=b, radius=0.1, density=100.0, ke=2000.0, kd=100.0, kf=200.0, mu=0.2)
    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=0.2)
    model = builder.finalize(requires_grad=True)

    return model

def pendulum_world_model(cfg: DictConfig, 
                         wall: bool = False) -> wp.sim.Model:
    assert cfg.robot.name == "pendulum"
    
    builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

    # Add the robot to the model
    parse_urdf_args = dict(cfg.robot.parse_urdf_args)
    parse_urdf_args["xform"] = get_robot_transform(cfg)
    parse_urdf_args["builder"] = builder
    wp.sim.parse_urdf(**parse_urdf_args)
    set_joint_config(cfg, builder)

    if wall:
        w = builder.add_body(origin=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()), name="wall")
        builder.add_shape_box(body=w, pos=wp.vec3(0.0, 0.0, 0.0),
                              hx=1.0, hy=0.25, hz=1.0,
                              ke=1e4, kf=0.0, kd=1e2, mu=0.2,
                              has_ground_collision=False)
    
    model = builder.finalize()
    model.ground = True
    return model

def carter_world_model(cfg: DictConfig) -> wp.sim.Model:
    assert cfg.robot.name == "carter"

    raise NotImplementedError("Carter model not implemented yet")