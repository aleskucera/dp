import os
import math
from typing import Tuple, Dict, Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import warp as wp
import warp.sim
import warp.sim.render
import matplotlib.pyplot as plt

from dp_utils import *

class Wheel:
    def __init__(self,
                 radius: float,
                 inertia: float,
                 damping: float,
                 max_torque: float,
                 max_speed: float,
                 ):
        self.radius = radius
        self.inertia = inertia
        self.damping = damping
        self.max_torque = max_torque
        self.max_speed = max_speed

class Robot:
    def __init__(self,
                 name: str):
        self.name = name
        self.builder = wp.sim.ModelBuilder()