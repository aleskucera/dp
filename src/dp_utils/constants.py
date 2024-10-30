import os

# import matplotlib.colors as mcolors

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        os.pardir, 
        os.pardir
    )
)
CONF_DIR = os.path.join(PROJECT_DIR, "conf")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
SRC_DIR = os.path.join(PROJECT_DIR, "src")

# class GEO_TYPE(Enum):
#     SPHERE = wp.constant(0)
#     BOX = wp.constant(1)
#     CAPSULE = wp.constant(2)
#     CYLINDER = wp.constant(3)
#     CONE = wp.constant(4)
#     MESH = wp.constant(5)
#     SDF = wp.constant(6)