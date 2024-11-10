import warp as wp

# Define the torque and force components
torque = (1.0, 0.0, 0.0)  # Replace with your desired torque values
force = (0.0, 9.8, 0.0)   # Replace with your desired force values

# Create the spatial_vectorf by combining torque and force components
spatial_force = wp.spatial_vectorf(*torque, *force)

print(spatial_force)