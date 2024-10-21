import warp as wp

# Initialize a source 2D array (e.g., 4x4)
source_array = wp.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]], dtype=wp.float32, device='cuda')

# Create a destination 2D array (e.g., 4x4) with zeros
dest_array = wp.zeros((4, 4), dtype=wp.float32, device='cuda')

# Copy a submatrix from the source array to the destination array
# For example, copy a 2x2 block (top-left corner of source to bottom-right of destination)
wp.copy(dest_array[2:4, 2:4], source_array[0:2, 0:2])

# Print the result to check if the copy worked
print("Source array:")
print(source_array[0])

print("Destination array after copy:")
print(dest_array)
