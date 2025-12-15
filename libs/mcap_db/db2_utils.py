


import o3d

def pose_to_coordframe(T, q):
    # Extract position
    position = np.array(T)

    # Extract orientation and convert quaternion to rotation matrix
    quaternion = np.array(q)
    if np.linalg.norm(quaternion) < 0.9 or np.any(np.isnan(quaternion)):
        return None
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create an Open3D coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coordinate_frame.translate(position)
    coordinate_frame.rotate(rotation_matrix, center=position)
    return coordinate_frame

def array2color(data, colormap_name, zero_color):
    # Create a colormap instance
    colormap = plt.get_cmap(colormap_name)

    # Normalize data to range [0, 1]
    norm = mcolors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))

    # Apply colormap
    scalar_mapped = colormap(norm(data))
    scalar_mapped[(data == 0) | (np.isnan(data))] = zero_color

    # Exclude alpha channel and convert to RGB format
    return scalar_mapped[:, :, :3]

