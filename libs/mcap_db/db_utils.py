import collections
import numpy as np
import transforms as tf
from scipy.spatial.transform import Rotation as R
from .mcap_db import McapDB
import open3d as o3d
import ros2_utils
import matplotlib.pyplot as plt

def iter_seq(msgs, every=1, skip=0, limit=None):
    count = 0
    recorded = 0
    for t in msgs:
        count += 1
        if count <= skip:
            continue
        if (count - skip) % every == 0:
            yield t
            recorded += 1
            if limit is not None and recorded >= limit:
                break

def msg_to_value(item, name_to_indices):
    ts = item['timestamp'] * 1e-9 
    results = {}
    for name, indices in name_to_indices.items():
        value = item
        for idx in indices:
            value = value[idx]
        results[name] = value
    return results, ts

def msgs_to_values(topic_db, name_to_indices, every=1, skip=0, limit=None):
    """Get sensor values
    Args:
      topic_db: THe topic db
      name_to_indices: {'return_key': ['level', 'of', 'indices']}
    """
    results = collections.defaultdict(list)
    for t in iter_seq(topic_db, every, skip, limit):
        item = topic_db[t]
        fields, ts = msg_to_value(item, name_to_indices)
        results['ts'].append(ts)
        for name, value in fields.items():
            results[name].append(value)
    ts = results['ts']
    print(f'Got {len(ts)} values in {ts[-1] - ts[0]} seconds')
    return results

def msg_to_image(item, use_received_time=True):
    assert 'timestamp' in item and 'msg' in item
    received_time = item['timestamp']
    msg = item['msg']
    if isinstance(msg, list):
        image = np.array(msg).astype(float)
        if np.max(image) <= 1.0:
            image = (image * 255.0)
        image = image.astype(np.uint8)
        captured_ns = received_time
    else:
        image = jdb.dict_to_image(msg)
        captured_ns = received_time if use_received_time else stamp_to_ns(msg['header']['stamp'])
    return image, captured_ns * 1e-9

def msgs_to_images(msgs, every=1, skip=0, use_received_time=True, limit=None):
    images = collections.OrderedDict()
    for t in iter_seq(msgs, every, skip, limit):
        image, captured_time_s = msg_to_image(msgs[t], use_received_time)
        images[captured_time_s] = image
    return images

# ROS / JDB Data Retrieval
def stamp_to_ns(stamp):
    return stamp['sec'] * 1e9 + stamp['nanosec']

def ns_to_ts(ns):
    return ns * 1e-9

def msg_to_R(q):
    quat = np.array([q.x, q.y, q.z, q.w])
    norm = np.linalg.norm(quat)
    if norm < 1e-5 or np.isnan(norm):
        return
    quat /= norm
    return R.from_quat(quat)

def msg_to_vec3(translation):
    return np.array([translation.x, translation.y, translation.z])


def _pose_to_coordframe(T, q):
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


class PoseSequence:
    def __init__(self):
        self.pub_ts = []
        self.tf_ts = []
        self.xyz = []
        self.yaw_pitch_roll = []
        self.Rs = []

    def __len__(self):
        return len(self.pub_ts)

    @property
    def x(self):
        return self.xyz[:, 0]

    @property
    def y(self):
        return self.xyz[:, 1]

    @property
    def z(self):
        return self.xyz[:, 2]

    @property
    def yaw(self):
        return self.yaw_pitch_roll[:, 0]

    @property
    def pitch(self):
        return self.yaw_pitch_roll[:, 1]

    @property
    def roll(self):
        return self.yaw_pitch_roll[:, 2]

    def append(self, pub_ts, ts, translation, rot):
        assert(isinstance(rot, R))
        self.pub_ts.append(pub_ts)
        self.tf_ts.append(ts)
        self.xyz.append(translation)
        # yaw, pitch, roll
        eulers = rot.as_euler('ZYX', degrees=False)
        self.yaw_pitch_roll.append(eulers)
        self.Rs.append(rot)

    def predict(self):
        self.predicted = []
        estimator = EstimatorD0()
        for ts, T, rot in zip(self.tf_ts, self.xyz, self.Rs):
            state = MultiBodyState.create_body_d0(T, rot, ts)
            estimator.update(state)
            next_state = estimator.predict(ts + 0.3)

            x_axis = nu.normalize(next_state.vel, axis=0)
            if not np.any(np.isnan(x_axis)):
                z_axis = [0.0, 0.0, 1.0]
                y_axis = np.cross(z_axis, x_axis)
                R = tf.axes_to_R('xyz', [x_axis, y_axis], normalize=False)
                pose = tf.create_tf(T, R)
                self.predicted.append(pose)
            else:
                self.predicted.append(None)

    def get_tfs(self, base_pose=None):
        tfs = []
        for T, rot in zip(self.xyz, self.Rs):
            pose = tf.create_tf(T, rot)
            if base_pose is not None:
                pose = tf.combine_tfs(base_pose, pose)
            tfs.append(pose)
        return tfs

    def get_tf(self, index):
        return tf.create_tf(self.xyz[index], self.Rs[index])

    def get_geoms(self, base_pose=None):
        geoms = []
        for pose in self.get_tfs(base_pose):
            T = pose.T
            q = pose.R.as_quat()
            geoms.append(_pose_to_coordframe(T, q))
        return geoms

    def finalize(self, base_pub_ts):
        self.xyz = np.array(self.xyz) if len(self.xyz) else np.zeros((0, 3))
        self.yaw_pitch_roll = np.array(self.yaw_pitch_roll) if len(self.yaw_pitch_roll) else np.zeros((0, 3))
        self.tf_ts = np.array(self.tf_ts)
        self.pub_ts = np.array(self.pub_ts) - base_pub_ts

    @staticmethod
    def find_matching_poses(sequences, tol_delta_ts=0.1):
        """
        Find groups of poses that are temporally close across multiple sequences.
        
        Args:
            sequences: List of N PoseSequences
            tol_delta_ts: Maximum time difference (in seconds) between poses to be considered matching
        
        Returns:
            List of N-tuples containing indices into each sequence. None if no matching pose found.
            Each group contains at least 2 valid poses.
        """
        if not sequences:
            return []

        # Get all unique timestamps across all sequences
        all_timestamps = []
        for seq_idx, seq in enumerate(sequences):
            all_timestamps.extend((ts, seq_idx) for ts in seq.tf_ts)
        
        # Sort timestamps with their sequence indices
        all_timestamps.sort()  # Sorts by timestamp first, then sequence index
        
        # Initialize result
        matching_groups = []
        current_group = [None] * len(sequences)
        current_ts = None
        
        for ts, seq_idx in all_timestamps:
            # If this is a new timestamp group
            if current_ts is None or ts - current_ts > tol_delta_ts:
                # Check if previous group has at least 2 valid poses
                valid_poses = sum(1 for idx in current_group if idx is not None)
                if valid_poses >= 2:
                    matching_groups.append(tuple(current_group))
                
                # Start new group
                current_group = [None] * len(sequences)
                current_ts = ts
            
            # Find index in the sequence for this timestamp
            seq = sequences[seq_idx]
            pose_idx = np.searchsorted(seq.tf_ts, ts)
            if pose_idx < len(seq.tf_ts) and abs(seq.tf_ts[pose_idx] - ts) <= tol_delta_ts:
                current_group[seq_idx] = pose_idx
        
        # Don't forget to check the last group
        valid_poses = sum(1 for idx in current_group if idx is not None)
        if valid_poses >= 2:
            matching_groups.append(tuple(current_group))
        
        return matching_groups


    @staticmethod
    def create_connection_geometries(sequences, matching_groups, base_pose=None):
        """
        Create line segments connecting corresponding poses across sequences.

        Args:
            sequences: List of N PoseSequences
            matching_groups: List of N-tuples containing indices into each sequence
            base_pose: Optional base transform to apply to all poses
        
        Returns:
            List of Open3D LineSet objects, each representing connections between poses
            at a specific timestamp
        """
        connection_geoms = []
        
        for group in matching_groups:
            # Get positions of valid poses in this group
            positions = []
            for seq_idx, pose_idx in enumerate(group):
                if pose_idx is not None:
                    seq = sequences[seq_idx]
                    T = seq.xyz[pose_idx]
                    if base_pose is not None:
                        # Apply base transform if provided
                        pose = tf.create_tf(T, seq.Rs[pose_idx])
                        pose = tf.combine_tfs(base_pose, pose)
                        T = pose.T
                    positions.append(T)
            
            if len(positions) < 2:
                continue
                
            # Create lines between all pairs of valid poses
            points = np.array(positions)
            lines = []
            colors = []
            
            # Create lines between consecutive valid poses
            for i in range(len(positions) - 1):
                lines.append([i, i + 1])
                colors.append([1, 0, 0])  # Red color for lines
            
            # Create Open3D LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            connection_geoms.append(line_set)
        
        return connection_geoms

def meter_to_inch(m):
    return m * 39.3701

def at_dict(info, index):
    return info[list(info.keys())[index]]


def get_event_from_mcap(db, event_id, topics, event_topic, base_ts=None, elapsed_ts_range=None):
    if event_id is None:
        current_event_ns = next_event_ns = None
    else:
        events = db.topic(event_topic)
        next_event_ns = events[event_id+1].pub_ns if event_id + 1 < len(events) else None
        current_event_ns = None if event_id < 0 else events[event_id].pub_ns

    available_topics = db.topics()

    out_event_db = collections.defaultdict(dict)
    for topic in topics:
        if topic not in available_topics:
            continue
        topic_db = db.topic(topic)
        for msg in topic_db:
            ns = msg.pub_ns
            if next_event_ns is not None and ns >= next_event_ns:
                break
            if elapsed_ts_range is not None:
                elapsed_ts = ns * 1e-9 - base_ts
                if not (elapsed_ts >= elapsed_ts_range[0] and elapsed_ts < elapsed_ts_range[1]):
                    continue
            if current_event_ns is None or ns >= current_event_ns:
                out_event_db[topic][ns * 1e-9] = msg.data

    return out_event_db


def get_series(msg_by_ts, retriever_callback, base_ts=0.0, is_data_array=True):
    timestamps = sorted(list(msg_by_ts.keys()))
    series = {}
    series['ts'] = np.array(timestamps).astype(float) - base_ts
    series['data'] = [retriever_callback(msg_by_ts[ts]) for ts in timestamps]
    if is_data_array:
        series['data'] = np.array(series['data'])
    return series

def get_header_ts(msg_by_ts):
    data = get_series(msg_by_ts, lambda msg: msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                      base_ts=0.0, is_data_array=True)
    return data['ts'], data['data']

def topic_db_to_images(topic_db):
    image_seq = collections.OrderedDict()
    for ts in topic_db:
        msg = topic_db[ts]
        try:
            image = McapDB.get_image(msg)
        except:
            import pdb; pdb.set_trace()
            continue
        image_seq[ts] = image
    return image_seq


def topic_db_to_images(topic_db):
    image_seq = collections.OrderedDict()
    for ns in topic_db:
        msg = topic_db[ns]
        try:
            image = McapDB.get_image(msg)
        except:
            import pdb; pdb.set_trace()
            continue
        captured_time_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        image_seq[captured_time_s] = image
    return image_seq


def images_from_event_db(db, event_id, event_topic, image_topics=None, other_topics=None):
    image_topics = [] if image_topics is None else image_topics
    other_topics = [] if other_topics is None else other_topics
    event_db = get_event_from_mcap(
        db, event_id, image_topics + other_topics, event_topic)
    for topic in event_db.keys():
        print(topic, len(event_db[topic]))

    results = {}
    for topic in image_topics:
        if topic not in event_db.keys():
            images = {}
        else:
            topic_db = event_db[topic]
            images = topic_db_to_images(topic_db)
        results[topic] = images

    for topic in other_topics:
        results[topic] = {ns * 1e-9: msg for ns, msg in event_db[topic].items()}

    return results

def topic_dict_per_event_db(db, event_id, event_topic, topics=None, base_ts=None, elapsed_ts_range=None):
    topics = db.topics() if topics is None else topics
    event_db = get_event_from_mcap(
        db, event_id, topics, event_topic, base_ts, elapsed_ts_range)
    for topic in event_db.keys():
        print(topic, len(event_db[topic]))

    results = {}
    for topic in topics:
        results[topic] = {ts: msg for ts, msg in event_db[topic].items()}

    return results

def get_intrinsics_from_caminfo(cam_info):
    # See below for details on camera matrix
    # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # fx = 340.81
    # fy = 340.81
    # cx = 328.81
    # cy = 189.81
    #
    #camera_intrinsics_matrix = np.array([
    #  [fx, 0, cx],
    #  [0, fy, cy],
    #  [0,  0,  1]])
    intrinsics = np.array(cam_info.k).reshape((3, 3))

    # [k1, k2, p1, p2, k3]
    #k1 = k2 = p1 = p2 = k3 = 0.0
    #distortion_coeffs = np.array([k1, k2, p1, p2, k3])
    distortion = np.array(cam_info.d)
    return intrinsics, distortion

def get_intrinsics_from_camconfig(cam_config):
    intrinsics_matrix = np.array([
      [cam_config.fx, 0, cam_config.cx],
      [0, cam_config.fy, cam_config.cy],
      [0,  0,  1]])
    distortion = np.array(cam_config.distortion_coeffs)
    return intrinsics_matrix, distortion

def get_tftree(messages, is_static=None):
    if is_static is None:
        topics = ['/tf_static', '/tf']
    elif is_static:
        topics = ['/tf_static']
    else:
        topics = ['/tf']

    tf_tree = tf.TfTree()
    for topic in topics:
        msgs = get_series(messages[topic], lambda x:x, base_ts=0.0)
        for pub_ts, msg in zip(msgs['ts'], msgs['data']):
            for transform in msg.transforms:
                tf_tree.add(tf.Transform(
                    transform.header.frame_id,
                    transform.child_frame_id,
                    pub_ts,
                    msg_to_vec3(transform.transform.translation),
                    msg_to_R(transform.transform.rotation)))
    return tf_tree

def merge_topic_messages(topic_msgs_dict):
    """
    Merge messages from multiple topics in chronological order.
    
    Args:
        topic_msgs_dict: Dictionary mapping topic names to their message series
                        Each series should have 'ts' and 'data' keys
    
    Yields:
        Tuple of (topic, pub_ts, msg) in chronological order
    """
    # Initialize indices for each topic
    current_indices = {
        topic: 0 for topic, msgs in topic_msgs_dict.items() 
        if len(msgs['ts']) > 0
    }
    
    # Continue while we have messages from any topic
    while current_indices:
        # Find topic with smallest next timestamp
        next_topic = min(current_indices.keys(), 
                        key=lambda t: topic_msgs_dict[t]['ts'][current_indices[t]])
        
        # Get the next message from this topic
        idx = current_indices[next_topic]
        pub_ts = topic_msgs_dict[next_topic]['ts'][idx]
        msg = topic_msgs_dict[next_topic]['data'][idx]
        
        # Increment the index for this topic
        current_indices[next_topic] += 1
        
        # Remove topic if we've reached the end of its messages
        if current_indices[next_topic] >= len(topic_msgs_dict[next_topic]['ts']):
            del current_indices[next_topic]
        
        yield next_topic, pub_ts, msg


# Example usage in your code:
def get_tf(messages, src_frame, dest_frame, base_ts, is_static=None, do_sample_upon_dest_frame=True):
    sequence = PoseSequence()
    tf_tree = tf.TfTree()
    
    if is_static is None:
        topics = ['/tf_static', '/tf']
    elif is_static:
        topics = ['/tf_static']
    else:
        topics = ['/tf']

    # Prepare messages from all topics
    topic_msgs = {}
    for topic in topics:
        # Set base_ts is 0.0 because it will eventually be re-based in finalize()
        topic_msgs[topic] = get_series(messages[topic], lambda x:x, base_ts=0.0)

    # Process messages in chronological order
    for topic, pub_ts, msg in merge_topic_messages(topic_msgs):
        for transform in msg.transforms:
            tf_ts = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            tf_tree.add(tf.Transform(
                transform.header.frame_id,
                transform.child_frame_id,
                tf_ts,
                msg_to_vec3(transform.transform.translation),
                msg_to_R(transform.transform.rotation)))
            if (not do_sample_upon_dest_frame) or transform.child_frame_id == dest_frame:
                tfm = tf_tree.query(src_frame, dest_frame)
                if tfm is not None:
                    sequence.append(pub_ts,
                                    tfm.timestamp,
                                    tfm.T,
                                    tfm.R)
    
    sequence.finalize(base_ts)
    return sequence

def find_index_by_time(timestamps, target_time):
    index = np.argmin(np.abs(timestamps - target_time))
    error = timestamps[index] - target_time
    return index, error

def _unpack_rgba(rgba_values):
    # Convert to a NumPy array if it's not already
    rgba_values = np.array(rgba_values, dtype=np.uint32)

    # Unpack the RGBA components
    red = (rgba_values & 0xFF).astype(np.uint8)
    green = ((rgba_values >> 8) & 0xFF).astype(np.uint8)
    blue = ((rgba_values >> 16) & 0xFF).astype(np.uint8)
    alpha = ((rgba_values >> 24) & 0xFF).astype(np.uint8)

    # Stack the components into an Nx4 array
    unpacked_values = np.stack((red, green, blue, alpha), axis=-1)

    return unpacked_values

def get_pointcloud(messages, topic, base_ts):

    map_bed = get_series(messages[topic], lambda x: x, base_ts= base_ts)
    points = []
    timestamps = []
    counts = []
    for i, pointcloud in enumerate(map_bed['data']):
        frame_points = np.array(list(ros2_utils.read_points(pointcloud, skip_nans=True)))
        if len(frame_points) == 0:
            continue

        # Create a column with the constant value
        constant_column = np.full((len(frame_points), 1), i)

        # Append the column to the original array
        new_array = np.hstack((frame_points, constant_column))

        points.append(new_array)
        timestamps.append(map_bed['ts'][i])
        counts.append(len(new_array))

    if len(points) == 0:
        zeros = np.zeros((0, 3))
        return zeros, np.zeros((0, 4)).astype(np.uint32), zeros.astype(np.uint32), np.array(timestamps), np.array([])
    points = np.concatenate(points, axis=0)
    xyz = points[:, :3]
    rgba = _unpack_rgba(points[:, 3].astype(np.uint32))
    frames = points[:, 4]

    # Normalize integers to the range [0, 1]
    normalized_integers = frames / np.max(frames)
    # Use a colormap (e.g., 'viridis', 'jet', 'rainbow', etc.)
    colormap = plt.cm.jet
    frame_rgb = (colormap(normalized_integers)[:,:3] * 255.9).astype(np.uint32)  # Extract RGB values
    return xyz, rgba, frame_rgb, np.array(timestamps), np.array(counts)

def decode_pointcloud(pointcloud):
    points = np.array(list(ros2_utils.read_points(pointcloud, skip_nans=True)))
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint32)
    xyz = points[:, :3]
    rgba = _unpack_rgba(points[:, 3].astype(np.uint32))
    colors = rgba[:, :3].astype(np.float32) / 255.0
    return xyz, colors

