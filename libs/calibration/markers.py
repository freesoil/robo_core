import cv2
import cv2.aruco as aruco
import collections
import numpy as np
import numpy_utils as nu
import munch
from scipy.spatial.transform import Rotation as R
import transforms as tf
import rolog
import matplotlib.pyplot as plt

def create_marker_tf(tvec, rvec):
    return munch.munchify({
                    'R': R.from_rotvec(rvec.flatten()),
                    'T': np.array(tvec).flatten(),
                })


def detect_fiducial(frame):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
    parameters = aruco.DetectorParameters_create()

    frame = frame.copy()

    ## Might need to adjust based on lighting conditions
    #parameters.adaptiveThreshConstant = 7
    ## Adjust for larger markers
    #parameters.maxMarkerPerimeterRate = 10
    ## Adjust for smaller markers
    #parameters.minMarkerPerimeterRate = 0.01
    ## Lower values can detect more markers but might increase false positives
    #parameters.polygonalApproxAccuracyRate = 0.03
    #parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    
    #parameters.adaptiveThreshWinSizeMin = 3
    #parameters.adaptiveThreshWinSizeStep = 20
    # Helps in distinguishing between close markers
    #parameters.minCornerDistanceRate = 0.01
    #parameters.perspectiveRemovePixelPerCell = 10
    #parameters.minOtsuStdDev = 8
    #parameters.maxErroneousBitsInBorderRate = 0.7
    #parameters.errorCorrectionRate = 0.9
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    return corners, ids


def detect_marker_poses(image, marker_length, intrinsics, distortion):
    """Try multiple detection strategies for robustness."""
    import cv2.aruco as aruco
    def detect_marker_poses_raw(frame, marker_length, camera_intrinsics_matrix, distortion_coeffs):
        corners, ids = detect_fiducial(frame)
        if ids is None or len(ids) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        ids = np.array(ids)[:,0]
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_intrinsics_matrix, distortion_coeffs)
        return corners, ids, rvecs, tvecs
    # Strategy 1: Direct detection on original image
    corners1, ids1, rvecs1, tvecs1 = detect_marker_poses_raw(image, marker_length, intrinsics, distortion)
    
    # Strategy 2: Detection on enhanced image
    enhanced_image = enhance_for_marker_detection(image)
    corners2, ids2, rvecs2, tvecs2 = detect_marker_poses_raw(enhanced_image, marker_length, intrinsics, distortion)
    
    # Combine results - prefer strategy with most detections
    strategies = [
        (corners1, ids1, rvecs1, tvecs1),
        (corners2, ids2, rvecs2, tvecs2),
    ]
    
    best_strategy = max(strategies, key=lambda x: len(x[1]) if x[1] is not None else 0)
    return best_strategy

def enhance_for_marker_detection(image):
    """Apply multiple enhancement techniques for better marker detection."""
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    # Convert back to RGB for consistency
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


def est_marker_poses_and_image_locs(
        image, marker_length, intrinsics, distortion,
        xyz_cam=None, specified_ids=None, enable_enhance=False, enable_enlarge=False):

    h, w = image.shape[:2]
    if enable_enlarge:
        h *= 2
        w *= 2
        image = cv2.resize(image, (w, h))
        intrinsics = np.copy(intrinsics)
        intrinsics[0, 0] *= 2
        intrinsics[1, 1] *= 2
        intrinsics[0, 2] *= 2
        intrinsics[1, 2] *= 2

    if enable_enhance:
        image = enhance_for_marker_detection(image)

    corners, visible_marker_ids, rvecs, tvecs = detect_marker_poses(
        image, marker_length, intrinsics, distortion)

    image_locs = {}
    tf_cam_to_markers = {}
    specified_ids = visible_marker_ids if specified_ids is None else specified_ids
    for marker_id in specified_ids:
        if marker_id not in visible_marker_ids:
            continue
        pos = visible_marker_ids.tolist().index(marker_id)
        if xyz_cam is None:
            translation = tvecs[pos]
        else:
            # Corrected translation by using the depth output.
            marker_corners = corners[pos][0]
            center = np.mean(marker_corners, axis=0)
            halves = (marker_corners + center) * 0.5
            samples = np.concatenate((halves, marker_corners, [center]))
            samples[:, 1] *= float(xyz_cam.shape[0]) / h
            samples[:, 0] *= float(xyz_cam.shape[1]) / w
            samples = samples.astype(int)
            samples[:, 1] = np.clip(samples[:, 1], 0, xyz_cam.shape[0]-1)
            samples[:, 0] = np.clip(samples[:, 0], 0, xyz_cam.shape[1]-1)
            translations = xyz_cam[samples[:, 1], samples[:, 0]]
            translations = translations[~np.isnan(np.sum(translations, axis=1))]
            translations = nu.select_percentile_range(translations, 0.3, 0.7)
            mask = (np.sum(np.isnan(translations), axis=1) == 0)
            if np.sum(mask) == 0:
                translation = tvecs[pos]
            else:
                translation = np.mean(translations[mask], axis=0)
        pose = create_marker_tf(translation, rvecs[pos])
        tf_cam_to_markers[marker_id] = pose
        image_locs[marker_id] = np.mean(corners[pos], axis=1)

    return tf_cam_to_markers, image_locs

def est_marker_poses(
        image, marker_length, intrinsics, distortion,
        xyz_cam=None, specified_ids=None, enable_enhance=False, enable_enlarge=False):
    tf_cam_to_markers, _ = est_marker_poses_and_image_locs(
        image, marker_length, intrinsics, distortion,
        xyz_cam, specified_ids, enable_enhance, enable_enlarge)
    return tf_cam_to_markers

def draw_marker_corners(frame, corners, ids):
    return aruco.drawDetectedMarkers(frame.copy(), corners, ids)

def draw_marker_poses(frame, rvecs, tvecs, camera_intrinsics_matrix, distortion_coeffs):
    frame = frame.copy()
    for i in range(rvecs.shape[0]):
        rvec = rvecs[i]
        tvec = tvecs[i]
        cv2.drawFrameAxes(frame, camera_intrinsics_matrix, distortion_coeffs, rvec, tvec, 0.1)
    return frame


def detect_markers(image, marker_length, camera_intrinsics_matrix, distortion_coeffs):
    """ Return {marker_id: tf} """
    corners, marker_ids, rvecs, tvecs = detect_marker_poses(
        image, marker_length, camera_intrinsics_matrix, distortion_coeffs)
    markers = {}
    for i, marker_id in enumerate(marker_ids):
        markers[marker_id] = munch.munchify({
            'T': np.array(tvecs[i]).flatten(),
            'R': R.from_rotvec(rvecs[i].flatten()),
        })
    return markers


def tf_camera_to_marker(image, marker_id, marker_length, camera_intrinsics_matrix, distortion_coeffs, scale=1):
    """Find the pose of a marker in an image"""

    if scale != 1:
        camera_intrinsics_matrix = scale_intrinsics(camera_intrinsics_matrix, scale)
        image = image[:, :, :3].copy()
        shape = image.shape[:2]
        image = cv2.resize(image, (shape[1] * scale, shape[0] * scale))

    corners, marker_ids, rvecs, tvecs = detect_marker_poses(
        image, marker_length, camera_intrinsics_matrix, distortion_coeffs)
    indices = np.where(marker_ids == marker_id)[0]
    if indices.size == 0:
        return None
    else:
        index = indices[0]
        return munch.munchify({
            'T': np.array(tvecs[index]).flatten(),
            'R': R.from_rotvec(rvecs[index].flatten()),
        })

def get_tf_op_from_marker(tf_front_to_marker, tf_back_to_marker):
    """ Given a front camera and back camera, and a marker lying on the ground,
        the anchor is the point that right beneath the back camera,
        on the same plane as the marker.
        Z axis points upward toward the back camera.
        Y axis is perpendicular to the Z axis and the line between the front and back camera.
        X axis is the ground projection of the line between back and front camera
    """
    tf_front_to_back = tf.combine_tfs(tf_front_to_marker, tf.inverse_tf(tf_back_to_marker))
    # keeps the position of base.
    marker_x, marker_y, marker_z = tf.R_to_axes(tf_front_to_marker.R)
    base_z = marker_z
    base_x = -nu.normalize(tf_front_to_back.T, axis=0)
    base_y = np.cross(base_z, base_x)
    base_x = np.cross(base_y, base_z)

    height_of_marker_to_front = np.dot(marker_z, tf_front_to_marker.T)
    height_of_back_to_front = np.dot(marker_z, tf_front_to_back.T)
    height_of_marker_to_back = height_of_marker_to_front - height_of_back_to_front
    origin = tf_front_to_back.T + height_of_marker_to_back * marker_z
    base_R = tf.axes_to_R('zxy', [base_z, base_x, base_y], normalize=True)
    return munch.munchify({'T': origin, 'R': base_R})

def find_image_by_markers(image_by_ts, target_marker_ids, scale,
                         marker_length, camera_intrinsics_matrix, distortion_coeffs):
    """Find one image that has all the markers."""
    intrinsics = scale_intrinsics(camera_intrinsics_matrix, scale)
    for ts, image in image_by_ts.items():
        image = image[:, :, :3].copy()
        shape = image.shape[:2]
        image = cv2.resize(image, (shape[1] * scale, shape[0] * scale))

        corners, marker_ids, rvecs, tvecs = detect_marker_poses(
            image, marker_length, intrinsics, distortion_coeffs)
        if (set(marker_ids) & set(target_marker_ids)) == set(target_marker_ids):
            return image
    return None

def scale_intrinsics(intrinsics, scale):
    """Scale intrinsics as if the image plane is moved."""
    intrinsics = intrinsics * scale
    intrinsics[2, 2] = 1.0
    return intrinsics

def get_frame_by_triplet(origin, pt_x, pt_y):
    """ Define a coordinate frame anchored at origin,
        with pt_x pointing to X axis and pt_y for roughly the y axis.
    """
    axis_x = nu.normalize(pt_x - origin, axis=0)
    axis_y = nu.normalize(pt_y - origin, axis=0)
    axis_z = np.cross(axis_x, axis_y)
    axis_y = np.cross(axis_z, axis_x)
    R = tf.axes_to_R('xyz', [axis_x, axis_y, axis_z])
    tf_src_to_origin = munch.munchify({'T': origin, 'R': R})
    return tf_src_to_origin


def calc_distances(marker_traces):
    """ Calculate the distance to the first marker position.

    marker_traces: {marker_id: {'tvecs': [], ...}}
    """
    # distances[marker_id] = {'ts':..., 'data':...} for distances of movement
    distances = collections.defaultdict(lambda: collections.defaultdict(list))
    for marker, trace in marker_traces.items():
        tvecs = trace['tvecs']
        tvecs = np.array(tvecs)
        first_t = tvecs[0]
        ref = np.tile(first_t, (tvecs.shape[0], 1))
        distance = np.linalg.norm(tvecs - ref, axis=1)
        distances[marker]['data'] = distance
        distances[marker]['ts'] = np.array(trace['ts'])
    return distances


def _iterate_object(obj):
    """
    Iterates over an object and yields (key, value) pairs.
    If the object is a dictionary, uses obj.items().
    Otherwise, assumes it is a list of (key, value) pairs.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
    else:
        for key, value in obj:
            yield key, value


def detect_marker_frames(image_by_ts, marker_length, camera_intrinsics_matrix, distortion_coeffs,
                         xyz_by_ts=None, enable_enhance=False, enable_enlarge=False):
    """ Take a lot of images to capture markers in various motion. """
    # marker_traces[marker_id] = {'ts': [...], 'tvecs': [...], 'rvecs': [...]}
    marker_traces = collections.defaultdict(lambda: collections.defaultdict(list))
    # frame_traces[ts] = {marker_id: {'rvec': ..., 'tvec': ...}}
    frame_traces = collections.OrderedDict()

    for ts, image in _iterate_object(image_by_ts):
        tf_cam_to_markers = est_marker_poses(
            image, marker_length, camera_intrinsics_matrix, distortion_coeffs,
            xyz_cam=None if xyz_by_ts is None else xyz_by_ts[ts],
            specified_ids=None, enable_enhance=enable_enhance,
            enable_enlarge=enable_enlarge)
        marker_info = {}
        # TODO: stick with pose, do not convert back to tvec and rvecs
        for marker_id, pose in tf_cam_to_markers.items():
            tvec = pose.T
            rvec = pose.R.as_rotvec()
            marker_traces[marker_id]['tvecs'].append(tvec)
            marker_traces[marker_id]['rvecs'].append(rvec)
            marker_traces[marker_id]['ts'].append(ts)
            marker_info[marker_id] = {
                'rvec': rvec,
                'tvec': tvec,
            }
        frame_traces[ts] = marker_info
    return marker_traces, frame_traces


def show_markers(front_lefts, marker_length, intrinsics, distortion, enhance=False, step_size=6):
    rows =len(front_lefts) // step_size

    plt.figure(figsize=(15, 5 * rows), dpi=80)

    # Create a Tonemapping object with a specific gamma value
    tonemap1 = cv2.createTonemap(gamma=0.5)
    #tonemap1 = cv2.createTonemapDrago(gamma=0.5)

    marker_count = collections.defaultdict(lambda: 0)

    for i in range(rows):
        db_image = front_lefts[i * step_size]
        image = db_image[:, :, :3].copy()

        if enhance:
            shape = image.shape

            image = cv2.resize(image, (shape[1] * 2, shape[0] * 2))


            # Apply tonemapping to convert the HDR image to a displayable format
            image = tonemap1.process(image.astype(np.float32))
            #image = tonemap2.process(image.astype(np.float32))


            # Scale the output to 8-bit, which is suitable for display or saving as a regular image
            image = np.clip(image*255, 0, 255).astype('uint8')

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Optional: Apply sharpening
            kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=float)
            sharpened = cv2.filter2D(gray, -1, kernel_sharpening)

            # Optional: Noise reduction
            blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)

            #blurred = cv2.medianBlur(sharpened, 3)

            image = cv2.resize(blurred, (shape[1], shape[0]))

            #sharpened = cv2.filter2D(image, -1, kernel_sharpening)
            #blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
            #image = blurred

        corners, marker_ids, rvecs, tvecs = detect_marker_poses(
            image, marker_length, intrinsics, distortion)
        content = draw_marker_corners(image, corners, marker_ids)
        plt.subplot(rows, 2, i * 2 + 1)
        plt.imshow(content)
        plt.title(f'Frame {i * step_size}, Markers: {marker_ids}')
        content = draw_marker_poses(image, rvecs, tvecs, intrinsics, distortion)
        plt.subplot(rows, 2, i * 2 + 2)
        plt.imshow(content)
        for m in marker_ids:
            marker_count[m] += 1
    plt.show()
    print(marker_count)
