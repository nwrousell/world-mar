import torch
import math
from time import time
import json
import numpy as np
from packaging import version as pver
from einops import rearrange

def euler_to_rotation_matrix(pitch, yaw):
    """
    Convert pitch and yaw angles (in radians) to a 3x3 rotation matrix.
    Supports batch input.

    Args:
        pitch (torch.Tensor): Pitch angles in radians.
        yaw (torch.Tensor): Yaw angles in radians.

    Returns:
        torch.Tensor: Rotation matrix of shape (batch_size, 3, 3).
    """
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    R_pitch = torch.stack([
        torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
        torch.zeros_like(pitch), cos_pitch, -sin_pitch,
        torch.zeros_like(pitch), sin_pitch, cos_pitch
    ], dim=-1).reshape(-1, 3, 3)

    R_yaw = torch.stack([
        cos_yaw, torch.zeros_like(yaw), sin_yaw,
        torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
        -sin_yaw, torch.zeros_like(yaw), cos_yaw
    ], dim=-1).reshape(-1, 3, 3)

    return torch.matmul(R_yaw, R_pitch)

def euler_to_camera_to_world_matrix(pose):
    """
    Convert (x, y, z, pitch, yaw) to a 4x4 camera-to-world transformation matrix using torch.
    Supports both (5,) and (f, b, 5) shaped inputs.

    Args:
        pose (torch.Tensor): Pose tensor of shape (5,) or (f, b, 5).

    Returns:
        torch.Tensor: Camera-to-world transformation matrix of shape (4, 4).
    """

    origin_dim = pose.ndim
    if origin_dim == 1:
        pose = pose.unsqueeze(0).unsqueeze(0)  # Convert (5,) -> (1, 1, 5)
    elif origin_dim == 2:
        pose = pose.unsqueeze(0)

    x, y, z, pitch, yaw = pose[..., 0], pose[..., 1], pose[..., 2], pose[..., 3], pose[..., 4]
    pitch, yaw = torch.deg2rad(pitch), torch.deg2rad(yaw)

    # Compute rotation matrix (batch mode)
    R = euler_to_rotation_matrix(pitch, yaw)  # Shape (f*b, 3, 3)

    # Create the 4x4 transformation matrix
    eye = torch.eye(4, dtype=torch.float32, device=pose.device)
    camera_to_world = eye.repeat(R.shape[0], 1, 1)  # Shape (f*b, 4, 4)

    # Assign rotation
    camera_to_world[:, :3, :3] = R

    # Assign translation
    camera_to_world[:, :3, 3] = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)

    # Reshape back to (f, b, 4, 4) if needed
    if origin_dim == 3:
        return camera_to_world.view(pose.shape[0], pose.shape[1], 4, 4)
    elif origin_dim == 2:
        return camera_to_world.unsqueeze(0)
    else:
        return camera_to_world.squeeze(0).squeeze(0)  # Convert (1,1,4,4) -> (4,4)

def camera_to_world_to_world_to_camera(camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    Convert Camera-to-World matrices to World-to-Camera matrices for a tensor with shape (f, b, 4, 4).

    Args:
        camera_to_world (torch.Tensor): A tensor of shape (f, b, 4, 4), where:
            f = number of frames,
            b = batch size.

    Returns:
        torch.Tensor: A tensor of shape (f, b, 4, 4) representing the World-to-Camera matrices.
    """
    # Ensure input is a 4D tensor
    assert camera_to_world.ndim == 4 and camera_to_world.shape[2:] == (4, 4), \
        "Input must be of shape (f, b, 4, 4)"
    
    # Extract the rotation (R) and translation (T) parts
    R = camera_to_world[:, :, :3, :3]  # Shape: (f, b, 3, 3)
    T = camera_to_world[:, :, :3, 3]   # Shape: (f, b, 3)
    
    # Initialize an identity matrix for the output
    world_to_camera = torch.eye(4, device=camera_to_world.device).unsqueeze(0).unsqueeze(0)
    world_to_camera = world_to_camera.repeat(camera_to_world.size(0), camera_to_world.size(1), 1, 1)  # Shape: (f, b, 4, 4)
    
    # Compute the rotation (transpose of R)
    world_to_camera[:, :, :3, :3] = R.transpose(2, 3)
    
    # Compute the translation (-R^T * T)
    world_to_camera[:, :, :3, 3] = -torch.matmul(R.transpose(2, 3), T.unsqueeze(-1)).squeeze(-1)
    
    return world_to_camera.to(camera_to_world.dtype)

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_relative_pose(abs_c2ws, zero_first_frame_scale):
    abs_w2cs = camera_to_world_to_world_to_camera(abs_c2ws)
    target_cam_c2w = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(abs_c2ws.device).to(abs_c2ws.dtype)
    abs2rel = target_cam_c2w @ abs_w2cs[zero_first_frame_scale]
    ret_poses = [abs2rel @ abs_c2w for abs_c2w in abs_c2ws]
    ret_poses = torch.stack(ret_poses)
    return ret_poses

def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i, device=device, dtype=c2w.dtype)  # [B, HxW]
    xs = -(i - cx) / fx * zs
    ys = -(j - cy) / fy * zs 

    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6

    return plucker

def convert_to_plucker(poses, curr_frame, focal_length=0.35, image_width=640, image_height=320):

    intrinsic = np.asarray([focal_length * image_width,
                                focal_length * image_height,
                                0.5 * image_width,
                                0.5 * image_height], dtype=np.float32)

    c2ws = get_relative_pose(poses, zero_first_frame_scale=curr_frame)
    c2ws = rearrange(c2ws, "t b m n -> b t m n")

    K = torch.as_tensor(intrinsic, device=poses.device, dtype=poses.dtype).repeat(c2ws.shape[0],c2ws.shape[1],1)  # [B, F, 4]
    plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device=c2ws.device)
    plucker_embedding = rearrange(plucker_embedding, "b t h w d -> t b h w d").contiguous()

    return plucker_embedding

def generate_points_in_sphere(n_points, radius):
    # Sample three independent uniform distributions
    samples_r = torch.rand(n_points)       # For radius distribution
    samples_phi = torch.rand(n_points)     # For azimuthal angle phi
    samples_u = torch.rand(n_points)       # For polar angle theta

    # Apply cube root to ensure uniform volumetric distribution
    r = radius * torch.pow(samples_r, 1/3)
    # Azimuthal angle phi uniformly distributed in [0, 2π]
    phi = 2 * math.pi * samples_phi
    # Convert u to theta to ensure cos(theta) is uniformly distributed
    theta = torch.acos(1 - 2 * samples_u)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points

def is_inside_fov_3d_hv(points, center, center_pitch, center_yaw, fov_half_h, fov_half_v):
    """
    Check whether points are within a given 3D field of view (FOV) 
    with separately defined horizontal and vertical ranges.

    The center view direction is specified by pitch and yaw (in degrees).

    :param points: (N, B, 3) Sample point coordinates
    :param center: (3,) Center coordinates of the FOV
    :param center_pitch: Pitch angle of the center view (in degrees)
    :param center_yaw: Yaw angle of the center view (in degrees)
    :param fov_half_h: Horizontal half-FOV angle (in degrees)
    :param fov_half_v: Vertical half-FOV angle (in degrees)
    :return: Boolean tensor (N, B), indicating whether each point is inside the FOV
    """
    # Compute vectors relative to the center
    vectors = points - center  # shape (N, B, 3)
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    
    # Compute horizontal angle (yaw): measured with respect to the z-axis as the forward direction,
    # and the x-axis as left-right, resulting in a range of -180 to 180 degrees.
    azimuth = torch.atan2(x, z) * (180 / math.pi)
    
    # Compute vertical angle (pitch): measured with respect to the horizontal plane,
    # resulting in a range of -90 to 90 degrees.
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi)
    
    # Compute the angular difference from the center view (handling circular angle wrap-around)
    diff_azimuth = (azimuth - center_yaw).abs() % 360
    diff_elevation = (elevation - center_pitch).abs() % 360
    
    # Adjust values greater than 180 degrees to the shorter angular difference
    diff_azimuth = torch.where(diff_azimuth > 180, 360 - diff_azimuth, diff_azimuth)
    diff_elevation = torch.where(diff_elevation > 180, 360 - diff_elevation, diff_elevation)
    
    # Check if both horizontal and vertical angles are within their respective FOV limits
    return (diff_azimuth < fov_half_h) & (diff_elevation < fov_half_v)

def is_inside_fovs_3d(points, centers, center_pitches, center_yaws, fov_half_h, fov_half_v):
    """
    Check whether points are within given 3D field of views (FOVs) 
    with separately defined horizontal and vertical ranges.

    The center view direction is specified by pitch and yaw (in degrees).

    :param points: (N, 3) Sample point coordinates
    :param center: (P, 3) Center coordinates of the FOV
    :param center_pitch: (P,) Pitch angle of the center view (in degrees)
    :param center_yaw: (P,) Yaw angle of the center view (in degrees)
    :param fov_half_h: Horizontal half-FOV angle (in degrees)
    :param fov_half_v: Vertical half-FOV angle (in degrees)
    :return: Boolean tensor (P, N) indicating whether each point is inside each FOV
    """
    # Compute vectors relative to the center
    start = time()
    points = points[None, :, :].repeat(centers.shape[0], 1, 1) # (P, N, #)
    vectors = points - centers[:, None, :]  # shape (P, N, 3)
    x = vectors[..., 0] # (P, N)
    y = vectors[..., 1] # (P, N)
    z = vectors[..., 2] # (P, N)
    end = time()
    print(f"first: {end - start}")
    
    # Compute horizontal angle (yaw): measured with respect to the z-axis as the forward direction,
    # and the x-axis as left-right, resulting in a range of -180 to 180 degrees.
    start = time()
    azimuth = torch.atan2(x, z) * (180 / math.pi) # (P, N)
    
    # Compute vertical angle (pitch): measured with respect to the horizontal plane,
    # resulting in a range of -90 to 90 degrees.
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi) # (P, N)
    end = time()
    print(f"second: {end - start}")
    
    # Compute the angular difference from the center view (handling circular angle wrap-around)
    start = time()
    diff_azimuth = (azimuth - center_yaws[:, None]).abs() % 360
    diff_elevation = (elevation - center_pitches[:, None]).abs() % 360
    end = time()
    print(f"third: {end - start}")

    start = time()
    # Adjust values greater than 180 degrees to the shorter angular difference
    diff_azimuth = torch.where(diff_azimuth > 180, 360 - diff_azimuth, diff_azimuth)
    diff_elevation = torch.where(diff_elevation > 180, 360 - diff_elevation, diff_elevation)
    end = time()
    print(f"fourth: {end - start}")
    
    # Check if both horizontal and vertical angles are within their respective FOV limits
    return (diff_azimuth < fov_half_h) & (diff_elevation < fov_half_v)

def get_most_relevant_poses_to_target(target_pose, other_poses, points, min_overlap=0.3, k=3, do_optim=True):
    """
    Returns the indices of up to k other_poses that have the most overlap with target_pose

    :param target_pose: (3,)
    :param other_poses: (P, 3)
    :param points: (N, 3)

    :return: (k,) indices (may actually be less than k if there aren't enough frames with >min_overlap)
    """
    fov_half_h = torch.tensor(105 / 2, device=target_pose.device)
    fov_half_v = torch.tensor(75 / 2, device=target_pose.device)

    target_pose = target_pose.clone()
    other_poses = other_poses.clone()
    points = points.clone()

    other_poses[:, :3] -= target_pose[:3]
    target_pose[:3] = torch.tensor([0,0,0])

    in_fov1 = is_inside_fov_3d_hv(
        points, 
        target_pose[:3], 
        target_pose[-2], 
        target_pose[-1], 
        fov_half_h, 
        fov_half_v
    )

    if do_optim:
        points = points[in_fov1]
        in_fov1 = in_fov1[in_fov1]

    in_fov_list = torch.stack([
        is_inside_fov_3d_hv(points, pc[:3], pc[-2], pc[-1], fov_half_h, fov_half_v)
        for pc in other_poses
    ])

    most_recent_frame_idx = other_poses.shape[0]-1
    top_k = [most_recent_frame_idx] # force most recent frame to be in context
    in_fov1 = in_fov1 & ~in_fov_list[other_poses.shape[0]-1]
    for _ in range(k-1):
        if in_fov1.sum() < 5:
            break

        overlap_ratio = ((in_fov1.bool() & in_fov_list).sum(1)) / in_fov1.sum()

        # add recency bias
        confidence = overlap_ratio + torch.arange(other_poses.shape[0], 0, -1) / other_poses.shape[0] * (-0.2)

        if len(top_k) > 0:
            confidence[torch.tensor(top_k)] = -1e10
        _, r_idx = torch.topk(confidence, k=1, dim=0)
        top_k.append(r_idx[0])

        # directly remove overlapping region
        in_fov1 = in_fov1 & ~in_fov_list[r_idx[0]]

    return torch.tensor(top_k)

def fast_way(target_pose, other_poses, points):
    fov_half_h = torch.tensor(105 / 2, device=pose_conditions.device)
    fov_half_v = torch.tensor(75 / 2, device=pose_conditions.device)

    all_poses = torch.cat([target_pose[None, :], other_poses])

    in_fov = is_inside_fovs_3d(
        points, 
        all_poses[:, :3], 
        all_poses[:, -2], 
        all_poses[:, -1], 
        fov_half_h, 
        fov_half_v
    )

    in_fov1, in_fov_list = in_fov[0], in_fov[1:]

    overlap_ratio = ((in_fov1.bool() & in_fov_list).sum(1)) / in_fov1.sum()
    return overlap_ratio

class Memory:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = []
    
    def insert(self, pose, frame, idx):
        self.cache.append((pose, frame, idx))
        if len(self.cache) > self.size:
            pass


if __name__ == "__main__":
    # pose_conditions = torch.tensor([
    #     [0, 0, 0, 30, 30],
    #     [5,-2, 0, 0, 0],
    # ])

    metadata_path = "../../data/cheeky-cornflower-setter-db485e7cdf63-20220416-103055.jsonl"

    with open(metadata_path, "rt") as f:
        metadata = json.loads("[" + ",".join(f.readlines()) + "]")

    pose_conditions = list(map(lambda s: torch.tensor([s["xpos"], s["ypos"], s["zpos"], s["pitch"], s["yaw"]]), metadata))
    pose_conditions = torch.stack(pose_conditions)

    pose_conditions, target_pose = pose_conditions[-1001:-4], pose_conditions[-5]

    print(pose_conditions.shape) # (997, 5)
    absolute_camera_to_world_matrices = euler_to_camera_to_world_matrix(pose_conditions[-5:])
    print(absolute_camera_to_world_matrices.shape)
    plucker = convert_to_plucker(poses=absolute_camera_to_world_matrices, curr_frame=-1)

    print(plucker.shape)

    # relative_poses = get_relative_pose(absolute_camera_to_world_matrices, -1)
    # print(relative_poses.shape)

    # print("relative", relative_poses[-1], "\n", relative_poses[-2], "\n", relative_poses[-3])

    # pose_conditions = torch.ransd((1000, 5)) # slow scales linearly with this
    # target_pose = torch.tensor([0,0,0,0,0])
    # points = generate_points_in_sphere(10000, 10) # slow way scales sub-linearly with this

    # start = time()
    # relevant_indices = get_most_relevant_poses_to_target(target_pose, pose_conditions, points)
    # end = time()
    # print(f"slow time: {end - start}")
    # print(relevant_indices)

    # start = time()
    # relevant_indices = get_most_relevant_poses_to_target(target_pose, pose_conditions, points, do_optim=True)
    # end = time()
    # print(f"fast time: {end - start}")
    # print(relevant_indices)

    # print(overlap1.shape, overlap1[:5])

    # start = time()
    # overlap2 = fast_way(target_pose, pose_conditions, points)
    # end = time()
    # print(f"fast time: {end - start}")

    # print(f"max difference: {torch.max(torch.abs(overlap1 - overlap2))}")
    # print(f"matching: {torch.allclose(overlap1, overlap2)}")


