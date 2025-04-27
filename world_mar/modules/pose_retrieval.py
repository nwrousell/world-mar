import torch
import math
from time import time
import json

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

def get_most_relevant_poses_to_target(target_pose, other_poses, points, min_overlap=0.3, k=3):
    """
    Returns the indices of up to k other_poses that have the most overlap with target_pose

    :param target_pose: (3,)
    :param other_poses: (P, 3)
    :param points: (N, 3)

    :return: (k,) indices (may actually be less than k if there aren't enough frames with >min_overlap)
    """
    fov_half_h = torch.tensor(105 / 2, device=pose_conditions.device)
    fov_half_v = torch.tensor(75 / 2, device=pose_conditions.device)

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

    

    in_fov_list = torch.stack([
        is_inside_fov_3d_hv(points, pc[:3], pc[-2], pc[-1], fov_half_h, fov_half_v)
        for pc in other_poses
    ])

    top_k = []
    for _ in range(k):
        overlap_ratio = ((in_fov1.bool() & in_fov_list).sum(1)) / in_fov1.sum()
        print(overlap_ratio.shape, overlap_ratio[-5:])

        # confidence = overlap_ratio + (curr_frame - frame_idx[:curr_frame]) / curr_frame * (-0.2)
        confidence = overlap_ratio + torch.arange(other_poses.shape[0], 0, -1) / other_poses.shape[0] * (-0.2) # adds recency bias

        if len(top_k) > 0:
            confidence[torch.tensor(top_k)] = -1e10
        _, r_idx = torch.topk(confidence, k=1, dim=0)
        top_k.append(r_idx[0])

        # choice 1: directly remove overlapping region
        # occupied_mask = in_fov_list[r_idx[0, range(in_fov1.shape[-1])], :, range(in_fov1.shape[-1])].permute(1,0)
        # occupied_mask = in_fov_list[r_idx[0]] & in_fov1
        in_fov1 = in_fov1 & ~occupied_mask

        # choice 2: apply similarity filter 
        # cos_sim = F.cosine_similarity(xs_pred.to(r_idx.device)[r_idx[:, range(in_fov1.shape[1])], 
        #     range(in_fov1.shape[1])], xs_pred.to(r_idx.device)[:curr_frame], dim=2)
        # cos_sim = cos_sim.mean((-2,-1))
        # mask_sim = cos_sim>0.9
        # in_fov_list = in_fov_list & ~mask_sim[:,None].to(in_fov_list.device)
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
    pose_conditions, target_pose = pose_conditions[-1001:-1], pose_conditions[-1]
    print(pose_conditions.shape, pose_conditions[:5])

    # pose_conditions = torch.ransd((1000, 5)) # slow scales linearly with this
    # target_pose = torch.tensor([0,0,0,0,0])
    points = generate_points_in_sphere(10000, 10) # slow way scales sub-linearly with this

    start = time()
    relevant_indices = get_most_relevant_poses_to_target(target_pose, pose_conditions, points)
    end = time()
    print(f"slow time: {end - start}")
    print(relevant_indices)

    # print(overlap1.shape, overlap1[:5])

    # start = time()
    # overlap2 = fast_way(target_pose, pose_conditions, points)
    # end = time()
    # print(f"fast time: {end - start}")

    # print(f"max difference: {torch.max(torch.abs(overlap1 - overlap2))}")
    # print(f"matching: {torch.allclose(overlap1, overlap2)}")


