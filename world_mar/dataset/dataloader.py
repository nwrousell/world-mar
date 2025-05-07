# Snippets from https://github.com/openai/Video-Pre-Training/blob/main/data_loader.py

from typing import Tuple
import json
import glob
import os
from time import time
import random
from multiprocessing import Process, Queue, Event

from safetensors.torch import load_file, save_file

from ..modules.pose_retrieval import get_most_relevant_poses_to_target, get_relative_pose, euler_to_camera_to_world_matrix, convert_to_plucker, generate_points_in_sphere

import numpy as np
import cv2
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
QUEUE_TIMEOUT = 10

CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

# AGENT_RESOLUTION = (128, 128)
AGENT_RESOLUTION = (224, 224)
MINEREC_ORIGINAL_HEIGHT_PX = 720

# If GUI is open, mouse dx/dy need also be adjusted with these scalers.
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "camera",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
# CAMERA_SCALER = 360.0 / 2400.0
CAMERA_SCALER = 360.0 / (2400.0 * 6) # 3 added by Noah to normalize mouse_dx * CAMERA_SCALER between -1 and 1


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img

def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action

def env_action_to_vector(action):    
    vector = []
    for key in ACTION_KEYS:
        if key == 'camera':
            vector.extend(action[key] * CAMERA_SCALER)  # extend with both pitch and yaw
        else:
            vector.append(action[key])
    
    return torch.tensor(vector, dtype=torch.float32)

def one_hot_actions(actions):
    actions_one_hot = torch.zeros(len(actions), len(ACTION_KEYS))
    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                value = (value - num_buckets) / num_buckets
                assert -1 - 1e-3 <= value <= 1 + 1e-3, f"Camera action value must be in [-1, 1], got {value}"
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            actions_one_hot[i, j] = value

    return actions_one_hot

def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha)

class MinecraftDataset(Dataset):
    def __init__(
        self, dataset_dir,
        memory_size=100,
        memory_distance=1000, prev_distance=5,
        num_mem_frames=2, num_prev_frames=2
    ):
        self.dataset_dir = dataset_dir
        # Window lengths defining how many frames are reserved for previous frame retrieval,
        # and how many are reserved for memory frame retrieval
        #
        # ... | mem_distance | prev_distance | <-- index 0
        #
        # Sample num_mem_frames many frames from mem_distance window
        # Sample num_prev_frames many frames from prev_distance window
        # 
        # As an optimization, only consider memory_size many frames
        # for WorldMem style point-based overlap scores
        self.memory_distance = memory_distance
        self.prev_distance = prev_distance
        self.memory_size = memory_size
        self.num_prev_frames = num_prev_frames
        self.num_mem_frames = num_mem_frames
        self.num_context_frames = num_mem_frames + num_prev_frames

        # read video frame counts metadata from json file
        with open(os.path.join(dataset_dir, "counts.json"), "rt") as f:
            counts_dict = json.load(f)
        self.total_frames = counts_dict["total_frames"]
        self.demo_to_num_frames = counts_dict["demonstration_id_to_num_frames"]
        self.unique_ids = sorted(list(self.demo_to_num_frames.keys()))

        # create dictionaries for computing global and local frame indices
        self.demo_to_start_frame = {}
        self.demo_to_metadata = {}
        current_frame = 0
        start = time()
        for demo_id in self.unique_ids:
            self.demo_to_start_frame[demo_id] = current_frame
            current_frame += self.demo_to_num_frames[demo_id] - 1

            # load all actions/poses into memory now and preprocess
            self.demo_to_metadata[demo_id] = self._preprocess_demo_metadata(demo_id)

        print(f"FINISHED PREPROCESSING ALL METADATA: {time() - start}")

        # prepare cursor image
        self.cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
        self.cursor_image = self.cursor_image[:16, :16, :] # Assume 16x16
        self.cursor_alpha = self.cursor_image[:, :, 3:] / 255.0
        self.cursor_image = self.cursor_image[:, :, :3]

        # generate points for monte-carlo memory retrieval
        self.points = generate_points_in_sphere(n_points=10_000, radius=30)
        self.mem_idx_shifts = self._compute_mem_idx_shifts()

    def _compute_prev_idx_shifts(self):
        valid_prev_shifts = torch.arange(-self.prev_distance, 0, dtype=torch.float32)  # (self.prev_distance)

        # compute positive distances from current frame at idx
        distances = -valid_prev_shifts

        # linearly decaying weights
        weights = ((self.prev_distance + 1) - distances)
        weights = weights / weights.sum()

        # sample without replacement according to weights
        picks = torch.multinomial(weights, self.num_prev_frames, replacement=False)  # (self.num_prev_frames)
        sampled_shifts = valid_prev_shifts[picks].to(torch.int64)                    # e.g. [-3, -1] (no ordering though)
        return sampled_shifts

    def _compute_mem_idx_shifts(self):
        # define candidate index shifts up to memory_distance indices behind previous frame window
        mem_min_shift = -self.prev_distance
        mem_max_shift = mem_min_shift - self.memory_distance
        valid_mem_shifts = torch.arange(mem_max_shift, mem_min_shift, dtype=torch.float32)  # (self.memory_size)

        # compute exponential weights that sum up to 1
        lamb = 0.005  # <-- dropoff-factor in e^{lamb * l} exponential
        w = torch.exp(lamb * valid_mem_shifts)
        w = w / w.sum()

        # invert CDF sampling for self.memory_size points
        cdf = torch.cumsum(w, dim=0)
        u = torch.linspace(1/(2*self.memory_size), 1 - 1/(2*self.memory_size), self.memory_size)
        picks = torch.searchsorted(cdf, u)

        # gather sampled indices and cast to ints
        sampled_shifts = valid_mem_shifts[picks].to(torch.int64)
        return sampled_shifts

    def _preprocess_demo_metadata(self, demo_id):
        with open(os.path.join(self.dataset_dir, f"{demo_id}.jsonl"), "rt") as f:
            raw_metadata = json.loads("[" + ",".join(f.readlines()) + "]")
        
        action_vectors = []
        pose_vectors = []
        is_gui_open = []
        mouse_pos = []
        last_hotbar = 0
        attack_is_stuck = False
        max_dx = 0
        for i, step_data in enumerate(raw_metadata):
            if i == 0:
                # Check if attack will be stuck down
                if step_data["mouse"]["newButtons"] == [0]:
                    attack_is_stuck = True
            elif attack_is_stuck:
                # Check if we press attack down, then it might not be stuck
                if 0 in step_data["mouse"]["newButtons"]:
                    attack_is_stuck = False
            # If still stuck, remove the action
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

            action, is_null_action = json_action_to_env_action(step_data)

            # Update hotbar selection
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar
            if abs(step_data["mouse"]["dx"]) > max_dx:
                max_dx = abs(step_data["mouse"]["dx"])

            action_vectors.append(env_action_to_vector(action))

            pose_vector = torch.tensor([step_data["xpos"], step_data["ypos"], step_data["zpos"], step_data["pitch"], step_data["yaw"]])
            pose_vectors.append(pose_vector)

            is_gui_open.append(step_data["isGuiOpen"])
            mouse_pos.append(step_data["mouse"])

        action_matrix = torch.stack(action_vectors)
        pose_matrix = torch.stack(pose_vectors, axis=0)
        is_gui_open = torch.tensor(is_gui_open)

        return { "action_matrix": action_matrix, "pose_matrix": pose_matrix, "is_gui_open": is_gui_open, "mouse_pos": mouse_pos }
    
    def _idx_to_demo_and_frame(self, idx) -> Tuple[str, int]:
        for demo_id, start_frame in self.demo_to_start_frame.items():
            n_usable_frames = self.demo_to_num_frames[demo_id] - 1 # can sample n-1 frames
            if idx >= start_frame and idx < start_frame + n_usable_frames:
                return demo_id, idx - start_frame + 1

        raise Exception(f"out of bounds - idx: {idx}, len: {self.__len__()}, start_frame: {start_frame}, n_usable_frames: {n_usable_frames}")

    def __len__(self) -> int:
        return sum([self.demo_to_num_frames[demo_id] for demo_id in self.unique_ids]) - len(self.unique_ids)

    def __getitem__(self, idx):
        # ----- extract relevant data -----
        demo_id, frame_idx = self._idx_to_demo_and_frame(idx)
        assert frame_idx > 0 and frame_idx < self.demo_to_num_frames[demo_id]

        action_matrix, pose_matrix, is_gui_open, mouse_pos = (
            self.demo_to_metadata[demo_id]["action_matrix"], 
            self.demo_to_metadata[demo_id]["pose_matrix"],
            self.demo_to_metadata[demo_id]["is_gui_open"],
            self.demo_to_metadata[demo_id]["mouse_pos"],
        )

        # ----- sample K context frames -----
        # self.num_prev_frames randomly selected from previous frame window --> forced to be selected in memory process
        # self.num_mem_frames selected using WorldMem monte-carlo overlap
        # we have to pass the previous frames through the memory process due to similarity filtering; we penalize memory
        # frames if they are too close to the FOV of the forced previous frames

        # calculate indices of context frames using backwards shifts from frame_idx
        mem_frame_idx_shifts = self.mem_idx_shifts               # pre-computed and cached
        prev_frame_idx_shifts = self._compute_prev_idx_shifts()  # randomly chosen each sample
        candidate_ctx_indices = frame_idx + torch.cat([prev_frame_idx_shifts, mem_frame_idx_shifts])

        # clean up context frame indices since samples from early in a video may not have many previous frames
        candidate_ctx_indices = candidate_ctx_indices[candidate_ctx_indices >= 0]           # filter out frame indices that are negative
        candidate_ctx_indices = candidate_ctx_indices[~is_gui_open[candidate_ctx_indices]]  # filter out frames where the GUI is open

        if len(candidate_ctx_indices) == 0:
            candidate_ctx_indices = torch.tensor([frame_idx - 1], dtype=torch.int64)

        # retrieve final context frame indices using forced prev frames and overlap based memory frames
        retrieved_indices = get_most_relevant_poses_to_target(
            target_pose=pose_matrix[frame_idx],
            other_poses=pose_matrix[candidate_ctx_indices], 
            points=self.points, 
            min_overlap=0.1, 
            k=self.num_context_frames,
            num_prev_frames=min(self.num_prev_frames, len(candidate_ctx_indices))
        )
        context_indices = candidate_ctx_indices[retrieved_indices[:len(candidate_ctx_indices)]].sort(descending=True)[0]

        # concatenate the target frame and context frame indices
        frame_indices = torch.cat([torch.tensor([frame_idx]), context_indices])

        # convert poses to plucker embeddings
        absolute_camera_to_world_matrices = euler_to_camera_to_world_matrix(pose_matrix[frame_indices])
        plucker = convert_to_plucker(poses=absolute_camera_to_world_matrices, curr_frame=0).squeeze()

        # read frames from disk
        frames = []
        for frame_i in frame_indices:
            latent_path = os.path.join(self.dataset_dir, demo_id, f"frame{frame_i:06d}.bin")
            latent = torch.tensor(np.fromfile(latent_path, np.float32).reshape(576, 16))
            frames.append(latent)

        # add padding frames if necessary
        num_non_padding_frames = len(frames)

        if len(frames) < self.num_context_frames + 1:
            num_padding = self.num_context_frames + 1 - len(frames)
            plucker = torch.cat([plucker, torch.zeros((num_padding, *plucker[0].shape))], dim=0)

            for _ in range(num_padding):
                frames.append(torch.zeros_like(frames[-1]))

            frame_indices = torch.cat([frame_indices, torch.full((num_padding,), -1)])

        frames = torch.stack(frames, axis=0)
        assert len(plucker) == self.num_context_frames + 1
        assert len(frames) == self.num_context_frames + 1

        # add padding actions if necessary (the final frame does not have an action, hence no +1)
        actions = action_matrix[context_indices]
        if len(actions) < self.num_context_frames:
            num_padding = self.num_context_frames - len(actions)
            actions = torch.cat([actions, torch.zeros((num_padding, *actions[0].shape))], dim=0)
        assert len(actions) == self.num_context_frames

        timestamps = frame_indices - frame_idx
        assert len(timestamps) == self.num_context_frames + 1

        return {
            "frames": frames,
            "plucker": plucker,
            "action": actions,
            "timestamps": timestamps,
            "num_non_padding_frames": num_non_padding_frames
        }

class MinecraftDataModule(L.LightningDataModule):
    def __init__(
        self, dataset_dir: str, batch_sz=64,
        memory_size=100,
        memory_distance=1000, prev_distance=5,
        num_mem_frames=2, num_prev_frames=2,
        train_split=0.9, num_workers=8
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_sz = batch_sz
        self.num_workers = num_workers
        dataset = MinecraftDataset(
            dataset_dir=dataset_dir,
            memory_size=memory_size,
            memory_distance=memory_distance, prev_distance=prev_distance,
            num_mem_frames=num_mem_frames, num_prev_frames=num_prev_frames
        )
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_split, 1-train_split])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_sz, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_sz, num_workers=self.num_workers)


if __name__ == "__main__":
    # dm = MinecraftDataModule(dataset_dir="/users/nrousell/scratch/minecraft-raw")
    dataset = MinecraftDataset(dataset_dir="/users/nrousell/scratch/minecraft-raw", num_context_frames=4)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    batch = dataset.__getitem__(50)

    # loader = DataLoader(dataset, batch_size=64, num_workers=0)

    # for num_workers in [12, 0, 2, 4, 8, 16]:
    #    loader = DataLoader(dataset, batch_size=64, num_workers=num_workers)
    #    start = time()
    #    for i, batch in enumerate(loader):
    #        if i == 10:
    #            break
    #    print(f"Workers: {num_workers}, Time: {time() - start:.2f}s")