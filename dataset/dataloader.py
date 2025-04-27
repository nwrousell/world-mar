# Snippets from https://github.com/openai/Video-Pre-Training/blob/main/data_loader.py

from typing import Tuple
import json
import glob
import os
import random
from multiprocessing import Process, Queue, Event

import numpy as np
import cv2
import lightning as L
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


# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0

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
    keys = NOOP_ACTION.keys()
    
    vector = []
    for key in keys:
        if key == 'camera':
            vector.extend(action[key])  # extend with both pitch and yaw
        else:
            vector.append(action[key])
    
    return np.array(vector, dtype=np.float32)

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
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)

class MinecraftDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        unique_ids = glob.glob(os.path.join(self.dataset_dir, "*.jsonl"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = sorted(unique_ids)

        # read counts metadata
        with open(os.path.join(dataset_dir, "counts.json"), "rt") as f:
            counts_dict = json.load(f)
        self.total_frames = counts_dict["total_frames"]
        self.demo_to_num_frames = counts_dict["demonstration_id_to_num_frames"]

        # construct demo_id -> start_frame map
        self.demo_to_start_frame = dict()
        self.demo_to_metadata = {}
        current_frame = 0
        for demo_id in self.unique_ids:
            self.demo_to_start_frame[demo_id] = current_frame
            current_frame += self.demo_to_num_frames[demo_id]

            # load all actions/poses into memory now and preprocess
            self.demo_to_metadata[demo_id] = self._preprocess_demo_metadata(demo_id)


        self.cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
        self.cursor_image = self.cursor_image[:16, :16, :] # Assume 16x16
        self.cursor_alpha = self.cursor_image[:, :, 3:] / 255.0
        self.cursor_image = self.cursor_image[:, :, :3]

    def _preprocess_demo_metadata(self, demo_id):
        with open(os.path.join(self.dataset_dir, f"{demo_id}.jsonl"), "rt") as f:
            raw_metadata = json.loads("[" + ",".join(f.readlines()) + "]")
        
        action_vectors = []
        pose_vectors = []
        is_gui_open = []
        mouse_pos = []
        last_hotbar = 0
        attack_is_stuck = False
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
            action_vectors.append(env_action_to_vector(action))

            # Update hotbar selection
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            pose_vector = np.array([step_data["xpos"], step_data["ypos"], step_data["zpos"], step_data["pitch"], step_data["yaw"]])
            pose_vectors.append(pose_vector)

            is_gui_open.append(step_data["isGuiOpen"])
            mouse_pos.append(step_data["mouse"])


        action_matrix = np.stack(action_vectors, axis=0)
        pose_matrix = np.stack(pose_vectors, axis=0)

        return { "action_matrix": action_vectors, "pose_matrix": pose_matrix, "is_gui_open": is_gui_open, "mouse_pos": mouse_pos }
    
    def _idx_to_demo_and_frame(self, idx) -> Tuple[str, int]:
        for demo_id, start_frame in self.demo_to_start_frame.items():
            n_frames = self.demo_to_num_frames[demo_id] - 1 # can sample n-1 frames
            if idx >= start_frame and idx < start_frame + n_frames:
                return demo_id, idx - start_frame + 1 # we can't sample first frames so add 1
        
        raise Exception("out of bounds")

    def __len__(self) -> int:
        return self.total_frames - len(self.demo_to_metadata.keys()) # we can't sample first frames cause we won't have context

    def __getitem__(self, idx):
        demo_id, frame_idx = self._idx_to_demo_and_frame(idx)

        assert frame_idx > 0

        # get target frame and trajectory metadata
        frame_path = os.path.join(self.dataset_dir, demo_id, f"frame{frame_idx:06d}.jpg")
        frame = cv2.imread(frame_path)

        action_matrix, pose_matrix, is_gui_open, mouse_pos = (
            self.demo_to_metadata[demo_id]["action_matrix"], 
            self.demo_to_metadata[demo_id]["pose_matrix"],
            self.demo_to_metadata[demo_id]["is_gui_open"],
            self.demo_to_metadata[demo_id]["mouse_pos"],
        )

        action, target_frame_pose = action_matrix[frame_idx-1], pose_matrix[frame_idx]

        # convert poses to relative
        relative_pose_matrix = pose_matrix.copy()[:frame_idx] - pose_matrix[frame_idx]
        relative_pose_matrix[:, [3,4]][relative_pose_matrix[:, [3,4]] > np.pi] -= 2 * np.pi
        relative_pose_matrix[:, [3,4]][relative_pose_matrix[:, [3,4]] < -np.pi] += 2 * np.pi

        # compose with cursor
        if is_gui_open[frame_idx]:
            camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
            cursor_x = int(mouse_pos[frame_idx]["x"] * camera_scaling_factor)
            cursor_y = int(mouse_pos[frame_idx]["y"] * camera_scaling_factor)
            composite_images_with_alpha(frame, self.cursor_image, self.cursor_alpha, cursor_x, cursor_y)

        # sample K context frames using monte-carlo overlap
        context_indices = [1, 2, 3]
        context_frames = []
        for context_i in context_indices:
            context_frame_path = os.path.join(self.dataset_dir, demo_id, f"frame{frame_idx:06d}.jpg")
            context_frames.append(cv2.imread(context_frame_path))
        context_relative_poses = relative_pose_matrix[context_indices]

        context_frames = np.stack(context_frames, axis=0)

        return frame, context_frames, context_relative_poses, action

class MinecraftDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir: str, index_path: str):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.index_path = index_path
        self.dataset = MinecraftDataset(dataset_dir=dataset_dir)
    
    def train_dataloader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    dataset = MinecraftDataset(dataset_dir="../data")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(len(dataloader))
    batch = next(iter(dataloader))
    for item in batch:
        print(item.shape)