# pasted from https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline/blob/main/utils/download_dataset.py

# A script to download OpenAI contractor data or BASALT datasets
# using the shared .json files (index file).
#
# Json files are in format:
# {"basedir": <prefix>, "relpaths": [<relpath>, ...]}
#
# The script will download all files in the relpaths list,
# or maximum of set number of demonstrations,
# to a specified directory.


import argparse
import urllib.request
import os
import glob, os, cv2, json
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from tqdm import tqdm
from world_mar.modules.utils import instantiate_from_config
import torch
import shutil
from time import time
import numpy as np

from .dataloader import MINEREC_ORIGINAL_HEIGHT_PX, composite_images_with_alpha, CURSOR_FILE

MAX_THREADS = 10

# frame_counter_lock = Lock()
# num_total_frames = 0
# demonstration_id_to_num_frames = {}


parser = argparse.ArgumentParser(description="Download OpenAI contractor datasets")
# parser.add_argument("--json-file", type=str, required=True, help="Path to the index .json file")
# parser.add_argument("--num-demos", type=int, default=None, help="Maximum number of demonstrations to download")
parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")

CONFIG_PATH = "configs/world_mar.yaml"

def relpaths_to_download(relpaths, output_dir):
    def read_json(file_name):
        with open(file_name, 'r') as json_file:
            text = json.loads('['+','.join(json_file.readlines())+']')

    data_path = '/'.join(relpaths[0].split('/')[:-1])
    non_defect=[]
    for vid_name in glob.glob(os.path.join(output_dir,'*.mp4')):
        try:
            vid = cv2.VideoCapture(vid_name)
            read_json(vid_name.replace('mp4', 'jsonl'))
            if vid.isOpened():
                non_defect.append(os.path.join(data_path, vid_name.split('/')[-1]))
        except Exception as e:
            # print(f"{vid_name}: {e}")
            continue

    relpaths = set(relpaths)
    non_defect = set(non_defect)
    diff_to_download = relpaths.difference(non_defect)
    print('total:', len(relpaths), '| exist:', len(non_defect), '| downloading:', len(diff_to_download))
    return diff_to_download

def unroll_mp4_into_jpgs(mp4_path: str, output_folder: str, jpg_quality=95) -> int:
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {mp4_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_filename = os.path.join(output_folder, f"frame{frame_idx:06d}.jpg")
        # Save frame as JPEG
        cv2.imwrite(frame_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

        frame_idx += 1

    cap.release()
    return frame_idx

def unroll_mp4_into_latents(mp4_path: str, output_folder: str, vae, gpu_id, cursor_image, cursor_alpha) -> int:
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {mp4_path}")

    # open jsonl file
    with open(mp4_path.replace("mp4", "jsonl"), 'r') as json_file:
        steps = json.loads('['+','.join(json_file.readlines())+']')

    frame_idx = 0
    BATCH_SIZE = 128
    frames = []
    for i, step_data in enumerate(steps):
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if step_data["isGuiOpen"]:
            camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
            cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
            cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
            composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = (frame_rgb.astype(np.float32) / 255.0) * 2 - 1 # bring to [-1, 1]
        frames.append(torch.tensor(frame_rgb).permute(2,0,1)) # permute to [c, h, w]

        if len(frames) == BATCH_SIZE:
            batch = torch.stack(frames).to(f"cuda:{gpu_id}")
            start = time()
            with torch.no_grad():
                latents = vae.encode(batch).sample()
            end = time()
            for latent in latents:
                path = os.path.join(output_folder, f"frame{frame_idx:06d}.bin")
                latent.cpu().numpy().tofile(path) # (576, 16), np.float32
                # torch.save(latent, path)
                frame_idx += 1
            print(f"wrote {BATCH_SIZE} latents to {output_folder}")
            frames = []


    batch = torch.stack(frames).to(f"cuda:{gpu_id}")
    latents = vae.encode(batch).sample()
    for latent in latents:
        path = os.path.join(output_folder, f"frame{frame_idx:06d}.bin")
        latent.cpu().numpy().tofile(path)
        # torch.save(latent, path)
        frame_idx += 1

    cap.release()

    os.remove(mp4_path)

    return frame_idx

def download_video_and_action_files(basedir: str, relpath: str, pbar):
    global num_total_frames
    url = basedir + relpath
    filename = os.path.basename(relpath)
    outpath = os.path.join(args.output_dir, filename)

    print(f"Downloading {outpath}...")
    try:
        urllib.request.urlretrieve(url, outpath)
    except Exception as e:
        print(f"\tError downloading {url}: {e}. Moving on")
        return

    # Also download corresponding .jsonl file
    jsonl_url = url.replace(".mp4", ".jsonl")
    jsonl_filename = filename.replace(".mp4", ".jsonl")
    jsonl_outpath = os.path.join(args.output_dir, jsonl_filename)
    try:
        urllib.request.urlretrieve(jsonl_url, jsonl_outpath)
    except Exception as e:
        print(f"\tError downloading {jsonl_url}: {e}. Cleaning up mp4 and moving on")
        os.remove(outpath)
        return
    
    try:
        with open(jsonl_outpath, "rt") as f:
            _ = json.loads("[" + ",".join(f.readlines()) + "]")
    except Exception as e:
        print(f"\tError decoding {jsonl_outpath}: {e}. Cleaning up mp4/jsonl and moving on")
        os.remove(outpath)
        os.remove(jsonl_outpath)
        return

    # unroll into folder of jpgs
    # folder_path = outpath.removesuffix(".mp4")
    # demo_id = os.path.basename(folder_path)
    # try:
        # num_frames = unroll_mp4_into_jpgs(outpath, folder_path)
    # num_frames = unroll_mp4_into_latents(outpath, folder_path, vae)
    # except Exception as e:
    #     shutil.rmtree(folder_path)
    #     os.remove(jsonl_outpath)
    #     os.remove(outpath)
    #     return
    
    # os.remove(outpath)

    # update count
    # with frame_counter_lock:
    #     num_total_frames += num_frames
    #     demonstration_id_to_num_frames[demo_id] = num_frames

    pbar.update(1)
    print(f"Finished downloading {outpath}")

def get_paths(index_path: str, output_dir: str, num_demos: int):
    with open(index_path, "r") as f:
        data = f.read()
    data = eval(data)
    basedir = data["basedir"]
    relpaths = data["relpaths"]

    relpaths = random.sample(relpaths, num_demos)
    relpaths = relpaths_to_download(relpaths, output_dir)
    return basedir, relpaths

def download_minecraft_data(basedir: str, relpaths: list[str], output_dir: str):
    global num_total_frames
    global demonstration_id_to_num_frames
    num_total_frames = 0
    demonstration_id_to_num_frames = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ThreadPoolExecutor(MAX_THREADS) as executor:
        with tqdm(total=len(relpaths), desc="Downloading Files") as pbar:
            executor.map(lambda rp: download_video_and_action_files(basedir, rp, pbar), relpaths)

def worker(gpu_id, demo_ids, dataset_dir, return_dict):
    torch.cuda.set_device(f"cuda:{gpu_id}")
    vae_cfg = OmegaConf.load(CONFIG_PATH)["model"]["params"]["vae_config"]
    vae = instantiate_from_config(vae_cfg).to(f"cuda:{gpu_id}")
    for param in vae.parameters():
        param.requires_grad=False

    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    cursor_image = cursor_image[:16, :16, :] # Assume 16x16
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_image = cursor_image[:, :, :3]

    for demo_id in demo_ids:
        mp4_path = os.path.join(dataset_dir, f"{demo_id}.mp4")
        demo_output_dir = os.path.join(dataset_dir, demo_id)
        try:
            num_frames = unroll_mp4_into_latents(mp4_path, demo_output_dir, vae, gpu_id, cursor_image, cursor_alpha)
        except Exception as e:
            print(f"failed to process {demo_id} with error {e}, continuing...")
            continue
        return_dict[demo_id] = num_frames

def precompute_latents(dataset_dir: str):
    unique_ids = glob.glob(os.path.join(dataset_dir, "*.jsonl"))
    unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))    
    
    chunks = [unique_ids[i::2] for i in range(2)]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for gpu_id, demo_chunk in enumerate(chunks):
        p = mp.Process(target=worker, args=(gpu_id, demo_chunk, dataset_dir, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    demonstration_id_to_num_frames = dict(return_dict)
    num_total_frames = sum(demonstration_id_to_num_frames.values())

    print(f"total frames: {num_total_frames}")
    counts_dict = { 
        "total_frames": num_total_frames,
        "demonstration_id_to_num_frames": demonstration_id_to_num_frames
    }
    with open(os.path.join(dataset_dir, "counts.json"), "wt") as f:
        json.dump(counts_dict, f)


if __name__ == "__main__":
    args = parser.parse_args()
    
    # basedir, relpaths = get_paths(args.json_file, args.output_dir, args.num_demos)
    # relpaths = list(relpaths)
    # d = {
    #     "basedir": basedir,
    #     "split1": relpaths[::2],
    #     "split2": relpaths[1::2]
    # }
    with open("splits.json", "rt") as f:
        d = json.load(f)
    
    # download mp4s and jsons with a bunch o' threads
    basedir, relpaths = d["basedir"], d["split1"]
    relpaths = relpaths_to_download(relpaths, args.output_dir)
    download_minecraft_data(basedir, relpaths, args.output_dir)

    # use 2 procs (each with with a gpu) to precompute latents
    precompute_latents(args.output_dir)

