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
from tqdm import tqdm
import shutil

MAX_THREADS = 10

frame_counter_lock = Lock()
num_total_frames = 0
demonstration_id_to_num_frames = {}


parser = argparse.ArgumentParser(description="Download OpenAI contractor datasets")
parser.add_argument("--json-file", type=str, required=True, help="Path to the index .json file")
parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
parser.add_argument("--num-demos", type=int, default=None, help="Maximum number of demonstrations to download")

def relpaths_to_download(relpaths, output_dir):
    def read_json(file_name):
        with open(file_name.replace('mp4', 'jsonl'), 'r') as json_file:
            text = json.loads('['+''.join(json_file.readlines()).replace('\n', ',')+']')

    data_path = '/'.join(relpaths[0].split('/')[:-1])
    non_defect=[]
    for vid_name in glob.glob(os.path.join(output_dir,'*.mp4')):
        try:
            vid = cv2.VideoCapture(vid_name)
            read_json(vid_name.replace('mp4', 'jsonl'))
            if vid.isOpened():
                non_defect.append(os.path.join(data_path, vid_name.split('/')[-1]))
        except:
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
    folder_path = outpath.removesuffix(".mp4")
    demo_id = os.path.basename(folder_path)
    try:
        num_frames = unroll_mp4_into_jpgs(outpath, folder_path)
    except Exception as e:
        shutil.rmtree(folder_path)
        os.remove(jsonl_outpath)
        os.remove(outpath)
        return
    
    os.remove(outpath)

    # update count
    with frame_counter_lock:
        num_total_frames += num_frames
        demonstration_id_to_num_frames[demo_id] = num_frames

    pbar.update(1)
    print(f"Finished downloading and unrolling {outpath}")

def download_minecraft_data(json_file: str, num_demos: int, output_dir: str):
    global num_total_frames
    global demonstration_id_to_num_frames
    num_total_frames = 0
    demonstration_id_to_num_frames = {}
    with open(json_file, "r") as f:
        data = f.read()
    data = eval(data)
    basedir = data["basedir"]
    relpaths = data["relpaths"]
    if args.num_demos is not None:
        # relpaths = relpaths[:args.num_demos]
        relpaths = random.sample(relpaths, num_demos)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    relpaths = relpaths_to_download(relpaths, output_dir)
    
    with ThreadPoolExecutor(MAX_THREADS) as executor:
        with tqdm(total=len(relpaths), desc="Downloading Files") as pbar:
            executor.map(lambda rp: download_video_and_action_files(basedir, rp, pbar), relpaths)
    

    print(f"total frames: {num_total_frames}")
    counts_dict = { 
        "total_frames": num_total_frames,
        "demonstration_id_to_num_frames": demonstration_id_to_num_frames
    }
    with open(os.path.join(output_dir, "counts.json"), "wt") as f:
        json.dump(counts_dict, f)
        
if __name__ == "__main__":
    args = parser.parse_args()
    download_minecraft_data(args.json_file, args.num_demos, args.output_dir)