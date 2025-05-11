# WorldMAR

## Usage

To use this repo for training, follow the following instructions.

### 1) Setup environment

Create an environment with torch and install the addition needed dependencies with

`pip install -r requirements.txt`

### 2) Obtain OASIS VAE weights

Follow the instructions at the [Open Oasis](https://github.com/etched-ai/open-oasis) repository to obtain
weights for their pretrained VAE that we used in our experiments. Place the `vit-l-20.safetensors` in `checkpoints/oasis`.

### 3) Download the dataset

We provide a script to obtain the datasets we used, this can be done with

`python -m world_mar.dataset.download --output-dir <dir> --split <1 or 2>`

which will download one of the halfs of the whole dataset we use, as well as precompute latents for each
frame with the OASIS VAE, to alleviate the burden of doing this at train time.

### 4) Train!

Now, to actually train, you can run

`train.py -c <config path>`

We use OmegaConf to conveniently change any model, training, or dataset params, which can be viewed in the `configs` directory.
As an example, for our largest model `configs/world_mar_l.yaml`, this looks like

```yaml
model:
  learning_rate: 5e-5
  target: world_mar.models.mar.WorldMAR
  params:
    vae_config:
      target: world_mar.oasis_utils.vae.ViT_L_20_Shallow_Encoder 
      params:
        load_from_ckpt: checkpoints/oasis/vit-l-20.safetensors
    encoder_depth: 12
    decoder_depth: 12
    st_embed_dim: 512
    diffloss_w: 512
    num_frames: 4
    prev_masking_rate: 0.25

dataloader:
  target: world_mar.dataset.dataloader.MinecraftDataModule
  params:
    dataset_dir: "<DATASET_DIR>"
    batch_sz: 64
    memory_distance: 1000
    memory_size: 100
    train_split: 0.99 # 0.01 for val
    num_mem_frames: 2
    num_prev_frames: 1
    prev_distance: 1
    num_workers: 16
```

make sure to set the `dataset_dir` of the `dataloader` to the path where you downloaded the dataset.

## TODOs

- [x] dataloader w/ relative pose
- [x] ROPE embeddings ([from WorldMem](https://github.com/xizaoqu/WorldMem/blob/main/algorithms/worldmem/models/rotary_embedding_torch.py))
- [x] retrieval func based on pose
- [x] Architecture
    - [x] ST transformer
    - [x] Oasis VAE
    - [x] Diffusion Model
    - [ ] _Optional:_ [Pose Predictor](https://github.com/xizaoqu/WorldMem/blob/main/algorithms/worldmem/models/pose_prediction.py)
- [x] training loop

## Relevant Papers
- [WorldMem](https://www.arxiv.org/pdf/2504.12369)
- [MAR](https://arxiv.org/pdf/2406.11838)
- [MARdini](https://arxiv.org/pdf/2410.20280)
- [Oasis](https://oasis-model.github.io/)
- [RoPE](https://arxiv.org/pdf/2104.09864)
- [2D RoPE](https://arxiv.org/pdf/2403.13298)

## Dataset
- VPT dataset collected by OpenAI contractors
- download script from [MineRL challenge](https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline)

