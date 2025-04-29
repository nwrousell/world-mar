# world-mar

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

