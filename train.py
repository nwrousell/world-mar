import os
import datetime
import shutil
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import os, tempfile
from torchvision.transforms.functional import to_pil_image
from world_mar.modules.utils import instantiate_from_config

LOG_PARENT = "logs"

class ImageLogger(pl.Callback):
    def __init__(self, log_every_n_steps=1000):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.cached_patch_dims = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps != 0:
            return
        
        self.B = 4
        latents       = batch["frames"].to(pl_module.device)[:self.B]
        batch_nframes = batch["num_non_padding_frames"].to(pl_module.device)[:self.B]
        action        = batch["action"].to(pl_module.device)[:self.B]
        poses         = batch["plucker"].to(pl_module.device)[:self.B]
        timestamps    = batch["timestamps"].to(pl_module.device)[:self.B]
        
        # sample latents
        pred_latent = pl_module.sample(latents, action, poses, timestamps, batch_nframes, prev_masking=True)  # n 576 16

        # convert latents to actual frames in pixel-space using Oasis VAE decoder
        B, T, HW, D = latents.shape
        to_decode = torch.cat([*latents.unbind(dim=1), pred_latent], dim=0)

        pl_module.vae.to("cuda")
        with torch.autocast(device_type="cuda", enabled=False):
            decoded = pl_module.vae.decode(to_decode).clip(-1, 1)
            decoded = ((decoded + 1) / 2 * 255).to(torch.uint8)  # (B(T+1), C, H_pix, W_pix)
        pl_module.vae.to("cpu")

        decoded = rearrange(decoded, "(t b) c h w -> t b c h w", t=T+1, b=B)  # (T+1, B, C, H, W)
        decoded_inputs = decoded[:T]                                          # (T, B, C, H, W)
        decoded_inputs_masked = [frame.clone() for frame in decoded_inputs]   # (T, B, C, H, W)
        decoded_pred = decoded[T]                                             # (B, C, H, W)

        # NOTE: the following should be done after pl_module.sample() to get the correct data
        # retrieve token masks and prediction index from forward pass
        pred_idx        = pl_module._last_pred_idx                                      # int
        pred_mask       = pl_module._last_pred_mask[:self.B]                            # (B, H_patch*W_patch)
        pred_mask_iters = [mask[:self.B] for mask in pl_module._last_pred_masks_iters]  # list of (B, H_patch*W_patch)
        pred_iters      = [pred[:self.B] for pred in pl_module._last_pred_iters]        # list of (B, H_patch, W_patch, D)

        if pl_module._last_ctx_mask is not None:
            ctx_mask = pl_module._last_ctx_mask[:self.B]                                    # (B, P-1, H_patch*W_patch)
            ctx_mask_left = ctx_mask[:, :pred_idx, :]
            ctx_mask_middle = torch.zeros_like(pred_mask, dtype=torch.bool).unsqueeze(-2)
            ctx_mask_right = ctx_mask[:, pred_idx:, :]
            ctx_mask = torch.cat([ctx_mask_left, ctx_mask_middle, ctx_mask_right], dim=-2)  # (B, P, H_patch*W_patch)
        else:
            ctx_mask = None

        # compute pixel‐patch dims
        if not self.cached_patch_dims:
            B, C, H_pix, W_pix = decoded_pred.shape
            latent_h, latent_w = pl_module.vae_seq_h, pl_module.vae_seq_w
            self.pix_per_lat_h = H_pix // latent_h
            self.pix_per_lat_w = W_pix // latent_w
            self.H_patch = latent_h // pl_module.patch_size
            self.W_patch = latent_w // pl_module.patch_size
            self.cached_patch_dims = True

        # color masked patches on the target frame 10% grey
        self.color_masked_patches(pl_module, decoded_inputs_masked[0], pred_mask, 230)

        # color masked patches on the context frames white
        if ctx_mask is not None:
            P = ctx_mask.shape[1]
            for p_idx in range(P):
                self.color_masked_patches(pl_module, decoded_inputs_masked[p_idx], ctx_mask[:, p_idx, :], 255)

        # three strips attached along width (horizontally)
        masked_strip = torch.cat(list(reversed(decoded_inputs_masked)), dim=-1)
        gt_strip     = torch.cat(list(reversed(decoded_inputs)), dim=-1)
        pred_strip   = torch.cat([*[torch.full_like(decoded_pred, 255) for _ in range(T-1)], decoded_pred], dim=-1)

        # stack the three strips along height (vertically)
        visualizations = torch.cat([masked_strip, gt_strip, pred_strip], dim=-2)

        # build per‐sample image captions
        bnf = batch_nframes.detach().cpu().tolist()
        ts_raw = timestamps.detach().cpu().tolist()
        ac = action.detach().cpu().tolist()

        captions = []
        for batch_idx in range(len(visualizations)):
            valid_ts = ts_raw[batch_idx][:bnf[batch_idx]]
            ts_str = ",".join(str(t) for t in reversed(valid_ts))
            ac_str = ",".join(str(a) for a in ac[batch_idx])
            captions.append(f"step={global_step}  ts=[{ts_str}]  action=[{ac_str}]")

        # Log to wandb
        wandb_images = [wandb.Image(img, caption=caption) for img, caption in zip(visualizations, captions)]
        trainer.logger.experiment.log({"generated_images": wandb_images}, step=global_step)

        # create a GIF of the predicted frame through each MAR sampling iteration
        wandb_gifs = self.mar_sampling_gifs(pl_module, pred_mask_iters, pred_iters)
        trainer.logger.experiment.log({"mar_sampling_gifs": wandb_gifs}, step=global_step)

    def color_masked_patches(self, pl_module, decoded_frame, mask, color):
        # color masked patch tokens on pred frame
        B, C, H, W = decoded_frame.shape
        mask = mask.reshape(B, self.H_patch, self.W_patch)
        for batch_idx in range(B):
            rows, cols = mask[batch_idx].nonzero(as_tuple=True)
            for r, c in zip(rows.tolist(), cols.tolist()):
                # lower-left and upper-right patch corners
                y0 =   r   * pl_module.patch_size * self.pix_per_lat_h
                y1 = (r+1) * pl_module.patch_size * self.pix_per_lat_h
                x0 =   c   * pl_module.patch_size * self.pix_per_lat_w
                x1 = (c+1) * pl_module.patch_size * self.pix_per_lat_w
                # color the whole patch
                decoded_frame[batch_idx,:, y0:y1, x0:x1] = color
    
    def mar_sampling_gifs(self, pl_module, pred_mask_iters, pred_iters):
        device = pl_module.device
        num_iters = len(pred_mask_iters)
        B = len(pred_mask_iters[0])

        wandb_gifs = []

        for batch_idx in range(B):
            cur_mask = torch.zeros_like(pred_mask_iters[0][0]).to(device)
            frames = []

            for iter_idx in range(num_iters):
                # get mask & pred for this sample & iteration
                pred = pred_iters[iter_idx][batch_idx].to(device)       
                mask = pred_mask_iters[iter_idx][batch_idx].to(device)
                cur_mask[mask] = 1

                # prepare for decoding
                pred = rearrange(pred, "h w d -> 1 (h w) d")
                pred = pl_module.unpatchify(pred)
                pred = pred / pl_module.scale_factor

                # decode to pixels
                pl_module.vae.to(device)
                with torch.no_grad():
                    dec = pl_module.vae.decode(pred)
                    dec = dec.clamp(-1, 1)
                    dec = ((dec + 1) / 2 * 255).to(torch.uint8).cpu()
                pl_module.vae.to("cpu")

                # color un-diffused tokens on pred frame 10% grey
                mask = ~cur_mask.unsqueeze(0)
                self.color_masked_patches(pl_module, dec, mask, 230)
                frames.append(dec.squeeze(0))

            frames = [torch.full_like(frames[0], 230)] + frames  # add grey frame at the beginning
            wandb_gif = wandb.Video(torch.stack(frames), fps=1, format="gif")
            wandb_gifs.append(wandb_gif)

        return wandb_gifs

def get_callbacks(logdir):
    return [
        ModelCheckpoint(
            dirpath=os.path.join(logdir, "checkpoints/"),
            filename="epoch{epoch:02d}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        ImageLogger(),
        LearningRateMonitor(logging_interval='step')
    ]

def find_latest_checkpoint(logdir):
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"no checkpoint directory found in {checkpoint_dir}")
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
    latest_ckpt = max(ckpts, key=os.path.getctime)
    return latest_ckpt

def main(args):
    # loading conf
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config: {args.config}")

    # set up logdir
    if args.name:
        name = args.name
    else:
        cfg_fname = os.path.splitext(os.path.split(args.config)[-1])[0]
        now = datetime.datetime.now().strftime("%m-%d-T%H-%M-%S")
        name = now + "_" + cfg_fname

    if not args.resume:
        logdir = os.path.join(LOG_PARENT, name)
        os.makedirs(logdir, exist_ok=True)
        shutil.copy(args.config, os.path.join(logdir, os.path.basename(args.config)))
        ckpt_path = None
    else:
        logdir = args.resume
        ckpt_path = find_latest_checkpoint(args.resume)
    print(f"Initialized logging directory: {logdir}")

    # load model and dataloader from config
    model_cfg = cfg.model
    # model_cfg.params.vae_config = None # destroy vae_config so it doesn't load the vae
    model = instantiate_from_config(model_cfg)
    model.vae.to("cpu")
    print("Loaded model")
    dataloader_cfg = cfg.dataloader
    if args.data_dir:
        dataloader_cfg.dataset_dir = args.data_dir
    dataloader = instantiate_from_config(dataloader_cfg)
    print("Loaded dataloader")

    # set the learning rate of the model
    model.learning_rate = model_cfg.learning_rate
    print(f"Set initial learning rate to: {model.learning_rate}")

    # get the callback functions for each batch
    callbacks = get_callbacks(logdir)

    if torch.cuda.is_available():
        print("CUDA available; printing GPU info:")
        num_devices = torch.cuda.device_count()
        accelerator = "gpu"
        model.learning_rate *= num_devices
        #torch.set_float32_matmul_precision('medium')
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA unavailable; running on CPU.")
        num_devices = 1
        accelerator = "cpu"

    trainer = pl.Trainer(
        #strategy="ddp",
        accelerator=accelerator, devices=num_devices, precision="bf16-mixed",
        check_val_every_n_epoch=1,
        callbacks=callbacks, logger=WandbLogger(project="WorldMar", log_model="all", name=name, entity="praccho-brown-university")
    )

    print("Starting training...")
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path = ckpt_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="experiment name"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config"
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default="",
        type=str,
        help="path to data dir"
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        help="path to previous run's logdir"
    )
    
    args = parser.parse_args()

    main(args)