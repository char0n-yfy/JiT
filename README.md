## Just image Transformer (JiT) for Pixel-space Diffusion

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2511.13720-b31b1b.svg)](https://arxiv.org/abs/2511.13720)&nbsp;

<p align="center">
  <img src="demo/visual.jpg" width="100%">
</p>


This is a PyTorch/GPU re-implementation of the paper [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720):

```
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

JiT adopts a minimalist and self-contained design for pixel-level high-resolution image diffusion. 
The original implementation was in JAX+TPU. This re-implementation is in PyTorch+GPU.

<p align="center">
  <img src="demo/jit.jpg" width="40%">
</p>

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/LTH14/JiT.git
cd JiT
```

A suitable [conda](https://conda.io/) environment named `jit` can be created and activated with:

```
conda env create -f environment.yaml
conda activate jit
```

If you get ```undefined symbol: iJIT_NotifyEvent``` when importing ```torch```, simply
```
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Check this [issue](https://github.com/conda/conda/issues/13812#issuecomment-2071445372) for more details.

### Config (YAML)
All hyperparameters can be specified in a YAML config and loaded via `--config` (CLI args override config).
See `configs/jit_uncond_imagenet256.yaml:1`.

### Training
The below training scripts have been tested on 8 H200 GPUs.

Example script for training JiT-B/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval_denoise \
--denoise_split val --denoise_t 0.5 --denoise_mode ode --denoise_steps 50 --denoise_num_images 2048 \
--eval_bsz 128
```

Example script for training JiT-B/32 on ImageNet 512x512 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 512 --noise_scale 2.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval_denoise \
--denoise_split val --denoise_t 0.5 --denoise_mode ode --denoise_steps 50 --denoise_num_images 2048 \
--eval_bsz 128
```

Example script for training JiT-H/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval_denoise \
--denoise_split val --denoise_t 0.5 --denoise_mode ode --denoise_steps 50 --denoise_num_images 2048 \
--eval_bsz 128
```

### Evaluation

Evaluate denoising on real images by: x -> z_t -> x_hat, report pixel MSE/L1/PSNR:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--img_size 256 --noise_scale 1.0 \
--denoise_split val --denoise_t 0.5 --denoise_mode ode --denoise_steps 50 --denoise_num_images 2048 \
--eval_bsz 128 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_denoise
```

### Acknowledgements

We thank Google TPU Research Cloud (TRC) for granting us access to TPUs, and the MIT
ORCD Seed Fund Grants for supporting GPU resources.

### Contact

If you have any questions, feel free to contact me through email (tianhong@mit.edu). Enjoy!
