# MVD-Fusion

Before we start working on MVD-Fusion it is important to preapare the Dataset. For preparing the dataset follow: [Github](https://github.com/jfasate/objaverse-rendering)

## Environment Setup
Please follow the environment setup guide in [ENVIRONMENT.md](ENVIRONMENT.md).

## Dataset
Prepare Dataset has been provided on the [Hugging Face](https://huggingface.co/datasets/jfasate/objaverse)

## Pretrained Weights
MVD-Fusion requires Zero-1-to-3 weights, CLIP ViT weights, and finetuned MVD-Fusion weights. 
1. Find MVD-Fusion weights [here](https://huggingface.co/datasets/jfasate/weights) and download them to `weights/`, a full set of weights will have `weights/clip_vit_14.ckpt`, `weights/mvdfusion_sep23.pt`, and `weights/zero123_105000_cc.ckpt`.

## Training

* Zero123 weights are required for training (for initialization). Please download them and extract them to `weights/zero123_105000.ckpt`.

Sample training code is provided in `train.py`. Please follow the evaluation tutorial above to setup the environment and pretrained  weights. It is recommended to directly modify `configs/mvd_train.yaml` to specify the experiment directory and set the training hyperparameters. We show training flags below. We recommend a minimum of 4 GPUs for training. 


### Flags
```
-g, --gpus              number of gpus to use (default: 1)
-p, --port              last digit of DDP port (default: 1)
-b, --backend           distributed data parallel backend (default: nccl)
```

### Using Custom Datasets
To train on a custom dataset, one needs to write a custom dataloader. We describe the required outputs for the `__getitem__` function, which should be a dictionary containing:
```
{
  'images': (B, 3, H, W) image tensor,
  'R': (B, 3, 3) PyTorch3D rotation,
  'T': (B, 3) PyTorch3D translation,
  'f': (B, 2) PyTorch3D focal_length in NDC space,
  'c': (B, 2) PyTorch3D principal_point in NDC space,
}
```


## Citation
If you find this work useful, a citation will be appreciated via:

```
@inproceedings{hu2024mvdfusion,
    title={MVD-Fusion: Single-view 3D via Depth-consistent Multi-view Generation}, 
    author={Hanzhe Hu and Zhizhuo Zhou and Varun Jampani and Shubham Tulsiani},
    booktitle={CVPR},
    year={2024}
}
```

