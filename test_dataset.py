from omegaconf import OmegaConf
from utils.load_model import instantiate_from_config

config = OmegaConf.load('configs/mvd_train.yaml')
dataset = instantiate_from_config(config['dataset'])

print(f"✅ Dataset size: {len(dataset)}")
print("Loading first batch...")
batch = dataset[0]
print(f"✅ Images shape: {batch['images'].shape}")
print(f"✅ Depths shape: {batch['depths'].shape if 'depths' in batch else 'No depths'}")
print("✅ Dataset loads successfully!")
