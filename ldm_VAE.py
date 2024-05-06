
from omegaconf import OmegaConf
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

config_path = 'autoencoder.yaml'
config = OmegaConf.load(config_path)

model = instantiate_from_config(config.model)
first_stage_model = model.eval()
first_stage_model.train = disabled_train
for param in first_stage_model.parameters():
    param.requires_grad = False