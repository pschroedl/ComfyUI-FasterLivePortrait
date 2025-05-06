from faster_live_portrait import FasterLivePortraitPipeline
from omegaconf import OmegaConf
from .config import get_live_portrait_config
import numpy as np
import torch
import cv2

class FasterLivePortrait:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
            },
            "optional": {
                "flag_normalize_lip": ("BOOLEAN", {"default": False}),
                "flag_source_video_eye_retargeting": ("BOOLEAN", {"default": False}),
                "flag_video_editing_head_rotation": ("BOOLEAN", {"default": False}),
                "flag_eye_retargeting": ("BOOLEAN", {"default": False}),
                "flag_lip_retargeting": ("BOOLEAN", {"default": False}),
                "flag_stitching": ("BOOLEAN", {"default": True}),
                "flag_pasteback": ("BOOLEAN", {"default": True}),
                "flag_do_crop": ("BOOLEAN", {"default": True}),
                "flag_do_rot": ("BOOLEAN", {"default": True}),
                "lip_normalize_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "driving_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "animation_region": (["lip", "full"], {"default": "lip"}), # currently only works for lip
                "cfg_mode": (["incremental", "reference"], {"default": "incremental"}),
                "cfg_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "source_max_dim": ("INT", {"default": 1280, "min": 64, "step": 8}),
                "source_division": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1})
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "FasterLivePortrait"
    def __init__(self):
        config_dict = get_live_portrait_config()
        self.pipeline = FasterLivePortraitPipeline(cfg=OmegaConf.create(config_dict), is_animal=False)
        self._last_cfg_inputs = {}
        self._cfg_keys = [
            "flag_normalize_lip",
            "flag_source_video_eye_retargeting",
            "flag_video_editing_head_rotation",
            "flag_eye_retargeting",
            "flag_lip_retargeting",
            "flag_stitching",
            "flag_pasteback",
            "flag_do_crop",
            "flag_do_rot",
            "lip_normalize_threshold",
            "driving_multiplier",
            "animation_region",
            "cfg_mode",
            "cfg_scale",
            "source_max_dim",
            "source_division"
        ]

    def process_image(self, source, target, **kwargs):
        for k in self._cfg_keys:
            new_val = kwargs[k]
            if self._last_cfg_inputs.get(k) != new_val:
                self.pipeline.set_cfg_param(k, new_val)
                self._last_cfg_inputs[k] = new_val
        source_np = tensor_to_cv2(source)
        target_np = tensor_to_cv2(target)
        processed_image = self.pipeline.animate_image(source_np, target_np)
        if processed_image is None:
            tensor = torch.from_numpy(source_np.astype(np.float32) / 255.0)
        else:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            tensor = torch.from_numpy(processed_image.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0)
        return (tensor,)
    
def tensor_to_cv2(tensor):
    arr = tensor.detach().cpu().numpy()
    # Remove batch dimension if present (N, C, H, W)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]  # now (C, H, W)

    # If shape is (C, H, W), convert to (H, W, C)
    # if arr.ndim == 3 and arr.shape[0] in [1, 3]:
    #     arr = np.transpose(arr, (1, 2, 0))

    arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Convert from float [0,1] to uint8 [0,255] if needed
    if arr.dtype in [np.float32, np.float64]:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr

NODE_CLASS_MAPPINGS = {
    "FasterLivePortrait": FasterLivePortrait,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FasterLivePortrait": "FasterLivePortrait",
}