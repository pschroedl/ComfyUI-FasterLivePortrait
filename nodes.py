from faster_live_portrait import FasterLivePortraitPipeline
from omegaconf import OmegaConf
from .config import LIVE_PORTRAIT_INFER_CFG

class FasterLivePortrait:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
            },
            "optional": {
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "FasterLivePortrait"

    def process_image(self, source, target):
        pipeline = FasterLivePortraitPipeline(cfg=OmegaConf.create(LIVE_PORTRAIT_INFER_CFG), is_animal=False)
        processed_image = pipeline.animate_image(source, target)
        return (processed_image,)

NODE_CLASS_MAPPINGS = {
    "FasterLivePortrait": FasterLivePortrait,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FasterLivePortrait": "FasterLivePortrait",
}