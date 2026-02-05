import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger, get_final_resolutions
import comfy.model_management as mm
import time
import tensorrt
import json 

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")

# --- SETTINGS FÜR DEIN SYSTEM (High-Res) ---
IMAGE_DIM_MIN = 64
IMAGE_DIM_OPT = 1024
IMAGE_DIM_MAX = 2600  # Deine angepasste Max-Größe

# --- Config Loader ---
def load_node_config(config_filename="load_upscaler_config.json"):
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)
    default_config = {
        "model": {"options": ["4x-UltraSharp"], "default": "4x-UltraSharp"},
        "precision": {"options": ["fp16", "fp32"], "default": "fp16"}
    }
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return default_config

LOAD_UPSCALER_NODE_CONFIG = load_node_config()

# ==============================================================================
# V2 NODE: UPSCALER (Verarbeitet das Bild)
# ==============================================================================
class UpscalerTensorrtV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": f"Images to be upscaled. Resolution must be between {IMAGE_DIM_MIN} and {IMAGE_DIM_MAX} px"}),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL", {"tooltip": "Tensorrt model built with V2 loader"}),
                "resize_to": (["none", "custom", "HD", "FHD", "2k", "4k", "1x", "1.5x", "2x", "2.5x", "3x", "3.5x", "4x", "5x", "6x", "7x", "8x", "9x", "10x"], {"tooltip": "Resize the upscaled image"}),
                "resize_width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "resize_height": ("INT", {"default": 1024, "min": 1, "max": 8192}),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt_v2"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Upscale images with tensorrt V2 (Explicit Scale Support)"

    def upscaler_tensorrt_v2(self, **kwargs):
        images = kwargs.get("images")
        upscaler_trt_model = kwargs.get("upscaler_trt_model")
        resize_to = kwargs.get("resize_to")

        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        # Check Dimensions
        for dim in (H, W):
            if dim > IMAGE_DIM_MAX or dim < IMAGE_DIM_MIN:
                raise ValueError(f"Input image dimensions fall outside of the supported range: {IMAGE_DIM_MIN} to {IMAGE_DIM_MAX} px! Input: {W}x{H}")

        # Scale Faktor aus der geladenen Engine holen (wurde im V2 Loader gesetzt)
        # Fallback auf 4, falls irgendwas schief geht
        model_scale = getattr(upscaler_trt_model, "upscale_factor", 4)

        if resize_to == "custom":
            final_width = kwargs.get("resize_width")
            final_height = kwargs.get("resize_height")
        else:
            final_width, final_height = get_final_resolutions(W, H, resize_to, model_scale)

        logger.info(f"V2 Upscale | Model Scale: {model_scale}x | Input: {W}x{H} -> Native Output: {W*model_scale}x{H*model_scale} | Final: {final_width}x{final_height}")

        # Shape Dictionary basierend auf dem expliziten Scale Faktor
        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H*model_scale, W*model_scale)}, 
        }
        
        upscaler_trt_model.activate()
        upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))

        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device())
        
        must_resize = (W * model_scale != final_width) or (H * model_scale != final_height)

        for i, img in enumerate(images_list):
            result = upscaler_trt_model.infer({"input": img}, cudaStream)
            result = result["output"]

            if must_resize:
                result = torch.nn.functional.interpolate(
                    result, size=(final_height, final_width), mode='bicubic', antialias=True
                )
            upscaled_frames[i] = result.to(mm.intermediate_device())
            pbar.update(1)

        output = upscaled_frames.permute(0, 2, 3, 1)
        upscaler_trt_model.reset()
        mm.soft_empty_cache()

        return (output,)

# ==============================================================================
# V2 NODE: LOADER (Baut/Lädt die Engine mit explizitem Namen)
# ==============================================================================
class LoadUpscalerTensorrtModelV2:
    @classmethod
    def INPUT_TYPES(cls): 
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("model", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})
        
        model_options = model_config.get("options", ["4x-UltraSharp"])
        model_default = model_config.get("default", "4x-UltraSharp")
        
        # Hier ist die neue manuelle Auswahl
        upscale_options = ["4x", "2x", "1x", "8x"]

        return {
            "required": {
                "model": (model_options, {"default": model_default}),
                "precision": (precision_config.get("options", ["fp16"]), {"default": "fp16"}),
                "upscale_type": (upscale_options, {"default": "4x", "tooltip": "Explicitly set the model scale (e.g. 1x for detailers, 4x for standard upscalers)."}),
            }
        }
    
    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load/Build TRT Model with explicit scale and resolution params in filename."
    FUNCTION = "load_upscaler_tensorrt_model_v2"
    
    def load_upscaler_tensorrt_model_v2(self, model, precision, upscale_type):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        # Scale Integer extrahieren ("4x" -> 4)
        scale_int = int(upscale_type.replace("x", ""))
        
        # ONNX Pfad
        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")
        
        # Parameter für den Engine-Namen
        engine_channel = 3
        # Hier nutzen wir deine globalen Limits
        min_h, min_w = IMAGE_DIM_MIN, IMAGE_DIM_MIN
        max_h, max_w = IMAGE_DIM_MAX, IMAGE_DIM_MAX
        
        # NEUER NAMENS-STANDARD:
        # {Model}_{Precision}_{Scale}x_{Min}x{Min}_{Max}x{Max}_{TrtVersion}.trt
        engine_name = f"{model}_{precision}_{upscale_type}_{min_h}x{min_w}_{max_h}x{max_w}_{tensorrt.__version__}.trt"
        tensorrt_model_path = os.path.join(tensorrt_models_dir, engine_name)

        # 1. Prüfen, ob genau diese Engine schon existiert
        if os.path.exists(tensorrt_model_path):
            logger.info(f"V2: Found existing engine matching criteria: {engine_name}")
        else:
            # 2. Wenn nicht, neu bauen
            logger.info(f"V2: Engine not found or params changed. Building new: {engine_name}")
            
            if not os.path.exists(onnx_model_path):
                # Download Fallback
                onnx_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading ONNX: {onnx_url}")
                download_file(url=onnx_url, save_path=onnx_model_path)

            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            
            # Engine Build mit den Limits
            # Wir nutzen hier statische Batch Size 1 für optimale Performance in Comfy
            engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" else False,
                input_profile=[
                    {"input": [
                        (1, 3, min_h, min_w),    # Min
                        (1, 3, IMAGE_DIM_OPT, IMAGE_DIM_OPT), # Opt
                        (1, 3, max_h, max_w)     # Max
                    ]},
                ],
            )
            e = time.time()
            logger.info(f"V2: Build finished in {(e-s):.2f} seconds")

        # Laden
        logger.info(f"V2: Loading Engine...")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()
        
        # WICHTIG: Den Scale-Faktor an das Objekt heften, damit der Upscaler-Node ihn kennt
        engine.upscale_factor = scale_int

        return (engine,)

# Alte Klassen (V1) können bleiben oder entfernt werden, hier bleiben sie für Kompatibilität
# (Ich habe sie hier gekürzt, um Fokus auf V2 zu legen, aber im File solltest du sie behalten wenn du alte Workflows nicht brechen willst)
class UpscalerTensorrt: # Legacy V1
    @classmethod
    def INPUT_TYPES(s): return {"required": {"images": ("IMAGE",), "upscaler_trt_model": ("UPSCALER_TRT_MODEL",), "resize_to": (["none"],)}}
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt"
    CATEGORY = "tensorrt"
    def upscaler_tensorrt(self, **kwargs): pass # Legacy Logic here if needed

# MAPPINGS
NODE_CLASS_MAPPINGS = {
    "UpscalerTensorrt": UpscalerTensorrtV2,         # Wir ersetzen den Standard V1 durch V2 Logic oder nutzen separate Namen
    "LoadUpscalerTensorrtModel": LoadUpscalerTensorrtModelV2, # Empfehlung: Ersetze die Standard Nodes direkt, ist sauberer im Fork
    # Falls du beide parallel willst:
    "UpscalerTensorrtV2": UpscalerTensorrtV2,
    "LoadUpscalerTensorrtModelV2": LoadUpscalerTensorrtModelV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrt": "Upscaler Tensorrt ⚡ (Legacy)",
    "LoadUpscalerTensorrtModel": "Load Upscale Tensorrt Model (Legacy)",
    "UpscalerTensorrtV2": "Upscaler Tensorrt V2 ⚡",
    "LoadUpscalerTensorrtModelV2": "Load Upscale Tensorrt Model V2",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
