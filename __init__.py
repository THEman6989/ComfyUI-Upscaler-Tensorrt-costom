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
# V2 NODE: UPSCALER (Dynamische Prüfung der Grenzen)
# ==============================================================================
class UpscalerTensorrtV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images"}),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL", {"tooltip": "Tensorrt model loaded via V2 Loader"}),
                "resize_to": (["none", "custom", "HD", "FHD", "2k", "4k", "1x", "1.5x", "2x", "2.5x", "3x", "3.5x", "4x", "5x", "6x", "7x", "8x", "9x", "10x"], {"tooltip": "Resize the upscaled image"}),
            },
            "optional": {
                # Diese Inputs werden nur genutzt, wenn resize_to="custom" ist.
                # Sie haben nichts mit den Engine-Grenzen zu tun (die kommen jetzt aus dem Model).
                "resize_width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "resize_height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt_v2"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Upscale with TRT. Checks image size against Engine limits automatically."

    def upscaler_tensorrt_v2(self, **kwargs):
        images = kwargs.get("images")
        upscaler_trt_model = kwargs.get("upscaler_trt_model")
        resize_to = kwargs.get("resize_to")

        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        # 1. Dynamische Grenzen aus dem Model holen
        # Falls das Model alt ist (V1), nutzen wir Fallbacks (256 - 1280)
        engine_min = getattr(upscaler_trt_model, "input_min", 256)
        engine_max = getattr(upscaler_trt_model, "input_max", 1280)
        model_scale = getattr(upscaler_trt_model, "upscale_factor", 4)

        # 2. Validierung: Ist das Bild im Rahmen der gebauten Engine?
        for dim, name in zip((H, W), ("Height", "Width")):
            if dim < engine_min or dim > engine_max:
                raise ValueError(
                    f"❌ Image Error: Input {name} ({dim}px) is out of bounds for this TRT Engine!\n"
                    f"   Engine Limits: Min {engine_min}px | Max {engine_max}px\n"
                    f"   👉 Solution: Adjust 'min_engine_size' or 'max_engine_size' in the LoadUpscalerTensorrtModelV2 node and reload."
                )

        if resize_to == "custom":
            final_width = kwargs.get("resize_width", 1024)
            final_height = kwargs.get("resize_height", 1024)
        else:
            final_width, final_height = get_final_resolutions(W, H, resize_to, model_scale)

        logger.info(f"V2 Upscale | Scale: {model_scale}x | Input: {W}x{H} (OK within {engine_min}-{engine_max}) -> Output: {final_width}x{final_height}")

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
# V2 NODE: LOADER (Engine bauen mit benutzerdefinierten Grenzen)
# ==============================================================================
class LoadUpscalerTensorrtModelV2:
    @classmethod
    def INPUT_TYPES(cls): 
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("model", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})
        
        model_options = model_config.get("options", ["4x-UltraSharp"])
        model_default = model_config.get("default", "4x-UltraSharp")
        
        upscale_options = ["4x", "2x", "1x", "8x"]

        return {
            "required": {
                "model": (model_options, {"default": model_default}),
                "precision": (precision_config.get("options", ["fp16"]), {"default": "fp16"}),
                "upscale_type": (upscale_options, {"default": "4x", "tooltip": "Explicit scale factor (1x, 2x, 4x)."}),
                
                # --- HIER SIND DIE NEUEN DYNAMISCHEN GRENZEN ---
                "min_engine_size": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64, "tooltip": "Minimum resolution the engine supports."}),
                "max_engine_size": ("INT", {"default": 2048, "min": 512, "max": 16384, "step": 256, "tooltip": "Maximum resolution the engine supports (affects VRAM usage and build time)."}),
            }
        }
    
    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load/Build TRT Model with explicit scale and CUSTOM resolution limits."
    FUNCTION = "load_upscaler_tensorrt_model_v2"
    
    def load_upscaler_tensorrt_model_v2(self, model, precision, upscale_type, min_engine_size, max_engine_size):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        scale_int = int(upscale_type.replace("x", ""))
        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")
        
        # Optimalwert für die Engine (in der Mitte)
        opt_size = (min_engine_size + max_engine_size) // 2

        # 1. Dateiname generieren (enthält jetzt min/max size!)
        engine_name = f"{model}_{precision}_{upscale_type}_{min_engine_size}x{min_engine_size}_{max_engine_size}x{max_engine_size}_{tensorrt.__version__}.trt"
        tensorrt_model_path = os.path.join(tensorrt_models_dir, engine_name)

        if os.path.exists(tensorrt_model_path):
            logger.info(f"V2: Found existing engine: {engine_name}")
        else:
            logger.info(f"V2: Building NEW engine with custom limits: {min_engine_size}px - {max_engine_size}px")
            
            if not os.path.exists(onnx_model_path):
                onnx_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading ONNX: {onnx_url}")
                download_file(url=onnx_url, save_path=onnx_model_path)

            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            
            # 2. Engine bauen mit den User-Inputs
            engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" else False,
                input_profile=[
                    {"input": [
                        (1, 3, min_engine_size, min_engine_size), # Min
                        (1, 3, opt_size, opt_size),                 # Opt
                        (1, 3, max_engine_size, max_engine_size)    # Max
                    ]},
                ],
            )
            e = time.time()
            logger.info(f"V2: Build finished in {(e-s):.2f} seconds")

        logger.info(f"V2: Loading Engine...")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()
        
        # 3. Metadaten im Objekt speichern (damit der Upscaler Node sie prüfen kann)
        engine.upscale_factor = scale_int
        engine.input_min = min_engine_size
        engine.input_max = max_engine_size

        return (engine,)

# Legacy Nodes (auskommentiert oder beibehalten)
class UpscalerTensorrt:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"images": ("IMAGE",), "upscaler_trt_model": ("UPSCALER_TRT_MODEL",), "resize_to": (["none"],)}}
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt"
    CATEGORY = "tensorrt"
    def upscaler_tensorrt(self, **kwargs): pass 

NODE_CLASS_MAPPINGS = {
    "UpscalerTensorrtV2": UpscalerTensorrtV2,
    "LoadUpscalerTensorrtModelV2": LoadUpscalerTensorrtModelV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrtV2": "Upscaler Tensorrt V2 ⚡",
    "LoadUpscalerTensorrtModelV2": "Load Upscale Tensorrt Model V2",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
