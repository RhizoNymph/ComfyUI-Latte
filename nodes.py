import torch
import numpy as np
from PIL import Image
from diffusers import LattePipeline
import comfy.model_management as mm

class LatteVideoGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A small cactus with a happy face in the Sahara desert."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 500
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "video_length": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 16
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 1024,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 1024,
                    "step": 64
                }),
                "num_images_per_prompt": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4
                }),
                "seed": ("INT", {
                    "default": -1
                }),
            },
            "optional": {
                "image": ("IMAGE", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_video"
    CATEGORY = "Video Generation"

    def __init__(self):
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            self.pipe = LattePipeline.from_pretrained("maxin-cn/Latte-1", torch_dtype=torch.float16).to(self.device)
            self.pipe.enable_model_cpu_offload()

    def generate_video(self, prompt, negative_prompt, num_inference_steps, guidance_scale, video_length, width, height, num_images_per_prompt, seed, image=None):
        self.load_model()

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "video_length": video_length,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator
        }

        if image is not None:
            kwargs["image"] = self.prepare_image(image)

        output = self.pipe(**kwargs)
        videos = output.frames
        
        # Convert list of PIL Images to a single tensor
        video_tensors = []
        for frame in videos:
            # Convert PIL Image to numpy array
            frame_np = np.array(frame)
            # Add to list
            video_tensors.append(frame_np)
        
        # Stack numpy arrays into a single tensor
        batch_tensor = torch.from_numpy(np.stack(video_tensors))
        
        # Reshape and reorder dimensions
        batch_tensor = batch_tensor.squeeze(0)  # Remove batch dimension if present
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # Reorder to (F, C, H, W)
        
        # Convert to float and normalize to [0, 1] range
        batch_tensor = batch_tensor.float() / 255.0
        
        # Reorder to (F, H, W, C) for compatibility with ComfyUI
        batch_tensor = batch_tensor.permute(0, 2, 3, 1)
        
        # Offload model if needed
        self.pipe.to(self.offload_device)
        mm.soft_empty_cache()

        return (batch_tensor,)

    def prepare_image(self, image):
        # Convert ComfyUI image to PIL Image
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        if image.shape[0] == 3:  # If the image is in CHW format
            image = image.transpose(1, 2, 0)
        image = Image.fromarray((image * 255).astype(np.uint8))
        return image

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LatteVideoGenerator": LatteVideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatteVideoGenerator": "Latte Video Generator"
}