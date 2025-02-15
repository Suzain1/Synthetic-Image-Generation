# -*- coding: utf-8 -*-
"""Image Generation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YdWcI6FRfWxKQkQU2mLIyTFbkyAXa_mh
"""

from diffusers import StableDiffusionPipeline
import torch
import os

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "A snowy forest with colorful northern lights shining above"

output_folder = "/content/drive/MyDrive/Synthetic image generation/generated"  # output folder

os.makedirs(output_folder, exist_ok=True)
for i in range(3):
    image = pipe(prompt).images[0]
    image_path = os.path.join(output_folder, f"generated_image_{i+1}.png")
    image.save(image_path)
print(f"Images generated and saved successfully in '{output_folder}'")

