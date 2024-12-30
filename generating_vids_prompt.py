import os
import random
from diffusers import CogVideoXImageToVideoPipeline
from PIL import Image
import torch
import numpy as np
import shutil
from tqdm import tqdm
from diffusers.utils import export_to_video, load_image

# dir_path = "detect_video_generation/detect_image"
# img_paths = [os.path.join(dir_path, f"{i:02d}.jpg") for i in range(1, 11)]

captions = [
    "A worker accidentally touches molten metal, risking severe burns and scalds.",
    "An operator's hand gets caught between moving machinery parts, causing a crushing injury.",
    "A worker slips on an oil spill in the workshop, leading to a potential fall hazard.",
    "An employee handling sharp metal sheets suffers a deep cut on their hand.",
    "A welder's unprotected eyes are injured by flying metal debris and sparks.",
    "A worker inhales toxic fumes from welding without proper ventilation, endangering respiratory health.",
    "An employee works near loud machinery without ear protection, risking permanent hearing loss.",
    "A technician suffers an electric shock due to contact with a live wire during repairs.",
    "A worker accidentally spills a chemical, causing burns on their arm and face.",
    "An employee lifting heavy metal beams experiences severe back strain and discomfort."
]

output_dir = "output_vids"
model_name = "CogVideoX-5b-I2V"
os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)

# for img_path in tqdm(img_paths):
#     image = Image.open(img_path)
    
#     for caption in tqdm(captions):
#         pipe = CogVideoXImageToVideoPipeline.from_pretrained(
#             model_name,
#             torch_dtype=torch.bfloat16
#         )
#         pipe.vae.enable_tiling()
#         pipe.vae.enable_slicing()
        
#         prompt = caption
#         image = load_image(image=img_path)
        
#         generator = torch.Generator(device="cuda").manual_seed(42)
#         video = pipe(
#             prompt=prompt,
#             image=image,
#             num_videos_per_prompt=1,
#             num_inference_steps=50,
#             num_frames=49,
#             guidance_scale=6,
#             generator=generator
#         ).frames[0]
        
#         img_name = os.path.basename(img_path).split('.')[0]
#         output_path = os.path.join(output_dir, model_name, f"{img_name}_{caption.replace(' ', '_')}.mp4")
#         export_to_video(video, output_path, fps=8)



import os
import random
from diffusers import I2VGenXLPipeline
from PIL import Image
import torch
import numpy as np
import shutil
from tqdm import tqdm

# dir_path = "detect_video_generation/detect_image"
# img_paths = [os.path.join(dir_path, f"{i:02d}.jpg") for i in range(1, 11)]

img_paths = ['/data/teja/video_generation_works/images/aluminum_sheets.png']


captions = [
    "A worker accidentally touches molten metal, risking severe burns and scalds.",
    "An operator's hand gets caught between moving machinery parts, causing a crushing injury.",
    "A worker slips on an oil spill in the workshop, leading to a potential fall hazard.",
    "An employee handling sharp metal sheets suffers a deep cut on their hand.",
    "A welder's unprotected eyes are injured by flying metal debris and sparks.",
    "A worker inhales toxic fumes from welding without proper ventilation, endangering respiratory health.",
    "An employee works near loud machinery without ear protection, risking permanent hearing loss.",
    "A technician suffers an electric shock due to contact with a live wire during repairs.",
    "A worker accidentally spills a chemical, causing burns on their arm and face.",
    "An employee lifting heavy metal beams experiences severe back strain and discomfort."
]

negative_prompt = "Distorted"

output_dir = "output_vids"
model_name = "i2vgen-xl"
os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)

for img_path in tqdm(img_paths):
    image = Image.open(img_path)
    
    for caption in tqdm(captions):
        pipeline = I2VGenXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()
        
        prompt = caption
        image = load_image(image=img_path)
        
        generator = torch.manual_seed(8888)
        frames = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=50,
            negative_prompt=negative_prompt,
            guidance_scale=9.0,
            generator=generator
        ).frames[0]
        
        img_name = os.path.basename(img_path).split('.')[0]
        output_path = os.path.join(output_dir, model_name, f"{img_name}_{caption.replace(' ', '_')}.mp4")
        export_to_video(frames, output_path)