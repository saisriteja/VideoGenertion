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



















# import torch
# from diffusers import CogVideoXImageToVideoPipeline
# from diffusers.utils import export_to_video, load_image

# prompt = "people moving and gun firing."
# image = load_image(image="gta.jpg")
# pipe = CogVideoXImageToVideoPipeline.from_pretrained(
#     "CogVideoX-5b-I2V",
#     torch_dtype=torch.bfloat16
# )

# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()

# video = pipe(
#     prompt=prompt,
#     image=image,
#     num_videos_per_prompt=1,
#     num_inference_steps=50,
#     num_frames=49,
#     guidance_scale=6,
#     generator=torch.Generator(device="cuda").manual_seed(42),
# ).frames[0]

# export_to_video(video, "output.mp4", fps=8)





# import torch
# from diffusers import StableVideoDiffusionPipeline
# from diffusers.utils import load_image, export_to_video

# pipeline = StableVideoDiffusionPipeline.from_pretrained(
#     "stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
# )
# pipeline.enable_model_cpu_offload()

# # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
# # image = image.resize((1024, 576))

# import PIL
# image = PIL.Image.open("gta.jpg")
# image = image.resize((1024, 576))


# generator = torch.manual_seed(42)
# frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
# export_to_video(frames, "generated.mp4", fps=7)




# import torch
# from diffusers import I2VGenXLPipeline
# from diffusers.utils import export_to_gif, load_image
# import PIL

# pipeline = I2VGenXLPipeline.from_pretrained("i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
# pipeline.enable_model_cpu_offload()

# image = PIL.Image.open("gta.jpg")
# # image = image.resize((1024, 576))

# prompt = "Papers were floating in the air on a table in the library"
# negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
# generator = torch.manual_seed(8888)

# frames = pipeline(
#     prompt=prompt,
#     image=image,
#     num_inference_steps=50,
#     negative_prompt=negative_prompt,
#     guidance_scale=9.0,
#     generator=generator
# ).frames[0]
# export_to_gif(frames, "i2v.gif")
