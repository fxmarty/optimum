# DISCLAIMER: create_scriptmodule.py needs to be run with the --for-gpu flag for this script to work

import torch

from utils import numpy_to_pil, StableDiffusionPreprocessor

from transformers import AutoTokenizer, CLIPTextConfig

from diffusers.schedulers import PNDMScheduler

scripted_pipeline = torch.load("scripted_sd_cuda.pt")

# NOTE: Beware that model_path should match with the .pt model!
#model_path = "hf-internal-testing/tiny-stable-diffusion-torch"
model_path = "CompVis/stable-diffusion-v1-4"
text_encoder_config = CLIPTextConfig.from_pretrained(model_path, subfolder="text_encoder")
tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")

preprocessor = StableDiffusionPreprocessor(
    tokenizer,
    text_encoder_config,
    do_classifier_free_guidance=True,
    scheduler=scheduler,
)

num_inference_steps = 50
preprocessed_input = preprocessor.preprocess("A cat sleeping on the beach", num_inference_steps=num_inference_steps, device="cuda")

text_input_ids = preprocessed_input["text_input_ids"].to("cuda")
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"].to("cuda")
timesteps = preprocessed_input["timesteps"].to("cuda")

scripted_pipeline = scripted_pipeline.to("cuda")

print(text_input_ids)
print(uncond_text_input_ids)
print(timesteps)

print("Running inference...")
with torch.inference_mode():
    torch_image = scripted_pipeline(
        text_input_ids=text_input_ids,
        uncond_text_input_ids=uncond_text_input_ids,
        timesteps=timesteps,
    )[0][0] # first item in "image"

np_image = torch_image.cpu().float().numpy()

image = numpy_to_pil(np_image)
image[0].save("scripted_out_gpu.png")