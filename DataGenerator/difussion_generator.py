from diffusers import StableDiffusionXLPipeline
import torch, os

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

os.makedirs("sd21", exist_ok=True)

prompts =  [...]
for i, p in enumerate(prompts):
    img = pipe(
        prompt=p,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    img.save(f"./sd21/{i:05d}.png")

