from diffusers import DiffusionPipeline
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5")
prompt = "a SPACE MINER MINING THE SURFACE OF AN ASTEROID IN SPACE in pixel art style"
image = pipe(prompt).images[0]
image.save("output.png")

