import torch 
from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
from PIL import Image
from torch import autocast
import yaml

def init_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)


    scheduler = DPMSolverMultistepScheduler.from_pretrained(config['model_path'], subfolder="scheduler", use_karras_sigmas=True)

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path = config['model_path'],
        torch_dtype = torch.float16,
        variant = "fp16",
        #safety_checker=None,
        scheduler=scheduler,
    )
    
    return pipeline, config

def inpaint(model, image_path, mask_path, config, seed=3119, device="cuda"): 
    model.enable_model_cpu_offload()

    generator = torch.Generator("cuda").manual_seed(seed)

    img  = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    orig_width, orig_height = img.size

    with autocast(device):
        result = model(prompt=config['prompt'],
                       image=img, 
                       mask_image=mask,
                       num_inference_steps=config['num_inference_steps'], 
                       strength=config['strength'], 
                       guidance_scale=config['guidance_scale'],
                       generator=generator,
                       width=orig_width,
                       height=orig_height,
                       clip_skip=config['clip_slip'],
                       padding_mask_crop=config['padding_mask_crop'],).images[0]
        
    return result



# Debug
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config_path = "./configs/inpaint_text.yaml"

    # init model
    model, config = init_pipeline(config_path=config_path)

    # load image and mask 
    image_name = "1003001137771826"
    image_path = f"./test_image/{image_name}.png"
    mask_path  = f"./test_image/{image_name}_mask.png"

    # inpaint image and visualize
    final = inpaint(model=model, image_path=image_path, mask_path=mask_path, config=config, device=device)
    final.show()

