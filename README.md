**Setup enviroment:**
```
conda create -n <enviroment_name> python=3.9 #or use -p with a specific path
conda activate <enviroment_nam> 
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #install torch with cuda independently 
```
**Download pretrained model:** <br>
 &nbsp;&nbsp;&nbsp;&nbsp;Download checkpoint from civitai folder. <br>
 &nbsp;&nbsp;&nbsp;&nbsp;Currently only support stable diffusion based model with safetensors extension. <br>
 &nbsp;&nbsp;&nbsp;&nbsp;Run convert_original_stable_diffusion_to_diffusers.py to convert diffusion to diffuser. <br>
 &nbsp;&nbsp;&nbsp;&nbsp;Example: 
 ```
  python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path <downloaded_model_path>.safetensors --dump_path <save_path>/ --from_safetensors --to_safetensors
 ```
 **Usage** <br> 
 &nbsp;&nbsp;&nbsp;&nbsp;Example in the debug section of atirk.py. <br>
