from flask import Flask, request
from flask_restful import Resource, Api
import requests
from PIL import Image
from io import BytesIO
import base64
from diffusers import AutoPipelineForText2Image, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
import torch
import os

torch.backends.cuda.matmul.allow_tf32 = True

def saveBytescale (data):
    headers = {
        'Authorization': 'Bearer public_12a1yrrGGApHW4eVGAfq3RnXk9uv',
        'Content-Type': 'image/png',
    }
    return requests.post('https://api.bytescale.com/v2/accounts/12a1yrr/uploads/binary', headers=headers, data=data)

app = Flask(__name__)
api = Api(app)

def decode_base64_image(image_string):
    image_string = image_string[len("data:image/png;base64,"):]
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    image = Image.open(buffer)
    rgb = image.convert('RGB')
    return rgb

def load_models():

   # TencentARC/t2i-adapter-openpose-sdxl-1.0
   # TencentARC/t2i-adapter-lineart-sdxl-1.0

    # Control Nets
    openpose = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        torch_dtype=torch.float16
    )

    # Pipelines
    text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
   #    local_files_only=True
    ).to("cuda")

    inpainting = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        controlnet=openpose,
 #      local_files_only=True
    ).to("cuda")

    text2image.enable_xformers_memory_efficient_attention()
    inpainting.enable_xformers_memory_efficient_attention()

    return (text2image, inpainting)

class Predict(Resource):
    def post(self):

        req = request.json
        layers=req.get("layers")

        (text2image, inpainting) = load_models()

        background = text2image(
            prompt="a jungle",
            num_inference_steps=30
        ).images[0]
        
        smallBg = background.resize((512,512))

        refined = inpainting(
            prompt="an explorer in a jungle",
            image=smallBg,
            num_inference_steps=30,
            mask_image=decode_base64_image(layers[1]["mask"]),
            control_image=decode_base64_image(layers[1]["control"]),
            controlnet_conditioning_scale=0.75,
            strength=0.99
        ).images[0]

        with BytesIO() as image_binary:
            refined.save(image_binary, "png")
            image_binary.seek(0)
            result = saveBytescale(image_binary)

        return result.json()

api.add_resource(Predict, "/")

if __name__ == "__main__":
    app.run(debug=True)