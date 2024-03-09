from flask import Flask, request
from flask_restful import Resource, Api
import requests
from PIL import Image
from io import BytesIO
import base64
from diffusers import StableDiffusionXLPipeline
import torch
import os


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
    torch.backends.cuda.matmul.allow_tf32 = True

    # Pipelines
    text2image = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    text2image.unet = torch.compile(text2image.unet, mode="reduce-overhead", fullgraph=True)

    return text2image

class Predict(Resource):
    def post(self):

        text2image = load_models()

        refined = text2image(
                prompt="A jungle",
                num_inference_steps=30
            ).images[0]
  
        with BytesIO() as image_binary:
            refined.save(image_binary, "png")
            image_binary.seek(0)
            result = saveBytescale(image_binary)
        return result.json()

api.add_resource(Predict, "/")

if __name__ == "__main__":
    app.run(debug=True)