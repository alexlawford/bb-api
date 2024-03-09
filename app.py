from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import requests
from PIL import Image
from io import BytesIO
import base64
from diffusers import StableDiffusionImg2ImgPipeline

def saveBytescale (data):
    headers = {
        'Authorization': 'Bearer public_12a1yrrGGApHW4eVGAfq3RnXk9uv',
        'Content-Type': 'image/png',
    }
    return requests.post('https://api.bytescale.com/v2/accounts/12a1yrr/uploads/binary', headers=headers, data=data)

app = Flask(__name__)
api = Api(app)

def generateImage (prompt):
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
    pipeline.to("cuda")
    return pipeline(prompt).images[0]

def decode_base64_image(image_string):
    image_string = image_string[len("data:image/png;base64,"):]
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    image = Image.open(buffer)
    rgb = image.convert('RGB')
    return rgb

class Predict(Resource):
    def post(self):
        req = request.json
        prediction = generateImage(
            prompt=req.get('prompt'),
            image=decode_base64_image(req.get('img'))
        )
        with BytesIO() as image_binary:
            prediction.save(image_binary, "png")
            image_binary.seek(0)
            result = saveBytescale(image_binary)
        return result.json()

api.add_resource(Predict, "/")

if __name__ == "__main__":
    app.run(debug=True)