from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import requests
from PIL import Image
from io import BytesIO
from diffusers import DiffusionPipeline

def saveBytescale (data):
    headers = {
        'Authorization': 'Bearer public_12a1yrrGGApHW4eVGAfq3RnXk9uv',
        'Content-Type': 'image/png',
    }
    return requests.post('https://api.bytescale.com/v2/accounts/12a1yrr/uploads/binary', headers=headers, data=data)

app = Flask(__name__)
api = Api(app)

def generateImage (prompt):
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
    pipeline.to("cuda")
    return pipeline(prompt).images[0]

class Predict(Resource):
    def post(self):
        req = request.json
        prediction = generateImage(req.get('prompt'))
        with BytesIO() as image_binary:
            prediction.save(image_binary, "png")
            image_binary.seek(0)
            result = saveBytescale(image_binary)
        return result.json()

api.add_resource(Predict, "/")

if __name__ == "__main__":
    app.run(debug=True)