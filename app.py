from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import requests
from PIL import Image
from io import BytesIO
import base64
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionLatentUpscalePipeline, AutoPipelineForImage2Image, ControlNetModel
import torch

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

    # Control Nets
    scribble = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16
    )

    openpose = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16
    )

    # Pipelines
    text2image = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    inpaintScribble = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=scribble,
        torch_dtype=torch.float16
    ).to("cuda")

    inpaintOpenpose = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=openpose,
        torch_dtype=torch.float16
    ).to("cuda")

    upscale = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch.float16
    ).to("cuda")

    refine = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    # Memory attention
    # text2image.enable_xformers_memory_efficient_attention()
    # inpaintScribble.enable_xformers_memory_efficient_attention()
    # inpaintOpenpose.enable_xformers_memory_efficient_attention()
    # upscale.enable_xformers_memory_efficient_attention()
    # refine.enable_xformers_memory_efficient_attention()

    return (
        text2image,
        inpaintScribble,
        inpaintOpenpose,
        upscale,
        refine
    )

class Predict(Resource):
    def post(self):
        (text2image, inpaintScribble, inpaintOpenpose, upscale, refine) = load_models()

        req = request.json
        layers=req.get("layers")
        full_prompt = ""
        img = Image.new(mode="RGB", size=(512,512))
        
        for layer in layers:
            prompt = layer["prompt"]
            full_prompt = full_prompt + ' ' + prompt

            if layer["type"] == "background":
                # DEBUG
                print("rendering background with prompt " + prompt)

                img = text2image(
                    prompt=prompt,
                    num_inference_steps=20
                ).images[0]
            elif layer["type"] == "figure":
                # DEBUG
                print("rendering a figure with prompt " + prompt)

                img = inpaintOpenpose(
                    prompt=prompt,
                    image=img,
                    mask_image=decode_base64_image(layer["mask"]),
                    control_image=decode_base64_image(layer["control"]),
                    num_inference_steps=20,
                    controlnet_conditioning_scale=0.75
                ).images[0]
            else:
                # DEBUG
                print("rendering a scribble with prompt " + prompt)

                img = inpaintScribble(
                    prompt=prompt,
                    image=img,
                    mask_image=decode_base64_image(layer["mask"]),
                    control_image=decode_base64_image(layer["control"]),
                    num_inference_steps=20,
                    controlnet_conditioning_scale=0.75
                ).images[0]
        
        upscaled = upscale(
            prompt=full_prompt,
            image=img,
            num_inference_steps=20,
            guidance_scale=6.0
        ).images[0]

        refined = refine(
            prompt=full_prompt,
            image=upscaled,
            num_inference_steps=50,
            guidance_scale=6.0,
            strength=0.25,
        ).images[0]
        
        with BytesIO() as image_binary:
            refined.save(image_binary, "png")
            image_binary.seek(0)
            result = saveBytescale(image_binary)
        return result.json()

api.add_resource(Predict, "/")

if __name__ == "__main__":
    app.run(debug=True)