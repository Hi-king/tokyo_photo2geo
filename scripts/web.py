import streamlit as st
import torch
import json
import pathlib
import photo2geo
import PIL.Image
import torchvision
import numpy as np
import pytorch_grad_cam

RESULT_PATH = pathlib.Path(__file__).parent.parent / "results/class7_resnet50_batch30_lr1e-05_commit6241653_202112070034"

def load_small_image(path):
    i = PIL.Image.open(path)
    i = torchvision.transforms.Resize((456, 456))(
        torchvision.transforms.CenterCrop(i.width)(i)
    )
    return i


def pred(path, model, params):
    i = load_small_image(path)
    with torch.inference_mode():
        predict = model(photo2geo.transform.transform_dict["test"](i).unsqueeze(0))
        return params["classes"][predict.argmax()]


def load_params(
        result_path: str = pathlib.Path(__file__).parent.parent
    / "results/class7_resnet50_batch30_lr1e-05_commit6241653_202112070034",

):
    params = json.load((pathlib.Path(result_path) / "params.json").open())
    print(params)
    models = list(pathlib.Path(result_path).glob("model*"))
    last_model_path = sorted(models)[-1]

    model = photo2geo.model.load_model(
        model=params["model"], classnum=len(params["classes"])
    )
    model.load_state_dict(torch.load(last_model_path))
    model = model.eval()
    return model, params

def predict(
    path: str,
    model,
    params
):
    predicted = pred(path, model, params)
    return predicted

def show_cam(path, model):
    def image2tensor(image):
        return photo2geo.transform.transform_dict["test"](image).unsqueeze(0)
    target_layers = [model.layer4[-1]]
    cam = pytorch_grad_cam.GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    input_Image = load_small_image(path)
    grayscale_cam = cam(
        input_tensor=image2tensor(input_Image))

    grayscale_cam = grayscale_cam[0, :]
    cam_image = pytorch_grad_cam.utils.image.show_cam_on_image(
        np.asarray(input_Image.resize((256, 256))).astype(np.float32) / 255, 
        grayscale_cam, use_rgb=True
    )
    return PIL.Image.fromarray(cam_image)

def myapp():
    st.write("hello")

    file_up = st.file_uploader("Upload an image", type=['png','jpg'])

    if file_up is not None:
        model, params = load_params(RESULT_PATH)
        image = PIL.Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        predicted = predict(file_up, model, params)
        st.text(f'予測: {predicted}')

        cam_image = show_cam(file_up, model)
        st.image(cam_image, caption='GradCAM', use_column_width=True)

if __name__ == "__main__":
    myapp()
