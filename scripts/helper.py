import json
import pathlib
import torch
import photo2geo
import PIL.Image
import torchvision
import numpy as np
import pytorch_grad_cam
import pytorch_grad_cam.utils
import pytorch_grad_cam.utils.image

def load_small_image(path):
    i = PIL.Image.open(path).convert("RGB")
    i = torchvision.transforms.Resize((456, 456))(
        torchvision.transforms.CenterCrop(i.width)(i)
    )
    return i

# def pred(path, model, params):
#     i = load_small_image(path)
#     with torch.inference_mode():
#         predict = model(photo2geo.transform.transform_dict["test"](i).unsqueeze(0))
#         return params["classes"][predict.argmax()]

def pred(path, model, params):
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    i = load_small_image(path)
    with torch.inference_mode():
        predict = model(photo2geo.transform.transform_dict["test"](i).unsqueeze(0))[0].numpy()
        scores = softmax(predict)
        return params["classes"][predict.argmax()], {label: score for label, score in zip(params["classes"], scores)}

def load_params(
        result_path: str = (pathlib.Path(__file__).parent.parent
    / "results/class7_resnet50_batch30_lr1e-05_commit6241653_202112070034").as_posix(),

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


def show_cam(path, model, params, target=None):
    def image2tensor(image):
        return photo2geo.transform.transform_dict["test"](image).unsqueeze(0)
    target_layers = [model.layer4[-1]]
    cam = pytorch_grad_cam.GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    input_Image = load_small_image(path)
    if target is not None:
        target_class_index = np.argwhere(np.array(params['classes']) == target).flatten()[0]
        grayscale_cam = cam(
            input_tensor=image2tensor(input_Image),
            targets=[
                pytorch_grad_cam.utils.model_targets.ClassifierOutputTarget(target_class_index) # type: ignore
            ]
        )
    else:
        grayscale_cam = cam(
            input_tensor=image2tensor(input_Image)
        )

    grayscale_cam = grayscale_cam[0, :]
    cam_image = pytorch_grad_cam.utils.image.show_cam_on_image(
        np.asarray(input_Image.resize((256, 256))).astype(np.float32) / 255, 
        grayscale_cam, use_rgb=True
    )
    return PIL.Image.fromarray(cam_image)
