import torch
import json
import pathlib
import photo2geo
import PIL.Image
import torchvision
import fire


def load_small_image(path):
    i = PIL.Image.open(path).convert("RGB")
    i = torchvision.transforms.Resize((456, 456))(
        torchvision.transforms.CenterCrop(i.width)(i)
    )
    return i


def pred(path, model, params):
    i = load_small_image(path)
    with torch.inference_mode():
        predict = model(photo2geo.transform.transform_dict["test"](i).unsqueeze(0))
        return params["classes"][predict.argmax()]


def predict(
    path: str,
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
    print(f'{path} -> {pred(path, model, params)}')


if __name__ == "__main__":
    fire.Fire(predict)
