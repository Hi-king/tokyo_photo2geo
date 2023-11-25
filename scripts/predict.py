import torch
import json
import pathlib
import photo2geo
import fire
import helper

def predict(
    path: str,
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
    print(f'{path} -> {helper.pred(path, model, params)[0]}')


if __name__ == "__main__":
    fire.Fire(predict)
