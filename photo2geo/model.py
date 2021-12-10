import torch
import torchvision

def load_model(model: str, classnum: int):
    # model setup
    if model == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, classnum)
    elif model == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, classnum)
    else:
        raise Exception()
    return model
