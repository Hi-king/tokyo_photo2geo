import torch
import torchvision

def load_model(model: str, classnum: int) -> torchvision.models.ResNet:
    # model setup
    if model == "resnet18":
        model_impl = torchvision.models.resnet18(pretrained=True)
        model_impl.fc = torch.nn.Linear(512, classnum)
    elif model == "resnet50":
        model_impl = torchvision.models.resnet50(pretrained=True)
        model_impl.fc = torch.nn.Linear(2048, classnum)
    else:
        raise Exception()
    return model_impl
