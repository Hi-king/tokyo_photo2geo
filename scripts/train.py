import photo2geo
import fire
import pathlib
import torch
import torchvision.datasets
import torchvision.models
import time
import tqdm
import copy
from torchvision import transforms
import datetime
import pandas as pd
import sklearn.metrics
import os


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    data_sizes,
    results_dir: pathlib.Path,
    num_epochs=25,
):
    use_gpu = torch.cuda.is_available()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    for epoch in range(num_epochs):
        if (epoch + 1) % 5 == 0:  # ５回に１回エポックを表示します。
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 学習モード。dropoutなどを行う。
            else:
                model.eval()  # 推論モード。dropoutなどを行わない。

            running_loss = 0.0
            running_corrects = 0
            epoch_predicts = []
            epoch_labels = []
            data_loader = dataloaders[phase]
            data_size = data_sizes[phase]

            for data in tqdm.tqdm(data_loader):
                inputs, labels = data

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                epoch_predicts += preds.tolist()
                epoch_labels += labels.tolist()

            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects.item() / data_size

            # Result
            acc = sklearn.metrics.accuracy_score(epoch_labels, epoch_predicts)
            print(
                f"[{phase}{len(loss_dict[phase])}] loss {epoch_loss}, acc {epoch_acc}, acc {acc}"
            )
            confusion_matrix = pd.DataFrame(
                sklearn.metrics.confusion_matrix(epoch_labels, epoch_predicts),
                index=data_loader.dataset.dataset.classes,
                columns=data_loader.dataset.dataset.classes,
            )
            print(confusion_matrix)

            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            # save
            torch.save(
                copy.deepcopy(model).to("cpu").state_dict(),
                results_dir / f"model_epoch{epoch}.pth",
            )
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val acc: {:.4f}".format(best_acc))

    # 最良のウェイトを読み込んで、返す。
    model.load_state_dict(best_model_wts)
    return model, loss_dict, acc_dict


transform_dict = {
    "train": transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


def main(
    batch_size=10,
    train_ratio=0.9,
    lr=1e-4,
    weight_decay=1e-5,
    epoch=40,
    model="resnet18",
):
    use_gpu = torch.cuda.is_available()
    basedir = pathlib.Path(__file__).parent.parent / "data"
    dataset: torchvision.datasets.ImageFolder = torchvision.datasets.ImageFolder(
        root=basedir
    )
    classnum = len(dataset.classes)
    print(dataset.classes)
    resultdir = (
        pathlib.Path(__file__).parent.parent
        / "results"
        / f'class{classnum}_batch{batch_size}_lr{lr}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    )
    resultdir.mkdir(parents=True, exist_ok=True)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    data_sizes = {"train": train_size, "val": val_size}
    data_train, data_val = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    data_train.dataset.transform = transform_dict["train"]
    data_val.dataset.transform = transform_dict["test"]

    # save data split
    filename_df = pd.DataFrame(dataset.imgs).assign(trainval="None")
    filename_df.loc[data_train.indices, "trainval"] = "train"
    filename_df.loc[data_val.indices, "trainval"] = "val"
    filename_df.to_csv(resultdir / "datasplit.csv")

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
    )
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    # model setup
    if model == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
    else:
        raise Exception()
    model.fc = torch.nn.Linear(512, classnum)

    if use_gpu:
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_gpu:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_ft, loss, acc = train_model(
        model,
        criterion,
        optim,
        data_sizes=data_sizes,
        num_epochs=epoch,
        dataloaders=dataloaders,
        results_dir=resultdir,
    )


if __name__ == "__main__":
    fire.Fire(main)
