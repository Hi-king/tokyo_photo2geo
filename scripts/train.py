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


def train_model(model, criterion, optimizer, dataloaders, data_size, scheduler=None, num_epochs=25):
    #:bool値を返す。
    use_gpu = torch.cuda.is_available()
    #始まりの時間
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #途中経過保存用に、リストを持った辞書を作ります。
    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    for epoch in range(num_epochs):
        if (epoch + 1) % 5 == 0:  #５回に１回エポックを表示します。
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # それぞれのエポックで、train, valを実行します。
        # 辞書に入れた威力がここで発揮され、trainもvalも１回で書く事ができます。
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 学習モード。dropoutなどを行う。
            else:
                model.eval()  # 推論モード。dropoutなどを行わない。

            running_loss = 0.0
            running_corrects = 0
            data_loader = dataloaders[phase]
            data_size = data_size[phase]

            for data in tqdm.tqdm(data_loader):
                inputs, labels = data  #ImageFolderで作成したデータは、
                #データをラベルを持ってくれます。

                #GPUを使わない場合不要
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                #~~~~~~~~~~~~~~forward~~~~~~~~~~~~~~~
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                #torch.maxは実際の値とインデクスを返します。
                #torch.max((0.8, 0.1),1)=> (0.8, 0)
                #引数の1は行方向、列方向、どちらの最大値を返すか、です。
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics #GPUなしの場合item()不要
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                # (preds == labels)は[True, True, False]などをかえしますが、
                # pythonのTrue, Falseはそれぞれ1, 0に対応しているので、
                # sumで合計する事ができます。

            # サンプル数で割って平均を求めます。
            # 辞書にサンプル数を入れたのが生きてきます。
            epoch_loss = running_loss / data_size
            #GPUなしの場合item()不要
            epoch_acc = running_corrects.item() / data_size

            #リストに途中経過を格納
            print(f'[{phase}{len(loss_dict[phase])}] loss {epoch_loss}, acc {epoch_acc}')
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # 精度が改善したらモデルを保存する
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # deepcopyをしないと、model.state_dict()の中身の変更に伴い、
            # コピーした（はずの）データも変わってしまいます。
            # copyとdeepcopyの違いはこの記事がわかりやすいです。
            # https://www.headboost.jp/python-copy-deepcopy/

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:.4f}'.format(best_acc))

    # 最良のウェイトを読み込んで、返す。
    model.load_state_dict(best_model_wts)
    return model, loss_dict, acc_dict


transform_dict = {
    'train':
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test':
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}


def main(batch_size=10, train_ratio=0.9):
    basedir = pathlib.Path(__file__).parent.parent / 'data'
    dataset = torchvision.datasets.ImageFolder(root=basedir, transform=transform_dict['test'])

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    data_size = {"train": train_size, "val": val_size}
    data_train, data_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)
    dataloaders = {"train": train_loader, "val": val_loader}

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)

    # model = model.cuda()　#GPUなしの場合はこの行はいらない。
    lr = 1e-4
    epoch = 40
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss().cuda() #GPUなしの場合は.cuda()はいらない。

    model_ft, loss, acc = train_model(model, criterion, optim, data_size=data_size, num_epochs=epoch, dataloaders=dataloaders)


if __name__ == '__main__':
    fire.Fire(main)
