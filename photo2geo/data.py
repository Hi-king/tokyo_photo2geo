class DatasetWithIndex:
    '''
    :see: https://zenn.dev/hidetoshi/articles/20210619_pytorch-dataset-with-index
    '''
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes
