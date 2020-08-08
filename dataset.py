from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, root, input_data, aug):
        self.file_data = input_data['FileID'].values
        self.label_data = input_data['SpeciesID'].values if 'SpeciesID' in input_data.columns else None
        self.aug = aug

        self.img_data = [Image.open(root + i + '.jpg') for i in self.file_data]

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img = self.img_data[index]

        if self.aug is not None:
            img = self.aug(img)

        if self.label_data is not None:
            return img, self.file_data[index], self.label_data[index]
        else:
            return img, self.file_data[index]

