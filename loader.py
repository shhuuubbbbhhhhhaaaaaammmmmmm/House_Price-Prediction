import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class customdata(Dataset):
    def __init__(self, root, train=True, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        image1 = "images/"
        self.image1 = image1
        self.image1_folder = sorted([os.path.join(self.root + image1, x) for x in os.listdir(self.root + image1)])

    def __len__(self):
        return len(self.image1_folder)

    def read_files(self, filename):
        P = sorted(os.listdir(filename))
        P = [i for i in P if i[-1] == 'g']
        p1 = [i.split('_') for i in P]
        s6 = []
        lines3 = []
        for i in range(len(P)):
            num = p1[i][-1]
            num1 = num[0:-4]
            s6.append(float(num1))
            lines3.append(P[i])
        return lines3, np.array(s6).astype(np.float32)

    def __getitem__(self, index):
        images_name, s6 = self.read_files(self.root + self.image1)
        img1 = cv2.imread(self.root + self.image1 + images_name[index], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (256, 256))
        img1 = np.expand_dims(img1, axis=2)  # Add channel dimension
        img1 = np.array(img1)
        s_6 = s6[index]
        if self.transforms is not None:
            img1 = self.transforms(img1)
        return img1, s_6

mt = transforms.Compose([
    transforms.ToTensor(),
])

dataset = customdata(root="Data/train/", train=True, transforms=mt)
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

for i, j in loader:
    print(i.shape, j.shape)
