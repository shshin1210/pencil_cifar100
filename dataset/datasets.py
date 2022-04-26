from operator import index
from cv2 import transform
from matplotlib import transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class C100Dataset(Dataset):

    classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
               'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
            'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
            'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    def __init__(self, train = True, val = False, transform = None ):
        super().__init__()
        self.csv_dir = './dataset/data/cifar100_nl.csv' if train else './dataset/data/cifar100_nl_test.csv'
        self.transform = transform
        dataset = pd.read_csv(self.csv_dir, names = ['filename', 'classname']) # 59998 / 9999
        self.train = train
        self.val = val

        # train set
        if train == True and val == False:
            # trainset '/train/'
            dataset = dataset[:39999]
        
        # val set
        if train == True and val == True:
            dataset = dataset[40000:49999] 

        # img paths
        self.img_paths = dataset['filename']
        # img_labels
        labels = dataset['classname']

        self.data = []
        if (train == True and val == False) or (train == False):
            for i in range(len(dataset)): #train 39999, test 9999
                img_path = './dataset/' + self.img_paths[i]
                image = read_image(img_path)
                self.data.append(image)
        else:
            for i in range(40000,49999): # val 9999
                img_path = './dataset/' + self.img_paths[i]
                image = read_image(img_path)
                self.data.append(image)


        # reshape & transpose to h/w/c
        self.data = np.concatenate(self.data)
        self.data = self.data.reshape(len(dataset), 3,32,32)
        self.data = self.data.transpose((0,2,3,1))

        self.img_labels = []
        for label in labels:
            self.img_labels.append(C100Dataset.classes.index(label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        image, label = self.data[index], self.img_labels[index]
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        if self.train == True and self.val == False:
            return [image, label, index]
        else:
            return [image, label]
    
if __name__ == "__main__":
    # Transform
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding =4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = C100Dataset(train=True, val = False, transform=transform_train)
    valset = C100Dataset(train=True, val = True, transform = transform_test)
    testset = C100Dataset(train=False, transform=transform_test)

    # data loader
    train_loader = DataLoader(trainset, batch_size=4, shuffle = True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

    # Check subtrain_lodaer 
    for images, labels, index in train_loader:
        print(len(labels))
        print("Batch of images has shape: ",images.size())
        print("Batch of labels has shape: ", labels.size())

    for images, labels in val_loader:
        print("Batch of images has shape: ",images.size())
        print("Batch of labels has shape: ", labels.size())
    
    for images, labels in test_loader:
        print("Batch of images has shape: ",images.size())
        print("Batch of labels has shape: ", labels.size())