import json
from PIL import Image
import torchvision
from torchvision.transforms import transforms
from collections import defaultdict
import cv2 as cv
from numpy.random import choice, shuffle

img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1,))])

digits = defaultdict(list)
img_dataset = torchvision.datasets.cifar.CIFAR10(root="./datasets/", train=False, download=True)
for idx, (img, label) in enumerate(img_dataset):
    digits[label].append(idx)


file = "./data/expr_test.json"
img_dir = "./data/Handwritten_Math_Symbols/"

X, Y, Z = [], [], []
paths = defaultdict(list)
get_pseudo_label = True

with open(file) as f:
    data = json.load(f)
    for idx in range(len(data)):
        imgs = []
        imgs_pseudo_label = []
        for img_path in data[idx]["img_paths"]:
            img = Image.open(img_dir + img_path).convert("L")
            img = img_transform(img)
            imgs.append(img)
            label = img_path.split("/")[0]
            if label in "123456789":
                paths[int(label)].append(img_path)
            else:
                img = cv.imread(img_dir + img_path, cv.IMREAD_GRAYSCALE)
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                img = cv.resize(img, (32, 32))
                cv.imwrite(img_dir + img_path, img)
            if get_pseudo_label:
                imgs_pseudo_label.append(img_path.split("/")[0])
        # if len(imgs) <= 2:
        #     continue
        X.append(imgs)
        if get_pseudo_label:
            Z.append(imgs_pseudo_label)
        Y.append(data[idx]["res"])

for key in list(paths.keys()):
    print(key)
    key = int(key)
    idxs = choice(len(digits[key]), len(paths[key]), replace=False)
    for idx, img_path in zip(idxs, paths[key]):
        img, label = img_dataset[digits[key][idx]]
        img.save(img_dir + img_path)
