# Street-View-House-Numbers-Detection
The proposed challenge is a street view house numbers detection, which contains two parts:
1. Do bounding box regression to find top, left, width and height of bounding boxes which contain digits in a given image
2. classify the digits of bounding boxes into 10 classes (0-9)

The giving SVHN dataset contains 33402 images for training and 13068 images for testing. This project uses the YOLOv5 pre-trained model to fix this challenge.


### Hardware
- Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
- NVIDIA GeForce RTX 2080 Ti

### Environment
- Microsoft win10
- Python 3.7.3
- Pytorch 1.7.0
- CUDA 10.2

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#install-packages)
2. [Data Preparation](#data-preparation)
3. [Set Configuration](#set-configuration)
4. [Download Pretrained Model](#download-pretrained-model)
5. [Training](#training)
6. [Testing](#testing)
7. [Reference](#reference)



### Install Packages
- install pytorch from https://pytorch.org/get-started/locally/
- install openCV
```
sudo apt-get install python-opencv
```
- install dependencies
```
pip install -r requirements.txt
```

### Data Preparation
Download the given dataset from [Google Drive](https://drive.google.com/drive/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl) or [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/).
```
data / svhn
  +- train
  |	+- xxx.jpg
  |	+- digitStruct.mat
  +- test
  |	+- yyy.jpg
  +- mat_to_yolo.py
  +- shvn.yaml
```
And run command `python mat_to_yolo.py` to create labels for yolo and reorganize the train data structure as below:
```
- train/
├── 1.png
├── 1.txt
├── 2.png
├── 2.txt
│     .
│     .
│     .
├── 33402.png
└── 33402.txt
```

### Set Configuration
- create `svhn.yaml` in `./data`
```
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: data/svhn/train  # 33402 images
val: data/svhn/valid  # 3000 images

# number of classes
nc: 10

# class names
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
```

### Download Pretrained Model
- https://github.com/ultralytics/yolov5/releases

### Training
- train model with pretrained model
```
python train.py --img 320 --batch 16 --epochs 50 --data svhn.yaml --weights yolov5m.pt
```

- Using the following script to get more information
```
python train.py --help
```

### Testing
- detect test data
```
python detect.py --source data/svhn/test/ --weights runs/train/exp18/weights/best.pt --conf 0.25 --save-txt --save-conf
```

- Using the following script to get more information
```
python detect.py --help
```

- Make Submission: output json format
```
python combine.py
```
```
[{
  "bbox": [[top, left, buttom, right]],
  "score": [confidence],
  "label": [predict_label]
 }, 
 {
  "bbox": [[7, 112, 28, 121], [9, 122, 27, 134],
  "score": [0.674805, 0.713867],
  "label": [1, 0]
 }
]
```

### Reference
- [h5py - Quick Start Guide](https://docs.h5py.org/en/stable/quick.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)