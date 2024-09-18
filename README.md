# README

## Info
方案说明：基于unetplusplus，使用efficientnet-b6作为backbone，使用了颜色变换、色彩抖动、噪声、缩放、翻转、旋转等数据增强。BCE损失函数。推理时使用了TTA。

## Env

* 显存要求：单卡显存>38G （为了尽可能减少复现误差，推荐使用我们使用的A800单卡）
* 依赖库：
	* torch==2.0.1
	* torchvision==0.15.2



## Train
1.确保数据、代码路径。本项目和官方数据集在同一目录下：

```
/xxx/
└──code_submit
└──COSAS24-TrainingSet
    ├── task1
    └── task2
```

2.训练：`python train.py`，开始训练时需要联网下载efficientnet-b6的ImageNet的预训练权重。

3.训练完模型保存在output下，我们得到的模型为 `output/unetplusplus_efficientnet-b6_e98_0.95522.pth`,然后拷贝到cosas-algorithm-submition下面,然后就可以编译镜像、提交。


## Build

Clone the repository and build the Docker image with the following commands:

```bash
git clone https://github.com/VIBE-Lab/cosas-algorithm-submition.git
cd cosas-algorithm-submition
docker build -t cosas .
```

## Local Test
Assuming your local test dataset for task2 is in /cosas/task2/input/domain1, the folder structure should be as follows:
```
/cosas/task2/input/domain1
└──images/adenocarcinoma-image
    ├── image1.mha
    ├── image2.mha
    ...
    └── imagen.mha
```

Run the following command to test the algorithm locally:
```
sudo docker run --gpus all --volume /cosas/task2/input/domain1:/input --volume /cosas/task2/output:/output cosas
```

In the output image, regions with pixel values of 0 represent negative areas, while other regions indicate tumor areas. The output filename of the image will be in grayscale, .mha format, and the same size as the input image. The output folder structure will be as follows:
```
/cosas/task2/output
└──images/adenocarcinoma-mask
    ├── image1.mha
    ├── image2.mha
    ...
    └── imagen.mha
```


