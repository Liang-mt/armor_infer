# 装甲板识别-----onnxruntime推理

本代码为沈阳航空航天大学2022年开源模型的onnxruntime推理

参考开源

[RangerOnMars/TUP-NN-Train: A YOLOX based training program for armor detect (github.com)](https://github.com/RangerOnMars/TUP-NN-Train#tup战队2022神经网络训练程序)

[tup-robomaster/TUP-InfantryVision-2022: 沈阳航空航天大学T-UP战队2022赛季步兵视觉识别程序 (github.com)](https://github.com/tup-robomaster/TUP-InfantryVision-2022)

video文件夹内有测试的buff和装甲板视频，可供自己模型测试



video视频文件请访问网盘链接下载

链接：https://pan.baidu.com/s/1QykXf3QvKQdGDIeRvCxdzw?pwd=0000 
提取码：0000 



安装所需依赖

```
pip install onnxruntime
pip install numpy
pip install opencv-python
pip install torch
pip install torchvision
#这样安装pytorch为cpu版本，gpu版本可从pytorch进行安装
```

运行代码  

```
python main.py
#代码默认cpu推理，更改gpu推理可参考网上相关资料进行修改，因本人电脑没有显卡，没进行相关测试
```



补充：

1.可在datasets.py里面对类别进行修改

```
#沈航格式
COCO_CLASSES = (
"BG",
"B1",
"B2",
"B3",
"B4",
"B5",
"BO",
"BB",
"RG",
"R1",
"R2",
"R3",
"R4",
"R5",
"RO",
"RB",
"NG",
"N1",
"N2",
"N3",
"N4",
"N5",
"NO",
"NB",
"PG",
"P1",
"P2",
"P3",
"P4",
"P5",
"PO",
"PB"
)
```

2.可根据自己需求在main.py里面做以下的修改

```python
#视频路径
video_path = "./video/3.mp4"
#模型路径
#onnx_model_path = "./model/500.onnx"
onnx_model_path = "./model/opt-0625-001.onnx"
#根据自己模型的不同可对关键点数量，颜色数量，类别数量进行相对应的修改
predictor = Predictor(onnx_model_path, num_apex = 4, num_class = 8,num_color = 4)
```
