# Yolov5目标检测模型

## YOLOv5在海洋生物检测领域的应用

## 环境配置
终端运行
```shell
pip install -r requirements.txt
```

## 从零开始训练海洋生物检测模型
终端运行命令，进行预训练
```shell
python train.py --img 640 --batch 32 --epochs 100 --data data/data.yaml  --name yolov5_pretrain --cache
```

## 预训练结果
> Precision-Confidence
<img src='./imgs/P_curve.png' width='50%'>

> Precision-Recall
<img src='./imgs/PR_curve.png' width='50%'>

> Recall-Confidence
<img src='./imgs/R_curve.png' width='50%'>

> 混淆矩阵
<img src='./imgs/confusion_matrix.png' width='50%'>

## 参考
YOLOv5[https://github.com/ultralytics/yolov5]