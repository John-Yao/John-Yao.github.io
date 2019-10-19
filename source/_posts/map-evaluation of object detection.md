---
title: Anlysis and implementation of mAP-evaluation of object detection
date: 2019-10-13 14:40:27
tags: Ojbect_Detection
categories: Ojbect_Detection
---
## Abstract

​	 目标检测中衡量识别精度的常用指标是mAP（mean average precision）。多个类别物体检测中，每一个类别都可以根据recall和precision绘制一条曲线，AP就是该曲线下的面积，mAP是多个类别AP的平均值 。

## Information

Code: 

-  https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py 
-  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py 

Reference:

- 目标检测中的mAP是什么含义？ - Wentao的回答 - 知乎
  https://www.zhihu.com/question/53405779/answer/419532990
- ROC，AUC，PR，AP介绍及python绘制 https://www.cnblogs.com/zf-blog/p/6734686.html 
- [The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Development Kit](https://link.zhihu.com/?target=http%3A//host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html%23SECTION00044000000000000000)
-  用mAP衡量目标检测的性能是否科学？ - Angzz的回答 - 知乎 https://www.zhihu.com/question/337856533/answer/769453722 

## Approach Description

### mAP定义集相关概念

- mAP: mean Average Precision, 即各类别AP的平均值
- AP: PR曲线下的面积
- PR曲线: Precision-Recall曲线
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- TP: IoU>0.5的检测框数量（同一Ground Truth只计算一次）
- FP: IoU<=0.5的检测框，或者是检测到同一个GT的**多余**检测框的数量 （ps:多余的意思是指对一个gt预测出多个框的iou大于阈值，只考虑最大iou的 检测框）
- FN: 没有检测到的GT的数量

#### mAP的计算方法

​	要计算mAP必须先绘出各类别PR曲线，通过对PR曲线的面积统计得出AP。



​		**如何采样PR曲线**，VOC，COCO都采用过下面几种不同方法。

- 在VOC2010以前，只需要选取当Recall >= 0, 0.1, 0.2, ..., 1共11个点时的**Precision最大值**，然后AP就是这11个Precision的平均值。

- 在VOC2010及以后， 假设设这一张图片M个正例，那么我们会得到M个recall值（1/M, 2/M, …, M/M）,对于每个recall值r，我们可以计算出对应（r’ > r）的最大precision，然后对这M个precision值取平均即得到最后的AP值。 

- COCO(pycocotools) 选取当Recall >= 0, 0.01, 0.02, ..., 1共101个点时的Precision最大值，然后AP就是这101个Precision的平均值。 

  

  **如何计算Precision, Recall**:

  根据Precision的定义需要确定TP、FP、FN的值，那么如何确定这些值并且和设定recall阈值对应起来呢？

  1. 确定iou_thres(0.5)，将pred_boxes按score排序，**相同imageID**下，将pred_boxes 和gt_boxes进行匹配：

     - (method1) 对每个gt_box 计算与未被匹配的pred_boxes的iou，取最大iou的pred_box，若iou>iou_thres则该pred_box置为TP,否则为FP

     - (method2，voc)对每个pred_box 计算与gt_boxes的iou,取最大iou的gt_box，若iou>iou_thres且该gt_box未被匹配则将该pred_box置为TP，否则为FP

       经过上述方法可以得到长度为len(pred_boxes)的tps/fps数组，可以用1代表对应位置为TP/FP

  2. 对tps，fps分别进行累加，得到累加数组，第i个元素代表在第i个pred_boxes的score划分正负样本时，TP和FP的值（ps：只考虑了预测为正样本部分），于是可以得到不同score下的Precision，Recall的值，分别记为precisions,recalls。Recall的分母为len(gt_boxes)

  3. **平滑处理**：

     1. 对precisions的每个元素，取当前位置后面所有precision的最大值

  4. 根据确定的Recall threshold找到其在Recall的位置，并确定其precision值，相邻的recall相减并与precision值相乘，所有乘积之和就是ap

note: 

​	precision的最大值体现在如果只是简单取recall阈值下的precision那么绘制出来的PR曲线并不是单调递减的。	举例来说tps为 [0,1,0,1,1]，则累加后为：[0,1,1,2,3]，precisions为：[0,1/2,1/3/，2/4，3/5]，可以看到precision可能是波动的，PR曲线并不是单调递减的，通过平滑处理，可以保证PR曲线的单调性，有利于观察PR曲线选择合适的score阈值。

#### 参考第2个链接的例子理解上述过程：

 假设从测试集中共检测出20个例子，而测试集中共有6个正例，则PR表如下： 

<img src="map-evaluation of object detection/PR_table.png" style="zoom:100%;" />

PR波形图如下

<img src="map-evaluation of object detection/PR_curve.png" style="zoom:80%;" />

## Code

pseudo code：给出一张图片，一个类别时的AP伪代码

```
import numpy as np
# Box = Tuple[int, int, int, int]
# [left, top, right, bottom]

def IoU(box1:Box, box2:Box) -> float:
    pass
    
def AP(boxes:List[Box], gt_boxes:List[Box], thres:float) -> float:
    N = len(boxes)
    M = len(gt_boxes)
    pred_match = [0 for i in range(N)]
    for i in range(M):
        max_id = 0
        max_iou = 0
        for j in range(N):
            if pred_match[j]==0:
               iou = IoU(gt_boxes[i],boxes[j]) 
               if iou>thres:
                   if iou>max_iou:
                       max_id = j
                       max_iou = iou
       pred_match[max_id] = i+1
   precisions = []
   # recalls = []
   tp = 0
   for i in range(N):
       if pred_match[i]!=0:
           tp += (pred_match[i]!=0)
           fp = N-tp
           recall = tp/M
           precision = tp/(i+1)
           # recalls.append(recall)
           precisions.append(precision)
    # smooth
    precisions = [0.0]+precisions+[0.0]
    for i in range(len(precisions)-1,0,-1):
        precisions[i-1] = max(precisions[i-1],precisions[i])

    precisions = precisions[1:-1]
    ap = [precisions[i]*1/M for i in range(len(precisions))]
    ap = sum(ap)
    
    return ap
```



## Extension:

​	在训练好detection model，可以通过softnms，多尺度测试，flip测试增强等trick提升mAP, 但是基本都是通过提升recall，涨的低Precision的区域，低精度区对应用场景来说没用 。

​	实际应用中可以使用FPPI(False Positve Per Image), MISS Rate(行人检测常用)