# Computer-Vision-Course-PJ2

## Classification 
We explored three kinds of augmentation methods. The accuracy of each model on cifar 100 is shown in the following table. **Passwords are all 1111**.

#### ResNet18
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Status</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">ResNet18</td>
<td align="center">Untrained</td>
<td align="center">21.68%</td>
<td align="center"><a href="https://pan.baidu.com/s/1K7y50PvgMlVAqfYFzoLskQ">
checkpoint</a></td>
</tr>

 <tr><td align="left">ResNet18</td>
<td align="center">Pretrained</td>
<td align="center">75.87%</td>
<td align="center"><a href="https://pan.baidu.com/s/1AzAlwf00U2d-vOELgnZ_oQ">checkpoint</a></td>
</tr>

</tbody></table>

#### VGG16
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Status</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">VGG16</td>
<td align="center">Untrained</td>
<td align="center">9.44%</td>
<td align="center"><a href="https://pan.baidu.com/s/1msMpEcS_Qu3u8hoigz5MpA">
checkpoint</a></td>
</tr>

 <tr><td align="left">VGG16</td>
<td align="center">Pretrained</td>
<td align="center">72.8%</td>
<td align="center"><a href="https://pan.baidu.com/s/1sfcIRokhWM4fmY0QNLu0Pg">checkpoint</a></td>
</tr>

</tbody></table>

#### GoogleNet
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Status</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">GoogleNet</td>
<td align="center">Untrained</td>
<td align="center">45.65%</td>
<td align="center"><a href="https://pan.baidu.com/s/1tJaTZr2DwVX33xzixjpypg">
checkpoint</a></td>
</tr>

 <tr><td align="left">GoogleNet</td>
<td align="center">Pretrained</td>
<td align="center">76.27%</td>
<td align="center"><a href="https://pan.baidu.com/s/1hvTi4GtDkt38JRVU8F_pVQ">checkpoint</a></td>
</tr>

</tbody></table>

### Training and Test
```
CUDA_VISIBLE_DEVICES=$GPU_ID python resnet_train.py --model resnet18 --mode 0 --epoch 50 --batchsize 256
CUDA_VISIBLE_DEVICES=$GPU_ID python googlenet_train.py --model googlenet --mode 0 --epoch 50 --batchsize 32
CUDA_VISIBLE_DEVICES=$GPU_ID python vgg16_train.py --model vgg16 --mode 0 --epoch 50 --batchsize 256
```

- `MODE=0`: Untrained
- `MODE=1`: Pretrained


### Visualization
```
pip install tensorboard
tensorboard --logdir=results/${model}/logdir
```

## Detection 

The config files for faster R-CNN and YOLOv3 are shown in the following table.
|   Model         | config name  | Download |
|:---------------:|:-----------:|:---------:|
| Faster R-CNN  | [Faster R-RNN](https://github.com/OriginSound/Computer-Vision-Course-PJ2/blob/main/detection/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py) | [checkpoint](https://pan.baidu.com/s/1e0oLWeVjqotOj84XMD07Lg)  |
|YOLOv3 | [YOLOv3](https://github.com/OriginSound/Computer-Vision-Course-PJ2/blob/main/detection/configs/pascal_voc/yolov3_d53_mstrain-608_100e_voc0712.py) | [checkpoint](https://pan.baidu.com/s/1zYT5zKR4aSKS7IM6vk2_QA)  |

### Training
please first turn to the mmdetection and then run 
```
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
```

### Test
To test our trained model, please run
```
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval mAP
```
