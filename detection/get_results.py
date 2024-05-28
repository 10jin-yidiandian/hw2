from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import pdb

# config_file = 'configs/pascal_voc/fcos_4x4.py'
# checkpoint_file = 'work_dirs/fcos_4x4/epoch_12.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = 'vis/fcos_vis_result'  # or img = mmcv.imread(img), which will only load it once
# # os.makedirs(img_path, exist_ok=True)
# filenames = os.listdir(img_path)
#
# for filename in filenames:
#     img_filename = os.path.join(img_path, filename)
#     result = inference_detector(model, img_filename)
#     model.show_result(img_filename, result, out_file=f'vis/fcos_vis_result/result_{filename}')


# config_file = 'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py'
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_4.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = 'vis/FasterRcnn_vis_result'  # or img = mmcv.imread(img), which will only load it once
# # os.makedirs(img_path, exist_ok=True)
# filenames = os.listdir(img_path)
#
# for filename in filenames:
#     img_filename = os.path.join(img_path, filename)
#     result = inference_detector(model, img_filename)
#     model.show_result(img_filename, result, out_file=f'vis/FasterRcnn_vis_result/result_{filename}')


# config_file = 'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_nms0d3.py'
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_4.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = 'vis/FasterRcnn_nms0d3_vis_result'  # or img = mmcv.imread(img), which will only load it once
# # os.makedirs(img_path, exist_ok=True)
# filenames = os.listdir(img_path)
#
# for filename in filenames:
#     img_filename = os.path.join(img_path, filename)
#     result = inference_detector(model, img_filename)
#     model.show_result(img_filename, result, out_file=f'vis/FasterRcnn_nms0d3_vis_result/result_{filename}')


config_file = 'configs/pascal_voc/yolov3_d53_mstrain-608_100e_voc0712.py'
checkpoint_file = 'ckpts/yolov3_e12.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img_path = 'vis/yolov3_vis_result'  # or img = mmcv.imread(img), which will only load it once
# os.makedirs(img_path, exist_ok=True)
filenames = os.listdir(img_path)

for filename in filenames:
    img_filename = os.path.join(img_path, filename)
    result = inference_detector(model, img_filename)
    model.show_result(img_filename, result, out_file=f'vis/yolov3_vis_result/result_{filename}')