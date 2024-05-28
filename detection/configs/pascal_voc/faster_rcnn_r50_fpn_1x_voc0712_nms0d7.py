_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)),
             test_cfg=dict(
                 rpn=dict(
                     nms_pre=1000,
                     max_per_img=1000,
                     nms=dict(type='nms', iou_threshold=0.7),
                     min_bbox_size=0),
                 rcnn=dict(
                     score_thr=0.15,
                     nms=dict(type='nms', iou_threshold=0.7),
                     max_per_img=100)
             ))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
)