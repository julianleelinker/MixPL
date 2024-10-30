# dataset settings
dataset_type = 'mmyolo.YOLOv5CocoDataset'
# data_root = 'data/coco/'
# data_root = '/mnt/data-home/julian/coco/'
data_root = '/mnt/data-home/julian/datasets/mobility-s8-s12-s24-COCO/'
class_name = (
    'BICYCLE',
    'BUS',
    'CAR',
    'CLEAN_TRAFFIC_SIGN',
    'CONE',
    'DIRTY_TRAFFIC_SIGN',
    'JERSEY_BARRIER',
    'MOTORCYCLE',
    'PEDESTRIAN',
    'RIDER',
    'ROAD_CRACKS',
    'ROAD_PATCH',
    'ROAD_POTHOLES',
    'TRAFFIC_LIGHT',
    'TRUCK',
    'UNCLEAR_ROAD_MARKING',
)
num_classes = len(class_name) # dataset category number
# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
metainfo = dict(classes=class_name, palette=[(20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60)])

color_space = [
    [dict(type='mmdet.ColorTransform')],
    [dict(type='mmdet.AutoContrast')],
    [dict(type='mmdet.Equalize')],
    [dict(type='mmdet.Sharpness')],
    [dict(type='mmdet.Posterize')],
    [dict(type='mmdet.Solarize')],
    [dict(type='mmdet.Color')],
    [dict(type='mmdet.Contrast')],
    [dict(type='mmdet.Brightness')],
]

geometric = [
    [dict(type='mmdet.Rotate')],
    [dict(type='mmdet.ShearX')],
    [dict(type='mmdet.ShearY')],
    [dict(type='mmdet.TranslateX')],
    [dict(type='mmdet.TranslateY')],
]

scale = [(1333, 400), (1333, 1200)]

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.RandomResize', scale=scale, keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.RandAugment', aug_space=color_space, aug_num=1),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        sup=dict(type='mmdet.PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='mmdet.RandomResize', scale=scale, keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='mmdet.RandomResize', scale=scale, keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.RandomOrder',
        transforms=[
            dict(type='mmdet.RandAugment', aug_space=color_space, aug_num=1),
            dict(type='mmdet.RandAugment', aug_space=geometric, aug_num=1),
        ]),
    # dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    # dict(type='LoadEmptyAnnotations'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 8
num_workers = 8
# There are two common semi-supervised learning settings on the coco dataset：
# (1) Divide the train2017 into labeled and unlabeled datasets
# by a fixed percentage, such as 1%, 2%, 5% and 10%.
# The format of labeled_ann_file and unlabeled_ann_file are
# instances_train2017.{fold}@{percent}.json, and
# instances_train2017.{fold}@{percent}-unlabeled.json
# `fold` is used for cross-validation, and `percent` represents
# the proportion of labeled data in the train2017.
# (2) Choose the train2017 as the labeled dataset
# and unlabeled2017 as the unlabeled dataset.
# The labeled_ann_file and unlabeled_ann_file are
# instances_train2017.json and image_info_unlabeled2017.json
# We use this configuration by default.
labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='annotations/instances_train2017.json',
    # data_prefix=dict(img='train2017/'),
    ann_file='train/labels.json',
    data_prefix=dict(img='train/data/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    metainfo=metainfo,
)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='annotations/instances_unlabeled2017.json',
    # data_prefix=dict(img='unlabeled2017/'),
    ann_file='unused/labels.json',
    data_prefix=dict(img='unused/data/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    metainfo=metainfo,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='mmdet.GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[4, 4]),
    dataset=dict(
        type='mmdet.ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset],
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_val2017.json',
        # data_prefix=dict(img='val2017/'),
        ann_file='val/labels.json',
        data_prefix=dict(img='val/data/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + 'val/labels.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
