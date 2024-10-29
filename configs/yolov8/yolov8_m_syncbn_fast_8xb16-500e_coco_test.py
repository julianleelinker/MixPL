
# Inherit and overwrite part of the config based on this config
_base_ = './yolov8_m_syncbn_fast_8xb16-500e_coco.py'

data_root = './data/cat/' # dataset root
data_root = '/mnt/data-home/julian/KITTI-COCO/'
class_name = (
    'Pedestrian',
    'Truck',
    'Car',
    'Cyclist',
    'DontCare',
    'Misc',
    'Van',
    'Tram',
    'Person_sitting',
)
num_classes = len(class_name) # dataset category number
# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
metainfo = dict(classes=class_name, palette=[(20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60), (20, 220, 60)])
_base_.model.bbox_head.head_module.num_classes = num_classes
_base_.model.train_cfg.assigner.num_classes = num_classes

# Adaptive anchor based on tools/analysis_tools/optimize_anchors.py
# anchors = [
#     [(68, 69), (154, 91), (143, 162)],  # P3/8
#     [(242, 160), (189, 287), (391, 207)],  # P4/16
#     [(353, 337), (539, 341), (443, 432)]  # P5/32
# ]
# Max training 40 epoch
max_epochs = 40
# Set batch size to 12
train_batch_size_per_gpu = 24
# dataloader num workers
train_num_workers = 4


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # Dataset annotation file of json path
        ann_file='train/labels.json',
        # Dataset prefix
        data_prefix=dict(img='train/data/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val/labels.json',
        data_prefix=dict(img='val/data/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'val/labels.json')
test_evaluator = val_evaluator

default_hooks = dict(
    # Save weights every 10 epochs and a maximum of two weights can be saved.
    # The best model is saved automatically during model evaluation
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    # The log printing interval is 5
    logger=dict(type='LoggerHook', interval=5))
# The evaluation interval is 10
train_cfg = dict(max_epochs=max_epochs, val_interval=1)