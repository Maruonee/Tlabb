_base_ = [
    '../_base_/models/fcn_unet_s5-d16_spine.py', '../_base_/datasets/spine512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=19),
    auxiliary_head=dict(num_classes=19),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
train_dataloader = dict(batch_size=4, num_workers=32)
val_dataloader = dict(batch_size=1, num_workers=32)
test_dataloader = val_dataloader
