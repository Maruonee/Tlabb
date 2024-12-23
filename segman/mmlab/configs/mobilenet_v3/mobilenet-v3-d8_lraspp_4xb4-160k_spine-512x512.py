_base_ = [
    '../_base_/models/lraspp_m-v3-d8_spine.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://contrib/mobilenet_v3_large')

# Re-config the data sampler.
train_dataloader = dict(batch_size=4, num_workers=32)
val_dataloader = dict(batch_size=1, num_workers=32)
test_dataloader = val_dataloader

runner = dict(type='IterBasedRunner', max_iters=160000)
