_base_ = [
    '../_base_/models/fcn_hr18_spine.py', '../_base_/datasets/spine512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
