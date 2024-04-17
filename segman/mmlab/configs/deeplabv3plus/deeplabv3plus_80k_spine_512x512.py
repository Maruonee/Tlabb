_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_spine.py',
    '../_base_/datasets/spine512x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'#학습횟수
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
