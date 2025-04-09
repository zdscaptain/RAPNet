_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/Landslide.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_adamw.py'
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
randomness = dict(seed=11)
find_unused_parameters = True
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        type='U_pswinHead2',
        feature_strides=[2, 4, 8, 16],
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        decoder_params=dict(
            embed_dim=768,
            num_heads=[32, 16, 8, 4],
            pool_ratio=[1, 2, 4, 8]),
        num_classes=2,
        # 关键修正：将 loss_decode 定义在 decode_head 内
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            # dict(type='BoundaryLoss', loss_weight=0.5),  # 边界损失
        ]),
    auxiliary_head=dict(in_channels=384, num_classes=2)
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader