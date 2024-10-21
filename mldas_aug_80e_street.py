_base_ = ['../../../../configs/_base_/default_runtime.py',
          '../../../../configs/_base_/models/minkunet.py',
          './lidarseg_80e.py',
          './mldas-seg.py']

custom_imports = dict(
    imports=['projects.spin_projects.mink_nusc'], allow_failed_imports=False)

backend_args = None

dataset_type = 'MLDASSegDataset'
data_root = 'data/MLDAS/Street'


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type=dataset_type,
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),  
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),  
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),  
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mldas_infos_train_Street.pkl',
        ignore_index=0,
        pipeline=train_pipeline,
        test_mode=False))
    
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mldas_infos_val_Street.pkl',
        pipeline=test_pipeline,     
        ignore_index=0,
        test_mode=True,
        backend_args=backend_args))

model = dict(
    data_preprocessor=dict(
        max_voxels=None,
        batch_first=True,
        voxel_layer=dict(voxel_size=[0.1, 0.1, 0.1])),
    backbone=dict(encoder_blocks=[2, 3, 4, 6],
                  sparseconv_backend='spconv'),
    decode_head=dict(
        num_classes=15,
        ignore_index=0
    ))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', save_best='miou'
    ),
    visualization=dict(type='Det3DVisualizationHook', vis_task='lidar-seg')
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

