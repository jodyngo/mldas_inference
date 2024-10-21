# Copyright (c) OpenMMLab. All rights reserved.

dataset_type = 'MLDASSegDataset'
data_root = 'data/MLDAS/Campus'
class_names = ["car","bike", "person","rider","road",
              "sidewalk","building","fence", "vegetation","trunk", 
              "pole", "sign", "board", "oth-obj"] 

labels_map = {
  0:   0,       #"unlabeled"
  10:1,     #car
  11:2,     #bike
  12:   0,     #cone
  13:1,     #bus->car
  18:1,     #truck->car
  20:1,     #oth-vehicle->car
  30:3,     #person
  31:4,     #rider
  40:5,     #road
  48:6,     #sidewalk
  49:6,     #oth-ground -> sidewalk
  50:7,     #building
  51:8,     #fence
  52:    0,     #tent
  70:9,     #vegetation
  71:10,    #trunk
  72:     0,    #12,    #terrain
  80:11,    #pole
  81:12,    #sign
  82:13,    #board
  99:14,    #oth-obj
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=99)

input_modality = dict(use_lidar=True, use_camera=False)

backend_args = None

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
        seg_3d_dtype='np.int32', # check
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
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
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
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32', # check
        dataset_type=dataset_type,
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
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
        dataset_type=dataset_type,
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[pcd_rotate_range, pcd_rotate_range],
                            scale_ratio_range=[
                                pcd_scale_factor, pcd_scale_factor
                            ],
                            translation_std=[0, 0, 0])
                        for pcd_rotate_range in [-0.78539816, 0.0, 0.78539816]
                        for pcd_scale_factor in [0.95, 1.0, 1.05]
                    ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mldas_infos_train_Campus.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=0,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mldas_infos_val_Campus.pkl',
        pipeline=test_pipeline,     
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=0,
        test_mode=True,
        backend_args=backend_args))

val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')
