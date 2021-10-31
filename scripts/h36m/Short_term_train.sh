#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
cd ../..
savepath='./exp/h36m/results'
modelpath='./exp/h36m/models'
bak_path='./exp/h36m/bak6'
# pretrain_modelpath='/home/data2/zhangjin/Trajectorylet_exp/327/h36m/models/pre'
mkdir ${modelpath}
mkdir ${savepath}
mkdir ${bak_path}
#sleep 4h
logname='logs/h36m/8_28.log'
nohup python3 -u train_TrajectoryletNet_h36m.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths h36m20_train_3d.npy \
    --valid_data_paths h36m20_val_3d.npy \
    --test_data_paths test20_npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --bak_dir ${bak_path}   \
    --pretrained_model ${pretrain_modelpath}  \
    --input_length 10 \
    --seq_length 20 \
    --stacklength 4 \
    --filter_size 3 \
    --lr 0.0001 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 3000000 \
    --display_interval 10 \
    --test_interval 500 \
    --n_gpu 2 \
    --snapshot_interval 500  >>${logname}  2>&1 &

tail -f ${logname}

#  --pretrained_model checkpoints/ske_predcnn/model.ckpt-1000 \
# --pretrained_model ${pretrain_modelpath}  \



