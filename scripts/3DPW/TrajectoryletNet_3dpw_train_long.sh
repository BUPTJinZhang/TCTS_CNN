#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
cd ../..
#savepath='/home/data2/zhangjin/Trajectorylet_exp/322/G3D/results/g3d_joints_351053_filtersize3_learky_relu_droupout_2/v2'
#modelpath='/home/data2/zhangjin/Trajectorylet_exp/322/G3D/models/g3d_joints_351053_filtersize3_learky_relu_droupout_2/v2'
#pretrain_modelpath='/home/data2/zhangjin/Trajectorylet_exp/322/G3D/models/g3d_joints_351053_filtersize3_learky_relu_droupout_2/model_155500/model.ckpt-155500'
pretrain_modelpath='/home/zhangjin/intf/Traj_2str_1/exp/3dpw_l/24pre/1/model.ckpt-59000'
# 462000+167000
#bak_path='/home/data2/zhangjin/Trajectorylet_exp/322/G3D/models/g3d_joints_351053_filtersize3_learky_relu_droupout_2/bak2'
savepath='./exp/3dpw_l/results'
modelpath='./exp/3dpw_l/models'
bak_path='./exp/3dpw_l/bak6'
#sleep 4h
mkdir ${modelpath}
mkdir ${savepath}
mkdir ${bak_path}
logname='logs/3DPW/1_31.log'
nohup python -u train_TrajectoryletNet_3dpw.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths train_3dpw0_40.npy \
    --valid_data_paths test_3dpw0_40.npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --bak_dir ${bak_path}   \
    --pretrained_model ${pretrain_modelpath}  \
    --input_length 10 \
    --seq_length 40 \
    --filter_size 3 \
    --stacklength 4 \
    --lr 0.0001 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 1200000 \
    --display_interval 10 \
    --test_interval 1000 \
    --sampling_stop_iter 500 \
	--snapshot_interval 500 >>${logname}  2>&1 & 
end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "end time: ${end_time}"
tail -f ${logname}
#  --pretrained_model checkpoints/ske_predcnn/model.ckpt-1000 \
# --pretrained_model ${pretrain_modelpath}  \



