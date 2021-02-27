#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..
savepath='/home/data2/lxldata/Trajectorylet_exp/h36m/results/traj_finetune_mpjpe_pretrain_stack4_droupout_call64_final_2/v4'
modelpath='/home/data2/lxldata/Trajectorylet_exp/h36m/models/traj_finetune_mpjpe_pretrain_stack4_droupout_call64_final_2/v4'
bak_path='/home/data2/lxldata/Trajectorylet_exp/h36m/models/traj_finetune_mpjpe_pretrain_stack4_droupout_call64_final_2/bak4'
pretrain_modelpath='/home/data2/lxldata/Trajectorylet_exp/h36m/models/traj_finetune_mpjpe_pretrain_stack4_droupout_call64_final_2/v3/model.ckpt-4500'
mkdir ${modelpath}
mkdir ${savepath}
mkdir ${bak_path}
#sleep 4h
logname='logs/h36m/h36m_joints_351053_filtersize3_learky_relu_traj_mpjpe_finetune_pretrain_stack4_droupout_call64_final_2.log'
nohup python -u train_TrajectoryletNet_h36m.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/h36m/h36m_train_3d.npy \
    --valid_data_paths data/h36m/h36m_val_3d.npy \
    --test_data_paths data/h36m/test_npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --bak_dir ${bak_path}   \
    --pretrained_model ${pretrain_modelpath}  \
    --input_length 10 \
    --seq_length 35 \
    --stacklength 4 \
    --filter_size 3 \
    --lr 0.00003 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 300000 \
    --display_interval 10 \
    --test_interval 500 \
    --snapshot_interval 500  >>${logname}  2>&1 &

tail -f ${logname}

#  --pretrained_model checkpoints/ske_predcnn/model.ckpt-1000 \
# --pretrained_model ${pretrain_modelpath}  \



