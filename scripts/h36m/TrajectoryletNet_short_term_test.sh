#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..
savepath='/home/data2/lxldata/Trajectorylet_exp/h36m/results/short_term_test'
modelpath='/home/data2/lxldata/Trajectorylet_exp/h36m/models/short_term_test'
bak_path='/home/data2/lxldata/Trajectorylet_exp/h36m/models/short_term_test/bak'
pretrain_modelpath='/home/data2/lxldata/Trajectorylet_exp/h36m/models/traj_finetune_mpjpe_pretrain_stack4_droupout_call64_final_20_2/bestmodel_26000/model.ckpt-26000'
mkdir ${modelpath}
mkdir ${savepath}
mkdir ${bak_path}
#sleep 4h
logname='logs/h36m/short_term_test_h36m.log'
nohup python -u test_TrajectoryletNet_h36m.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/h36m20/h36m20_train_3d.npy \
    --valid_data_paths data/h36m20/h36m20_val_3d.npy \
    --test_data_paths data/h36m20/test20_npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --bak_dir ${bak_path}   \
    --pretrained_model ${pretrain_modelpath}  \
    --input_length 10 \
    --seq_length 20 \
    --stacklength 4 \
    --filter_size 3 \
    --lr 0.00003 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 300000 \
    --display_interval 10 \
    --test_interval 500 \
    --snapshot_interval 500  >${logname}  2>&1 &

tail -f ${logname}

#  --pretrained_model checkpoints/ske_predcnn/model.ckpt-1000 \
# --pretrained_model ${pretrain_modelpath}  \



