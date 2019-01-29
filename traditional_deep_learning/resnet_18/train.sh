#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
train_dir="./resnet_baseline"
train_data_list="/home1/machen/dataset/CIFAR-10/split_data_mem/train_image_label.txt"
val_data_list="/home1/machen/dataset/CIFAR-10/split_data_mem/test_image_label.txt"
train_image_root="/home1/machen/dataset/CIFAR-10/split_data_mem/train"
val_image_root="/home1/machen/dataset/CIFAR-10/split_data_mem/test"

CLASS_NUMBER = 15
python train.py --train_dir $train_dir \
    --train_dataset $train_data_list \
    --train_image_root $train_image_root \
    --val_dataset $val_data_list \
    --val_image_root $val_image_root \
    --batch_size 64 \
    --num_gpus 4 \
    --val_interval 1000 \
    --val_iter 100 \
    --l2_weight 0.0001 \
    --initial_lr 0.01 \
    --lr_step_epoch 10.0 \
    --lr_decay 0.1 \
    --max_steps 100000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 100 \
    --num_classes 15
    #--finetune True \

# ResNet-18 baseline loaded from torch resnet-18.t7
# Finetune for 10 epochs
