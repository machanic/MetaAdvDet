nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset CIFAR-10 --adv_arch conv3 --shot 5 --gpu 1 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset CIFAR-10 --adv_arch conv3 --shot 5 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset SVHN --adv_arch conv3 --shot 1 --gpu 5 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset SVHN --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset MNIST --adv_arch conv3 --shot 5 --gpu 1 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset MNIST --adv_arch conv3 --shot 5 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &


nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset F-MNIST --adv_arch conv3 --shot 5 --gpu 1 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector MetaAdvDet --dataset F-MNIST --adv_arch conv3 --shot 5 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &




nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector DNN --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &


nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector RotateDet --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &



nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset CIFAR-10 --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset SVHN --adv_arch conv3 --shot 5 --gpu 2 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 0 --attack CW_L2 --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &
nohup python white_box_attack/generate_white_box_attack.py --detector NeuralFP --dataset F-MNIST --adv_arch conv3 --shot 1 --gpu 2 --attack FGSM --protocol TRAIN_ALL_TEST_ALL > /dev/null 2>&1 &

