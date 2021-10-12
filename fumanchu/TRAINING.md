
## MNIST

#### CNNMNIST
```
python fumanchu.py -a cnnmnist -d mnist --train-batch 100 --epochs 40 --lr 0.01 --dropout 0.4 --wd 0.0 --checkpoint checkpoints/cifar10/alexnet 
```


## CIFAR-10

#### AlexNet
```
python fumanchu.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet 
```

#### AllCNN-C
```
python fumanchu.py -a allcnnc --dataset cifar10 --checkpoint checkpoints/cifar10/allcnnc --epochs 350 --schedule 200 250 300 --gamma 0.1  --wd 0.001 --lr 0.01
```

#### EfficientNet
```
python fumanchu.py -a efficientnet --dataset cifar10 --checkpoint checkpoints/cifar10/efficientnet --drop 0.2 --wd 0.00001 --lr 0.016 --train-batch 256 --test-batch 200
```

#### ResNet-32
```
python fumanchu.py -a resnet --depth 32 --epochs 200 --schedule 100 150 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-32
```

#### WRN-16-8-drop
```
python fumanchu.py -a wrn --depth 16 --depth 16 --widen-factor 8 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-16-8-drop
```

## CIFAR-100

#### AlexNet
```
python fumanchu.py -a alexnet --dataset cifar100 --checkpoint checkpoints/cifar100/alexnet --epochs 164 --schedule 81 122 --gamma 0.1 
```
