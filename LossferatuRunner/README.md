# LossferatuRunner

## Test Commands

### One-Shot

One-shot training MNIST:
```
rm -rf results/testDir && .build/debug/LossferatuRunner init results/testDir experiments/oneShot/MNIST.json && .build/debug/LossferatuRunner start results/testDir

.build/debug/LossferatuRunner check results/testDir
```

One-shot training CIFAR-10:
```
rm -rf results/testDirCifar10 && .build/debug/LossferatuRunner init results/testDirCifar10 experiments/oneShot/CIFAR10.json && .build/debug/LossferatuRunner start results/testDirCifar10

.build/debug/LossferatuRunner check results/testDirCifar10
```

One-shot training CIFAR-10:
```
rm -rf results/testDirCifar10ResNet && .build/debug/LossferatuRunner init results/testDirCifar10ResNet experiments/oneShot/CIFAR10ResNet.json && .build/debug/LossferatuRunner start results/testDirCifar10ResNet

.build/debug/LossferatuRunner check results/testDirCifar10ResNet
```

### TaylorGLO

MNIST:
```
rm -rf results/testDir && .build/debug/LossferatuRunner init results/testDir experiments/gloTaylor/MNIST.json && .build/debug/LossferatuRunner start results/testDir

.build/debug/LossferatuRunner check results/testDir
```
