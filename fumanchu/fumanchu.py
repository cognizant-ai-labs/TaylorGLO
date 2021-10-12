'''
Fumanchu: training framework for classification domains.
Santiago Gonzalez <slgonzalez@utexas.edu>
'''
from __future__ import print_function

import argparse
import os
import sys
import shutil
import time
import random
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from datasets.crispr import load_crispr_genomic

import numpy as np

from utils import Bar, Logger, AverageMeter, accuracy, calc_mse, mkdir_p, savefig, LossferatuLoss, LossferatuActivation, LossferatuSampler, Cutout, rand_bbox, fgsm_attack


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names += ['nobn_' + n for n in model_names if n.endswith('resnet')]
model_names += ['noskip_' + n for n in model_names if n.endswith('resnet')]

parser = argparse.ArgumentParser(description='Fumanchu Training')

parser.add_argument('--loss', default='loss_x*tf.log(loss_y)', type=str)
parser.add_argument('--activation-fn', default=None, type=str)
parser.add_argument('--start-cross-entropy-loss-epoch', default=None, type=int)
parser.add_argument('--eval-dataset', default='test', type=str)
parser.add_argument('--dataset-percentage', default=1.0, type=float, help='A value in (0.0,1.0].')
parser.add_argument('--full-run-epochs', default=None, type=int,
                    help='number of epochs in a training session, defaults to the number of epochs')
# parser.add_argument('--evaluated-metrics', default='{"metric": "accuracy", "dataset": "validation"},{"metric": "loss", "dataset": "validation"}', type=str)

parser.add_argument('--auxiliary-classifiers', default=0, type=int)

parser.add_argument('--unacceptable-accuracy-threshold', default=0.0, type=float)
parser.add_argument('--unacceptable-accuracy-threshold-epoch', default=10, type=int)

parser.add_argument('--record-loss-inputs', default=False, type=bool, help='1 if record else 0')

# Cutout
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--cutout-n-holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--cutout-length', type=int, default=16,
                    help='length of the holes')

# Cutmix
parser.add_argument('--cutmix', action='store_true', default=False,
                    help='apply cutmix')
parser.add_argument('--cutmix-prob', type=float, default=0.5,
                    help='cutmix probability')
parser.add_argument('--cutmix-alpha', type=float, default=1.0,
                    help='cutmix alpha')

# FGSM Attack
parser.add_argument('--fgsm', action='store_true', default=False,
                    help='run FGSM evals')
parser.add_argument('--fgsm-epsilon', type=float, default=0.1,
                    help='FGSM attack epsilon')
# parser.add_argument('--fgsm-epsilons', type=float, nargs='+', default=[0, .05, .1, .15, .2, .25, .3],
#                         help='Epsilon values to evaluate for the FGSM attack.')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--every-epoch-gamma', type=float, default=1.0, help='LR is multiplied by every_epoch_gamma at every epoch.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD or Adam')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

# Architecture configuration
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--pyramidnet-alpha', type=int, default=48, help='PyramidNet alpha value.')
parser.add_argument('--pyramidnet-bottleneck', action='store_true', default=False, help='1 if bottleneck else 0')
# Miscs
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--evaluate-hessian-minmax-eigens', dest='evaluate_hessian_minmax_eigens', action='store_true',
                    help='Whether to calculate the min and max eigenvalues of the Hessian matrix')
parser.add_argument('--evaluation-grid', type=str, default=None, help='xmin:xmax:xnum::ymin:ymax:ynum')
parser.add_argument('--evaluation-grid-directory', type=str, default='vis', help='A fresh directory where the evaluation grid data is saved')
parser.add_argument('--evaluation-store-loss-activations', dest='evaluation_store_loss_activations', action='store_true')
parser.add_argument('--evaluation-store-loss-activations-directory', type=str, default='activations', help='A fresh directory where the loss activation data is saved')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# # Process metrics
# data = json.loads(args.evaluated_metrics)
# metrics = [metric['metric'] for metric in data]
# print('==> Metrics to evaluate: ' + ', '.join(metrics))

# Validate dataset
allowable_datasets = ['cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn', 'aircraft', 'crispr']
assert args.dataset in allowable_datasets, 'Dataset can only be ' + ' or '.join(allowable_datasets) + '.'

if args.dataset == 'svhn' or args.dataset == 'imagenet':
    import scipy # needed to load svhn and imagenet datasets

if args.dataset == 'crispr':
    import datasets.crispr as crispr

is_classification = (args.dataset != 'crispr')

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
np.random.seed(args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)

best_acc = 0  # the best eval accuracy
latest_acc = 0  # the most recent eval accuracy
best_mse = 9999999
latest_mse = 9999999
latest_loss = 0  # the most recent loss on the eval set

def main():
    # global metrics
    global best_acc
    global latest_acc
    global best_mse
    global latest_mse
    global latest_loss
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if args.full_run_epochs is None:
        args.full_run_epochs = args.epochs

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    
    if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    if args.dataset == 'aircraft':
        dimension = [100, 100]
        padding = (4, 4)

        transform_train = transforms.Compose([
            # transforms.Pad((5,14)),
            # transforms.RandomCrop(dimension, padding=padding),
            # transforms.CenterCrop(dimension),
            transforms.RandomCrop([72, 72]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop([72, 72]),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else: # TODO: handle normalization for SVHN
        
        dimension = [32, 32]
        padding = (4, 4)

        if args.dataset == 'svhn':
            normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                             std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        else:
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop([32, 32], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        transform_train.transforms.append(normalize)
        transform_test.transforms.append(normalize)

    # Add cutout if warranted.
    if args.cutout:
        transform_train.transforms.append(Cutout(n_holes=args.cutout_n_holes, length=args.cutout_length))


    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
    elif args.dataset == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10
    elif args.dataset == 'fashionmnist':
        dataloader = dataset.FashionMNIST
        num_classes = 10
    elif args.dataset == 'mnist':
        dataloader = datasets.MNIST
        num_classes = 10
    elif args.dataset == 'aircraft':
        num_classes = 100

    print("* ")

    # Define dataset splits using samplers.
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_sampler = LossferatuSampler(round(45000*args.dataset_percentage), total_samples=50000, start=0, shuffle=True, preshuffle_with_seed=42)
        val_sampler = LossferatuSampler(5000, total_samples=50000, start=45000, shuffle=False, preshuffle_with_seed=42)
        test_sampler = LossferatuSampler(10000, total_samples=10000, start=0, shuffle=False)
    elif args.dataset == 'svhn':
        train_sampler = LossferatuSampler(round(47225*args.dataset_percentage), total_samples=73257, start=0, shuffle=True, preshuffle_with_seed=42)
        val_sampler = LossferatuSampler(26032, total_samples=73257, start=47225, shuffle=False, preshuffle_with_seed=42)
        test_sampler = LossferatuSampler(26032, total_samples=26032, start=0, shuffle=False)
    elif args.dataset == 'mnist' or args.dataset == 'fashionmnist':
        train_sampler = LossferatuSampler(round(55000*args.dataset_percentage), total_samples=60000, start=0, shuffle=True, preshuffle_with_seed=42)
        val_sampler = LossferatuSampler(5000, total_samples=60000, start=55000, shuffle=False, preshuffle_with_seed=42)
        test_sampler = LossferatuSampler(10000, total_samples=10000, start=0, shuffle=False)
    elif args.dataset == 'aircraft':
        train_sampler = LossferatuSampler(round(3334*args.dataset_percentage), total_samples=3334, start=0, shuffle=True, preshuffle_with_seed=42)
        val_sampler = LossferatuSampler(1667, total_samples=3333, start=1666, shuffle=False, preshuffle_with_seed=42)
        test_sampler = LossferatuSampler(1666, total_samples=3333, start=0, shuffle=False, preshuffle_with_seed=42)


    if args.dataset == 'svhn':
        trainset = dataloader(root='./data', split='train', download=True, transform=transform_train)
        valset = dataloader(root='./data', split='train', download=True, transform=transform_test)
        testset = dataloader(root='./data', split='test', download=True, transform=transform_test)
    elif args.dataset == 'aircraft':
        trainset = datasets.ImageFolder(root='./data/train', transform=transform_train)
        valset = datasets.ImageFolder(root='./data/val', transform=transform_test)
        testset = datasets.ImageFolder(root='./data/val', transform=transform_test)
    elif args.dataset != 'crispr':
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        valset = dataloader(root='./data', train=True, download=True, transform=transform_test)
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

    if args.dataset == 'crispr':
        trainsize = 30786441
        valsize = 1512119 if args.eval_dataset == 'test' else 1513446
        trainloader, valloader, testloader, classes = load_crispr_genomic(dataset_folder='./data', batch_size=args.train_batch, steps_per_epoch=int(trainsize/args.train_batch), validation_steps=int(valsize/args.test_batch), dataset_percentage=args.dataset_percentage)
        testloader = testloader if args.eval_dataset == 'test' else valloader
    else:
        trainloader = data.DataLoader(trainset,
            batch_size=args.train_batch,
            shuffle=False, # Shuffling needs to be false since we are using a custom sampler.
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True
        )
        testloader = data.DataLoader(testset if args.eval_dataset == 'test' else valset,
            batch_size=args.test_batch,
            shuffle=False,
            sampler=test_sampler if args.eval_dataset == 'test' else val_sampler,
            num_workers=args.workers,
            pin_memory=True
        )



    # Model
    print("==> Creating model '{}'".format(args.arch))
    real_model_reference = args.arch.split('_')[-1]
    if args.arch.startswith('resnext'):
        model = models.__dict__[real_model_reference](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[real_model_reference](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        activation_fn = LossferatuActivation(args.activation_fn) if args.activation_fn != None else nn.ReLU(inplace=True)
        model = models.__dict__[real_model_reference](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                    activation_fn=activation_fn,
                )
    elif args.arch.endswith('fixup_resnet'):
        model = models.__dict__[real_model_reference](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch.endswith('preresnet'): # THIS IS TEMPORARY, SINCE PRERESNET DOES NOT SUPPORT SOME FLAGS AND VICE-VERSA
        activation_fn = LossferatuActivation(args.activation_fn) if args.activation_fn != None else nn.ReLU(inplace=True)
        model = models.__dict__[real_model_reference](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                    auxiliary_classifiers=(args.auxiliary_classifiers > 0),
                    activation_fn=activation_fn,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[real_model_reference](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                    use_batch_norm=('nobn' not in args.arch),
                    use_skip_connection=('noskip' not in args.arch),
                )
    elif args.arch.startswith('efficientnet'):
        model = models.__dict__[real_model_reference](
                    width_coeff=1,
                    depth_coeff=1,
                    dropout_rate=args.drop,
                    num_classes=num_classes,
                )
    elif args.arch.startswith('pyramidnet'):
        model = models.__dict__[real_model_reference](
                    dataset=args.dataset,
                    depth=args.depth,
                    alpha=args.pyramidnet_alpha,
                    num_classes=num_classes,
                    bottleneck=args.pyramidnet_bottleneck,
                )
    elif args.arch.startswith('allcnn'):
        activation_fn = LossferatuActivation(args.activation_fn) if args.activation_fn != None else nn.ReLU(inplace=True)
        model = models.__dict__[real_model_reference](
                    num_classes=num_classes,
                    auxiliary_classifiers=(args.auxiliary_classifiers > 0),
                    activation_fn=activation_fn
                )
    elif args.arch.startswith('deepbind'):
        model = models.__dict__[real_model_reference](
                    context_size=0,
                    block_in=16,
                    block_out= 16,
                    model_config={'context_size': 20},
                    hyper=False,
                    filters=256,
                    hidden_units=256
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criteria = [LossferatuLoss(loss_str, classification=is_classification) for loss_str in args.loss.split('===') ]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("INVALID OPTIMIZER: " + args.optimizer)
        sys.exit(1)

    # Resume
    title = args.dataset + '-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint...')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        best_mse = checkpoint['best_mse']
        start_epoch = checkpoint['epoch']

        print("-----")
        print(checkpoint['state_dict'].keys())
        print("-----")

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train MSE', 'Valid MSE'])
 

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc, test_mse = test(testloader, model, criteria, start_epoch, use_cuda)
        if is_classification:
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        else:
            print(' Test Loss:  %.8f, Test MSE:  %.2f' % (test_loss, test_mse))
        

        if args.evaluation_grid is not None:
            print("==> Starting grid evaluation...")
            training_progress = float(start_epoch) / float(args.full_run_epochs)
            geval_train = True
            geval_test = True
            grid_eval(model, trainloader, testloader, geval_train, geval_test, criteria, start_epoch, training_progress)


        if args.evaluate_hessian_minmax_eigens:
            import loss_landscape.hess_vec_prod as hess_vec_prod
            model.eval()
            training_progress = float(start_epoch) / float(args.full_run_epochs) #training_progress = float(epoch if args.full_run_epochs is None else args.full_run_epochs) / float(args.full_run_epochs)
            maxeig, mineig, iter_count = hess_vec_prod.min_max_hessian_eigs(model.module, trainloader, \
                                                criteria, rank=0, use_cuda=use_cuda, verbose=True, training_progress=training_progress)
            print("MAX, MIN, ITER_COUNT")
            print(maxeig, mineig, iter_count)


        # Print dict.
        metrics_dict = {'epoch': args.epochs+1, 'reached_unacceptable_accuracy_threshold': False}
        # if 'loss' in metrics_dict:
        # if 'accuracy' in metrics_dict:
        metrics_dict['loss'] = test_loss
        if is_classification:
            metrics_dict['accuracy'] = test_acc/100.0
        else:
            metrics_dict['mse'] = test_mse
        print(metrics_dict)



        return

    # Train and perform inference.
    reached_unacceptable_accuracy_threshold = False
    latest_epoch = 0
    for epoch in range(start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        latest_epoch = epoch
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        if args.start_cross_entropy_loss_epoch is not None and epoch == args.start_cross_entropy_loss_epoch:
            criteria = [LossferatuLoss('loss_x*tf.log(loss_y)', classification=is_classification)] # TODO: make this work for auxiliary classifiers

        train_loss, train_acc, train_mse = train(trainloader, model, criteria, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_mse = test(testloader, model, criteria, epoch, use_cuda)

        # Append to log file.
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_mse, test_mse])

        # Save model.
        is_best = (test_acc > best_acc) if is_classification else (test_mse < best_mse)
        best_acc = max(test_acc, best_acc)
        best_mse = min(test_mse, best_mse)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'best_mse': best_mse,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        # Keep some metrics.
        latest_acc = test_acc
        latest_mse = test_mse
        latest_loss = test_loss

        # Check performance threshold.
        if epoch == args.unacceptable_accuracy_threshold_epoch and (is_classification and test_acc/100.0 < args.unacceptable_accuracy_threshold):
            reached_unacceptable_accuracy_threshold = True
            break


    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    if is_classification:
        print('Best acc:')
        print(best_acc)
    else:
        print('Best MSE:')
        print(best_mse)

    # Print dict.
    metrics_dict = {'epoch': latest_epoch+1, 'reached_unacceptable_accuracy_threshold': reached_unacceptable_accuracy_threshold}
    # if 'loss' in metrics_dict:
    # if 'accuracy' in metrics_dict:
    metrics_dict['loss'] = latest_loss
    if is_classification:
        metrics_dict['accuracy'] = latest_acc/100.0
    else:
        metrics_dict['mse'] = latest_mse
    print(metrics_dict)


def train(trainloader, model, criteria, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mse = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        training_progress = float(epoch if args.full_run_epochs is None else args.full_run_epochs) / float(args.full_run_epochs)

        # Measure data loading time.
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # Compute output.
        if args.cutmix and args.cutmix_alpha > 0 and np.random.rand(1) < args.cutmix_prob:

            if args.auxiliary_classifiers > 0:
                print("---------------------")
                print("-- Unsupported number of auxiliary classifiers with MixUp.")
                exit(1)

            # Mixup.
            lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            criterion_a = criteria[0]
            criterion_b = criteria[1] if len(criteria) == 2 else criteria[0]
            loss = criterion_a(outputs, target_a, training_progress) * lam + criterion_b(outputs, target_b, training_progress) * (1. - lam)

        else:
            
            # Compute output.
            if args.auxiliary_classifiers == 0:
                outputs = model(inputs)
                loss = criteria[0](outputs, targets, training_progress)
            elif args.auxiliary_classifiers == 2:
                outputs, aux1, aux2 = model(inputs)
                loss = criteria[0](outputs, targets, training_progress)
                loss += criteria[1](aux1, targets, training_progress)
                loss += criteria[2](aux2, targets, training_progress)
                # print(criteria[2](aux2, targets, training_progress).data.item())
            else:
                print("---------------------")
                print("-- Unsupported number of auxiliary classifiers.")
                exit(1)

        

        # Measure accuracy and record loss.
        losses.update(loss.data.item(), inputs.size(0))
        if is_classification:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            mse.update(calc_mse(outputs.data, targets.data).item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if is_classification:
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        else:
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | MSE: {mse: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    mse=mse.avg,
                    )

        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, mse.avg)

def test(testloader, model, criteria, epoch, use_cuda):
    global best_acc
    global best_mse

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mse = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):

        training_progress = float(epoch if args.full_run_epochs is None else args.full_run_epochs) / float(args.full_run_epochs)

        # Measure data loading time.
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        # Setup for FGSM.
        if args.fgsm:
            inputs.requires_grad = True

        # Compute output.
        if args.auxiliary_classifiers == 0:
            outputs = model(inputs)
            loss = criteria[0](outputs, targets, training_progress)
        elif args.auxiliary_classifiers == 2:
            outputs, aux1, aux2 = model(inputs)
            loss = criteria[0](outputs, targets, training_progress)
            loss += criteria[1](aux1, targets, training_progress)
            loss += criteria[2](aux2, targets, training_progress)
        else:
            print("---------------------")
            print("-- Unsupported number of auxiliary classifiers.")
            exit(1)

        # Run FGSM if needed.
        if args.fgsm:
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = inputs.grad.data
            # Call FGSM Attack
            perturbed_data = fgsm_attack(inputs, args.fgsm_epsilon, data_grad)
            # Re-classify the perturbed image
            if args.auxiliary_classifiers == 0:
                outputs = model(perturbed_data)
                loss = criteria[0](outputs, targets, training_progress)
            elif args.auxiliary_classifiers == 2:
                outputs, aux1, aux2 = model(perturbed_data)
                loss = criteria[0](outputs, targets, training_progress)
                loss += criteria[1](aux1, targets, training_progress)
                loss += criteria[2](aux2, targets, training_progress)
            else:
                print("---------------------")
                print("-- Unsupported number of auxiliary classifiers.")
                exit(1)


            # # Check for success
            # final_pred = outputs.max(1, keepdim=True)[1] # get the index of the max log-probability
            # if final_pred.item() == target.item():
            #     correct += 1
            #     # Special case for saving 0 epsilon examples
            #     if (args.fgsm_epsilon == 0) and (len(adv_examples) < 5):
            #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            #         adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            # else:
            #     # Save some adv examples for visualization later
            #     if len(adv_examples) < 5:
            #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            #         adv_examples.append( (init_pred.item(), final_pred.item(), 

        # Store the activations if needed.
        if args.evaluation_store_loss_activations:
            combined = np.concatenate((targets.cpu().detach().numpy()[..., np.newaxis], outputs.cpu().detach().numpy()), axis=1)
            savefilename = os.path.join(args.evaluation_store_loss_activations_directory, 'activations_' + str(epoch) + '.csv')
            if batch_idx == 0:
                with open(savefilename, 'w') as file:
                    file.write("Label,\n")
            with open(savefilename, 'ab') as file:
                np.savetxt(file, combined, delimiter=',')


        # Measure accuracy and record loss.
        losses.update(loss.data.item(), inputs.size(0))
        if is_classification:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            mse.update(calc_mse(outputs.data, targets.data).item(), inputs.size(0))

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        # Plot progress.
        if is_classification:
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        else:
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | MSE: {mse: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    mse=mse.avg,
                    )

        bar.next()
    bar.finish()


    return (losses.avg, top1.avg, mse.avg)






def grid_eval(model, train_dataloader, test_dataloader, eval_train, eval_test, criteria, epoch, training_progress):
    args.dir_file = ''
    args.dir_type = 'weights'
    args.xnorm = 'filter'
    args.ynorm = 'filter'
    args.xignore = 'biasbn'
    args.yignore = 'biasbn'
    args.same_dir = False
    args.surf_file = ''
    args.model_file1 = False
    args.model_file2 = False
    args.model_file3 = False
    args.ngpu = 1

    rank = 0
    comm = None

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        gridaxes = args.evaluation_grid.split('::')
        args.x = gridaxes[0]
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        args.xnum = int(args.xnum)
        if len(gridaxes) > 1:
            args.y = gridaxes[1]
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            args.ynum = int(args.ynum)
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format grid')

    import loss_landscape.net_plotter as net_plotter
    import loss_landscape.scheduler as scheduler
    import loss_landscape.projection as proj
    import loss_landscape.h52vtp as h52vtp
    import copy
    import h5py
    import numpy as np

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    net = model.module.cuda()
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = args.evaluation_grid_directory + '/dirfile' #net_plotter.name_direction_file(args) # name the direction file
    net_plotter.setup_direction(args, dir_file, net)

    surf_file = args.evaluation_grid_directory + '/surffile'
    net_plotter.setup_surface_file(args, surf_file, dir_file)


    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    # Start crunch
    train_loss_key = 'train_loss'
    train_acc_key = 'train_acc'
    test_loss_key = 'test_loss'
    test_acc_key = 'test_acc'

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if eval_train:
        if train_loss_key not in f.keys():
            shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
            train_losses = -np.ones(shape=shape)
            train_accuracies = -np.ones(shape=shape)
            if rank == 0:
                f[train_loss_key] = train_losses
                f[train_acc_key] = train_accuracies
        else:
            train_losses = f[train_loss_key][:]
            train_accuracies = f[train_acc_key][:]
    if eval_test:
        if test_loss_key not in f.keys():
            shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
            test_losses = -np.ones(shape=shape)
            test_accuracies = -np.ones(shape=shape)
            if rank == 0:
                f[test_loss_key] = test_losses
                f[test_acc_key] = test_accuracies
        else:
            test_losses = f[test_loss_key][:]
            test_accuracies = f[test_acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(train_losses if eval_train else test_losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        # loss, acc = evaluation.eval_loss(net, criteria, dataloader, args.cuda)
        if eval_train:
            train_loss, train_acc, train_mse = test(train_dataloader, net, criteria, epoch, use_cuda)
        if eval_test:
            test_loss, test_acc, train_mse = test(test_dataloader, net, criteria, epoch, use_cuda)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        if eval_train:
            train_losses.ravel()[ind] = train_loss
            train_accuracies.ravel()[ind] = train_acc
            print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), train_loss_key, train_loss,
                train_acc_key, train_acc, loss_compute_time))
        if eval_test:
            test_losses.ravel()[ind] = test_loss
            test_accuracies.ravel()[ind] = test_acc
            print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), test_loss_key, test_loss,
                test_acc_key, test_acc, loss_compute_time))

    if eval_train:
        f[train_loss_key][:] = train_losses
        f[train_acc_key][:] = train_accuracies
    if eval_test:
        f[test_loss_key][:] = test_losses
        f[test_acc_key][:] = test_accuracies
    f.flush()

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

#    # Create paraview files.
#    if eval_train:
#        h52vtp.h5_to_vtp(surf_file, train_loss_key, log=False, zmax=10, interp=1000)
#        h52vtp.h5_to_vtp(surf_file, train_acc_key, log=False, zmax=10, interp=1000)
#    if eval_test:
#        h52vtp.h5_to_vtp(surf_file, test_loss_key, log=False, zmax=10, interp=1000)
#        h52vtp.h5_to_vtp(surf_file, test_acc_key, log=False, zmax=10, interp=1000)






def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    state['lr'] *= args.every_epoch_gamma # decay at every epoch
    if epoch in args.schedule:
        state['lr'] *= args.gamma # decay on schedule
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
