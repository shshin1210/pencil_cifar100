import argparse
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import os
import os.path
import sys
import matplotlib.pyplot as plt

from dataset import datasets
from torch.utils.data import random_split

from resnet34 import ResNet34

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

print('device number:', torch.cuda.device_count())
print('current device:', torch.cuda.current_device())


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

# lr
parser.add_argument('--lr', '--learning-rate', default=0.35, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--lr2', '--learning-rate2', default=0.2, type=float,
                    metavar='H-P', help='initial learning rate of stage3')

# a/b/lambda
parser.add_argument('--alpha', default=0.1, type=float,
                    metavar='H-P', help='the coefficient of Compatibility Loss')
parser.add_argument('--beta', default=0.4, type=float,
                    metavar='H-P', help='the coefficient of Entropy Loss')
parser.add_argument('--lambda1', default=10000, type=int,
                    metavar='H-P', help='the value of lambda')

# epochs
parser.add_argument('--stage1', default=70, type=int,
                    metavar='H-P', help='number of epochs utill stage1')
parser.add_argument('--stage2', default=200, type=int,
                    metavar='H-P', help='number of epochs utill stage2')
parser.add_argument('--epochs', default=320, type=int, metavar='H-P',
                    help='number of total epochs to run')

# about data
parser.add_argument('--datanum', default=39999, type=int,
                    metavar='H-P', help='number of train dataset samples')
parser.add_argument('--classnum', default=100, type=int,
                    metavar='H-P', help='number of train dataset classes')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', default=False,dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')

parser.add_argument('--dir', dest='dir', default='./model_dir', type=str, metavar='PATH',
                    help='model dir')
parser.add_argument('--best',default= False, type=bool, help='use model_best.pth.tar or not')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    y_file = args.dir + "/y.npy"

    if os.path.exists('./model_dir'):
        print("model_dir already exists")
    else:
        os.makedirs(args.dir)
        os.makedirs(args.dir+'/record')

    model = ResNet34()
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    checkpoint_dir = args.dir + "/checkpoint.pth.tar"
    modelbest_dir = args.dir + "/model_best.pth.tar"

    # optionally resume from a checkpoint
    if args.best:
        if os.path.isfile(modelbest_dir):
            print("=> loading checkpoint '{}'".format(modelbest_dir))
            checkpoint = torch.load(modelbest_dir)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(modelbest_dir, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(modelbest_dir))

    else:
        if os.path.isfile(checkpoint_dir):
            print("=> loading checkpoint '{}'".format(checkpoint_dir))
            checkpoint = torch.load(checkpoint_dir)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_dir, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_dir))

    cudnn.benchmark = True

    # Data loading code
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding =4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # dataset
    trainset = datasets.C100Dataset(train=True, val = False, transform=transform_train)
    valset = datasets.C100Dataset(train=True, val = True, transform = transform_test)
    testset = datasets.C100Dataset(train=False, transform=transform_test)


    # data loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True,num_workers=args.workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False,num_workers=args.workers, pin_memory=True, drop_last=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, pin_memory=True)

    # for testing --evaluate
    if args.evaluate:
        _, acc = validate(testloader, model, criterion)
        print(acc)
        return

    acc_trainlist = []
    acc_testlist = []

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # load y_tilde
        if os.path.isfile(y_file):
            y = np.load(y_file)
        else:
            y = []
        
        acc = train(trainloader, model, criterion, optimizer, epoch, y)
        acc_trainlist.append(acc)
        print("train total acc of epoch [%d] : %.4f %%" %(epoch,acc))

        # evaluate on validation set
        prec1, _ = validate(valloader, model, criterion)
        _, acc_test = validate(testloader, model, criterion)
        acc_testlist.append(acc_test)
        print("test total acc of epoch [%d] : %.4f %%" %(epoch,acc_test))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,filename=checkpoint_dir,modelbest=modelbest_dir)

    # draw plot
    epochs = np.arange(0,args.epochs)
    plt.figure(figsize=(10,5))
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy') 
    plt.plot(epochs, acc_trainlist) 
    plt.plot(epochs, acc_testlist) 
    plt.savefig('plot.png')

    print(acc_testlist)
    print(max(acc_testlist))

def train(train_loader, model, criterion, optimizer, epoch, y):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    model.train()

    # new y is y_tilde after updating
    new_y = np.zeros([args.datanum,args.classnum])

    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda()
        index = index.numpy()

        target1 = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target1)

        # compute output
        output = model(input_var)

        logsoftmax = nn.LogSoftmax(dim=1).cuda()
        softmax = nn.Softmax(dim=1).cuda()

        # loss
        if epoch < args.stage1:
            # lc is classification loss
            lc = criterion(output, target_var)
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(target.size(0), 100).scatter_(1, target.view(-1, 1), 10.0)
            onehot = onehot.numpy()
            new_y[index, :] = onehot
        else:
            yy = y
            yy = yy[index,:]
            yy = torch.FloatTensor(yy)
            yy = yy.cuda(non_blocking = True)
            yy = torch.autograd.Variable(yy,requires_grad = True)
            # obtain label distributions (y_hat)
            last_y_var = softmax(yy)
            lc = torch.mean(softmax(output)*(logsoftmax(output)-torch.log((last_y_var))))
            # lo is compatibility loss
            lo = criterion(last_y_var, target_var)
        # le is entropy loss
        le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

        if epoch < args.stage1:
            loss = lc
        elif epoch < args.stage2:
            loss = lc + args.alpha * lo + args.beta * le
        else:
            loss = lc

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target1, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # real acc
        _, predicted = output.max(1)
        total += target_var.size(0)
        correct += predicted.eq(target_var).sum().item()
        acc = (100.*correct/total)


        if epoch >= args.stage1 and epoch < args.stage2:
            lambda1 = args.lambda1
            # update y_tilde by back-propagation
            yy.data.sub_(lambda1*yy.grad.data)
            # update new_y (index)
            new_y[index,:] = yy.data.cpu().numpy()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
    if epoch < args.stage2:
        # save y_tilde
        y = new_y
        y_file = args.dir + "/y.npy"
        np.save(y_file,y)
        y_record = args.dir + "/record/y_%03d.npy" % epoch
        np.save(y_record,y)

    return acc


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0
    total = 0

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # real acc
        _, predicted = output.max(1)
        total += target_var.size(0)
        correct += predicted.eq(target_var).sum().item()
        acc = (100.*correct/total)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, acc


def save_checkpoint(state, is_best, filename='', modelbest = ''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if epoch < args.stage1:
        lr = args.lr
    elif epoch < args.stage2 :
        lr = args.lr/10
    elif epoch < (args.epochs - args.stage2)//3 + args.stage2:
        lr = args.lr2
    elif epoch < 2 * (args.epochs - args.stage2)//3 + args.stage2:
        lr = args.lr2/10
    else:
        lr = args.lr2/100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk) # 5
    batch_size = target.size(0)

    _ , pred = output.topk(maxk, 1, True, True)
    pred = pred.t() # transposes dimensions 0 and 1
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # correct to same size as pred

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
