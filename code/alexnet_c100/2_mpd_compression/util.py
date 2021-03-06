import os
import torch
import math
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
def load_checkpoint(model, file,args):
    if os.path.isfile(file):
        print ("=> loading checkpoint '{}'".format(file))
        checkpoint = torch.load(file)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])

        temp1=torch.index_select(model.fc1.weight, 0, model.rowp1.cuda())
        model.fc1.weight=torch.nn.Parameter(torch.index_select(temp1, 1, model.colp1.cuda()))
        temp2=torch.index_select(model.fc2.weight, 0, model.rowp2.cuda())
        model.fc2.weight=torch.nn.Parameter(torch.index_select(temp2, 1, model.colp2.cuda()))
        temp3=torch.index_select(model.fc3.weight, 0, model.rowp3.cuda())
        model.fc3.weight=torch.nn.Parameter(torch.index_select(temp3, 1, model.colp3.cuda()))
        print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
    else:
        print ("=> no checkpoint found at '{}'".format(file))
    return model

def save_masked_checkpoint(model, mode, is_best, best_prec1, epoch, args):
    tempfc1=torch.index_select(model.fc1.weight, 0, model.invrow1.cuda())
    model.fc1.weight=torch.nn.Parameter(torch.index_select(tempfc1, 1, model.invcol1.cuda()))
    tempfc2=torch.index_select(model.fc2.weight, 0, model.invrow2.cuda())
    model.fc2.weight=torch.nn.Parameter(torch.index_select(tempfc2, 1, model.invcol2.cuda()))
    tempfc3=torch.index_select(model.fc3.weight, 0, model.invrow3.cuda())
    model.fc3.weight=torch.nn.Parameter(torch.index_select(tempfc3, 1, model.invcol3.cuda()))
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}_alpha_{}_{}.tar'.format(mode,args.alpha,epoch)))
    temp1=torch.index_select(model.fc1.weight, 0, model.rowp1.cuda())
    model.fc1.weight=torch.nn.Parameter(torch.index_select(temp1, 1, model.colp1.cuda()))
    temp2=torch.index_select(model.fc2.weight, 0, model.rowp2.cuda())
    model.fc2.weight=torch.nn.Parameter(torch.index_select(temp2, 1, model.colp2.cuda()))
    temp3=torch.index_select(model.fc3.weight, 0, model.rowp3.cuda())
    model.fc3.weight=torch.nn.Parameter(torch.index_select(temp3, 1, model.colp3.cuda()))
    return model
def layer2torch(folder_path, model):
    for name, modules in model.named_children():
        torch.save(modules.weight.data,f"{folder_path}/{name}.txt")
def parameters2list(model):
    weight_list = []
    for module in model:
        dev = module.weight.device
        layer_weight = module.weight.data.cpu().numpy()
        shape = layer_weight.shape
        layer_weight = layer_weight.reshape(1, -1)
        for i in range(len(layer_weight)):
            mat = layer_weight[i].reshape(-1)
            if i == 0:
                weights = mat
            elif i == 1:
                weights = np.append([weights],[mat], axis=0)
            else:
                weights = np.append(weights,[mat], axis=0)
        weights = weights.reshape(shape)
        weight_list.append(weights)
    return weight_list

def save_parameters(file_folder, quantized_parameters):
    for i in range(len(quantized_parameters)):
        layer_weight = quantized_parameters[i]
        shape = layer_weight.shape
        if len(shape)>1:
            f = open(file_folder+'/layer_'+str(i)+'.txt','a')
            if len(shape)==2:
                for weight in layer_weight:
                    for j in range(len(weight)):
                        f.write(str(weight[j])+"\t")
                    f.write("\n")
            else:
                f.write(str(shape)+"\n")
                for filter_weight in layer_weight:
                    for weight in filter_weight:
                        weight = weight.reshape(-1)
                        for a_weight in weight:
                            f.write(str(a_weight)+"\t")
                        f.write("\n")
                    f.write("\n")

def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)

def print_nonzeros(model, log_file):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        log(log_file, f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    log(log_file, f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
def initial_train(model, args, train_loader, val_loader, tok, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    best_prec1=0
    # Define model
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    if tok == 'prune_re': 
        epochs = args.reepochs
        args.lr = 0.0001
    else:
        epochs = args.epochs
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    initial_optimizer_state_dict = optimizer.state_dict()
    optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer

    print('start epoch', args.start_epoch)
    print('epoch',epochs)
    for epoch in range(args.start_epoch, epochs):
        optimizer = adjust_learning_rate(optimizer, epoch, args)
        print('in_epoch',epoch)
        # train for one epoch
        train(train_loader, model, optimizer, epoch, args.alpha, args, tok)
        # evaluate on validation set
        prec1 = validate(val_loader, model, args)

    model = save_masked_checkpoint(model, tok, True, best_prec1, epochs, args)
    return model
def quantized_retrain(model, args, quantized_index_list, quantized_center_list, train_loader, val_loader, use_cuda=True):

    criterion = nn.CrossEntropyLoss().cuda()
    best_prec1=0
    args.lr = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.qauntize_epochs):
        # train for one epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to train mode
        model.train()
        end = time.time()
        optimizer = adjust_learning_rate(optimizer, epoch, args)
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()
            k=0

            for name, p in model.named_parameters():
                if args.model_mode == 'd' and 'conv' in name:
                    continue
                if args.model_mode == 'c' and 'fc' in name:
                    continue
                if 'mask' in name or 'bias' in name or 'bn' in name:
                    continue

                quantized_index = quantized_index_list[k]
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_center_array = []
                for j in range(2**args.bits):
                    grad_by_index = grad_tensor[quantized_index==j]
                    grad_center = np.mean(grad_by_index)
                    grad_center_array.append(grad_center)

                grad_center_array = np.array(grad_center_array)
                grad_tensor = grad_center_array[quantized_index]
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).cuda()
                k+=1

            optimizer.step()
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                '''print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1))'''
        # evaluate on validation set
        prec1 = validate(val_loader, model, args)
    model = save_masked_checkpoint(model,'quantized_re', True, best_prec1, args.reepochs, args)
    return model
def test(model, test_loader, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy
def validate(val_loader, model, args):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()
    penalty = nn.MSELoss(reduction='sum').cuda()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    print(' * Prec {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg
def fc_penalty(model, penalty):
    
    penalty_fc1=penalty_fc2=penalty_fc3=0
    for i in range(int(model.partition_size['fc1'])-1):
        penalty_fc1 += penalty(model.fc1.weight[i*model.block_row_size1:(i+1)*model.block_row_size1:1,i*model.block_col_size1:(i+1)*model.block_col_size1:1],
            model.fc1.weight[(i+1)*model.block_row_size1:(i+2)*model.block_row_size1:1,(i+1)*model.block_col_size1:(i+2)*model.block_col_size1:1])
    for i in range(int(model.partition_size['fc2'])-1):
        penalty_fc2 += penalty(model.fc2.weight[i*model.block_row_size2:(i+1)*model.block_row_size2:1,i*model.block_col_size2:(i+1)*model.block_col_size2:1],
            model.fc2.weight[(i+1)*model.block_row_size2:(i+2)*model.block_row_size2:1,(i+1)*model.block_col_size2:(i+2)*model.block_col_size2:1])
    for i in range(int(model.partition_size['fc3'])-1):
        penalty_fc3 += penalty(model.fc3.weight[i*model.block_row_size3:(i+1)*model.block_row_size3:1,i*model.block_col_size3:(i+1)*model.block_col_size3:1],
            model.fc3.weight[(i+1)*model.block_row_size3:(i+2)*model.block_row_size3:1,(i+1)*model.block_col_size3:(i+2)*model.block_col_size3:1])

    penalty_fc1=penalty_fc1/(int(model.partition_size['fc1'])-1)
    penalty_fc2=penalty_fc2/(int(model.partition_size['fc2'])-1)
    penalty_fc3=penalty_fc3/(int(model.partition_size['fc3'])-1)
    penalty=0.33*(penalty_fc1+penalty_fc2+penalty_fc3)
    return penalty.cuda()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def train(train_loader, model, optimizer, epoch, alpha, args, tok=""):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()

        if tok == "prune_re":
            for name, p in model.named_parameters():
                if args.model_mode == 'd' and 'conv' in name:
                    continue
                if 'mask' in name:
                    continue
                if 'bias' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).cuda()

        optimizer.step()
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

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
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer