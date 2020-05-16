import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import LeNet5_mask
import util
from net.quantization import apply_weight_sharing
import warnings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


parser = argparse.ArgumentParser(description='PyTorch MINST Training')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', '-ep', default=100, type=int, metavar='N',
                    help='number of total initial epochs to run (default: 250)')
parser.add_argument('--qauntize_epochs', '-qep', default=100, type=int, metavar='N',
                    help='number of quantize retrain epochs to run (default: 100)')
parser.add_argument('--reepochs', '-reep', default=20, type=int, metavar='N',
                    help='number of pruning retrain epochs to run (default: 20)')


parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-bs', default=256, type=int,metavar='N', 
                    help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-pf', default=1000, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# ------------- mask seed -----------------------------------------------------------
parser.add_argument('--mask-seed','-s',type=int, default=10, 
                    help='random seed (default: 10)')
# ------------- alpha ---------------------------------------------------------------
parser.add_argument('--alpha','-al', default=0.1, type=float, metavar='M', 
                    help='alpha(default=0.1)')

# ---------- partition size for fc1, fc2, fc3 --------------------------------------
parser.add_argument("--partition",'-p', dest="partition", 
                    action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='partition size of fc layer (eg. fc1=8,fc2=8,fc3=10)')

# ---------- quantize bits ---------------------------------------------------------
parser.add_argument('--bits', '-b',
                    action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='partition size of fc layer (eg. fc1=8,fc2=8,fc3=10)')

# ---------- log file --------------------------------------------------------------
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')

# ---------- base dir --------------------------------------------------------------
parser.add_argument('--save-dir','-sd', type=str,default='model_default',
                    help='model store path(defalut="model_default")')

# ---------- load file -------------------------------------------------------------
parser.add_argument('--load-model','-lm', type=str,default='checkpoint_default.tar',
                    help='load exist checkpoint file')

# ---------- train mode ------------------------------------------------------------ 
parser.add_argument('--train-mode', '-tm', default=1, type=int, metavar='M', 
                    help='1:initial training 2:load model+CNN pruning 3:load model+quantize'+
                          '4:initial+prune cnn+qauntize 5:initial+quantize 6:load model+pruning+quantize')
# ---------- dnn or cnn or all -----------------------------------------------------
parser.add_argument('--model-mode', '-mm', default='d', type=str, metavar='M', 
                    help='d:only qauntize dnn c:only qauntize cnn a:all qauntize\n')


parser.add_argument('--out-oldweight-folder', default='model_before_prune', type=str,
                    help='path to model output')
parser.add_argument('--out-pruned-folder', default='model_prune', type=str,
                    help='path to model output')
parser.add_argument('--out-pruned-re-folder', default='model_prune_re', type=str,
                    help='path to model output')
parser.add_argument('--out-quantized-folder', default='model_quantized', type=str,
                    help='path to model output')
parser.add_argument('--out-quantized-re-folder', default='model_quantized_retrain', type=str,
                    help='path to model output')

def main():
    print('this is alpha {}'.format(args.alpha))
    os.makedirs(f'{args.save_dir}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_oldweight_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_pruned_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_pruned_re_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_quantized_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_quantized_re_folder}', exist_ok=True)
    util.log(f"{args.save_dir}/{args.log}", "--------------------------configure----------------------")
    util.log(f"{args.save_dir}/{args.log}", f"{args}\n")
    if args.train_mode == 1:
        # Define model
        model = LeNet5_mask.LeNet5_mask('LeNet5_mask', args.partition,args.mask_seed, mask_flag=True).cuda()
        model = initial_process(model)
    elif args.train_mode == 2:
        if os.path.isfile(f"{args.load_model}"):
            print("-------load "+f"{args.load_model} ----")
            model = LeNet5_mask.LeNet5_mask('LeNet5_mask', args.partition,args.mask_seed, mask_flag=True).cuda()
            model = util.load_checkpoint(model,  f"{args.load_model}",args)
            model = pruning_process(model)
        else:
            print("---not found "+f"{args.load_model} ----")
    elif args.train_mode == 3:
        if os.path.isfile(f"{args.load_model}"):
            print("-------load "+f"{args.load_model} ----")
            model = LeNet5_mask.LeNet5_mask('LeNet5_mask', args.partition,args.mask_seed, mask_flag=True).cuda()
            model = util.load_checkpoint(model,  f"{args.load_model}",args)
            model = quantize_process(model)
        else:
            print("---not found "+f"{args.load_model} ----")
    elif args.train_mode == 4: # initial train/ prune cnn/ qauntize
        model = LeNet5_mask.LeNet5_mask('LeNet5_mask', args.partition,args.mask_seed, mask_flag=True).cuda()
        model = initial_process(model)
        model = pruning_process(model)
        quantize_process(model)
    elif args.train_mode == 5: # initial train/ qauntize
        model = LeNet5_mask.LeNet5_mask('LeNet5_mask', args.partition,args.mask_seed, mask_flag=True).cuda()
        model = initial_process(model)
        model = quantize_process(model)
    elif args.train_mode == 6: #load base model, prune and quantization
        if os.path.isfile(f"{args.load_model}"):
            print("-------load "+f"{args.load_model} ----")
            model = LeNet5_mask.LeNet5_mask('LeNet5_mask', args.partition,args.mask_seed, mask_flag=True).cuda()
            model = util.load_checkpoint(model, f"{args.load_model}",args)
            model = pruning_process(model)
            model = quantize_process(model)
        else:
            print("---not found "+f"{args.load_model} ----")
    elif args.train_mode == 7: #inference
        if os.path.isfile(f"{args.load_model}"):
            print("-------load "+f"{args.load_model} ----")
            model = LeNet5_mask.LeNet5_mask('LeNet5_mask', args.partition,args.mask_seed, mask_flag=True).cuda()
            model = util.load_checkpoint(model, f"{args.load_model}",args)
            inference(model)
        else:
            print("---not found "+f"{args.load_model} ----")
def inference(model):
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy = util.validate(val_loader, model, args)
    print(accuracy)

def initial_process(model):
    print(model)
    util.print_model_parameters(model)
    print("------------------------- Initial training -------------------------------")
    model = util.initial_train(model, args, train_loader, val_loader, 'initial')
    accuracy = util.validate(val_loader, model, args)

    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_oldweight_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"initial_accuracy\t{accuracy}")

    util.layer2torch(f"{args.save_dir}/{args.out_oldweight_folder}",model)
    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_oldweight_folder}", weight_list)
    return model

def pruning_process(model):

    print("------------------------- Before pruning --------------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy = util.validate(val_loader, model, args)

    print("------------------------- pruning CNN--------------------------------------")
    model.prune_by_percentile( ['conv1'], q=100-66.0)
    model.prune_by_percentile( ['conv2'], q=100-12.0)

    print("------------------------------- After prune CNN ----------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    prec1 = util.validate(val_loader, model, args)
    
    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_pruned_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model\t{args.save_dir}/model_pruned.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"prune acc\t{prec1}")
    
    util.layer2torch(f"{args.save_dir}/{args.out_pruned_folder}" , model)
    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_pruned_folder}", weight_list)
    
    print("------------------------- start retrain after prune CNN----------------------------")
    util.initial_train(model, args, train_loader, val_loader, 'prune_re')
    
    print("------------------------- After Retraining -----------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy = util.validate(val_loader, model, args)
    
    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_pruned_re_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model\t{args.save_dir}/model_prune_retrain_{args.reepochs}.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"prune and retrain acc\t{accuracy}")
    
    util.layer2torch(f"{args.save_dir}/{args.out_pruned_re_folder}" , model)
    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_pruned_re_folder}", weight_list)

    return model

def quantize_process(model):
    print('------------------------------- accuracy before weight sharing ----------------------------------')
    acc = util.validate(val_loader, model, args)
    util.log(f"{args.save_dir}/{args.log}", f"accuracy before weight sharing\t{acc}")

    print('------------------------------- accuacy after weight sharing -------------------------------')

    old_weight_list, new_weight_list, quantized_index_list, quantized_center_list = apply_weight_sharing(model, args.model_mode, args.bits)

    acc = util.validate(val_loader, model, args)

    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_quantized_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model\t{args.save_dir}/model_quantized.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"accuracy after weight sharing {args.bits}bits\t{acc}")

    util.layer2torch(f"{args.save_dir}/{args.out_quantized_folder}" , model)
    util.save_parameters(f"{args.save_dir}/{args.out_quantized_folder}", new_weight_list)
    
    print('------------------------------- retraining -------------------------------------------')

    util.quantized_retrain(model, args, quantized_index_list, quantized_center_list, train_loader, val_loader)

    acc = util.validate(val_loader, model, args)
    util.layer2torch(f"{args.save_dir}/{args.out_quantized_re_folder}" , model)

    util.log(f"{args.save_dir}/{args.log}", f"weight:{args.save_dir}/{args.out_quantized_re_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model:{args.save_dir}/model_quantized_retrain{args.reepochs}.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"acc after qauntize and retrain\t{acc}")

    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_quantized_re_folder}", weight_list)
    return model


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    global args, best_prec1, train_loade, val_loader
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    if use_cuda:
        print("Using CUDA!")
        torch.cuda.manual_seed(args.seed)
    else:
        print('Not using CUDA!!!')
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True,num_workers=args.workers,pin_memory=True)# no idea why I can't use data loader
    
    val_loader = torch.utils.data.DataLoader(
    
    datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=False,num_workers=args.workers,pin_memory=True)
    main()
