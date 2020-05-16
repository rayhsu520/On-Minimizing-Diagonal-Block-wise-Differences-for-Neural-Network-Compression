import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import AlexNet
from net.quantization import apply_weight_sharing
import util

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch cifar pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')

parser.add_argument('--lr','-lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=128, type=int,
                    metavar='N', help='print frequency (default: 128)')

# --------- run mode ---------------------------------------------------------------------------------
parser.add_argument('--train-mode', '-tm',type=int, default=1, metavar='P',
                    help='1: initial training\n 2: pre training\n 3:prune and training\n 4:quantization training\n')

# --------- epochs -----------------------------------------------------------------------------------
parser.add_argument('--epochs', '-ep', default=300, type=int, metavar='N',
                    help='number of total epochs to run')

# --------- reepochs -----------------------------------------------------------------------------------
parser.add_argument('--reepochs','-reep', default=10, type=int, metavar='N',
                    help='number of total prune retrain epochs to run')

# ---------- quantize bits ---------------------------------------------------------------------------
parser.add_argument('--bits', '-b', action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='quantize bit (eg. conv=8,fc=5)')

# ---------- log file --------------------------------------------------------------------------------
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')

# ------------ save folder ---------------------------------------------------------------------------
parser.add_argument('--save-dir', '-sd', default='model_default', type=str,                                    
                    help='path to model output')                                                      
                                                                                                      
# ------------ load model ----------------------------------------------------------------------------
parser.add_argument('--load-model', '-lm', type=str, default='', metavar='P',   
                    help='load model file')                                                           
                                                                                                      
# ---------- dnn or cnn or all -----------------------------------------------------------------------
parser.add_argument('--model-mode', '-mm', default='d', type=str, metavar='M', 
                    help='d: dnn \n c:cnn \n a:all')

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
    os.makedirs(f'{args.save_dir}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_oldweight_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_pruned_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_pruned_re_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_quantized_folder}', exist_ok=True)
    os.makedirs(f'{args.save_dir}/{args.out_quantized_re_folder}', exist_ok=True)
    util.log(f"{args.save_dir}/{args.log}", "-------------------------configure------------------------")
    util.log(f"{args.save_dir}/{args.log}", f"{args}")
    torch.nn.Module.dump_patches = True
    if args.train_mode == 1:
        model = AlexNet(mask=True).to(device)
        model = initial_process(model)
    elif args.train_mode == 2:
        if os.path.isfile(f"{args.load_model}"):
            print("-------load "+f"{args.load_model} ----")
            model = torch.load(f"{args.load_model}")
            model = pruning_process(model)
        else:
            print("---not found "+f"{args.load_model} ----")
    elif args.train_mode == 3:
        if os.path.isfile(f"{args.load_model}"):
            print("-------load "+f"{args.load_model} ----")
            model = torch.load(f"{args.load_model}")
            model = quantize_process(model)
        else:
            print("---not found "+f"{args.load_model} ----")

    elif args.train_mode == 4: # whole Deep Compression
        model = AlexNet(mask=True).to(device)
        model = initial_process(model)
        model = pruning_process(model)
        quantize_process(model)

    elif args.train_mode == 5: #load base model, prune and quantization
        if os.path.isfile(f"{args.load_model}"):
            print("-------load "+f"{args.load_model} ----")
            model = torch.load(f"{args.load_model}")
            model = pruning_process(model)
            model = quantize_process(model)
        else:
            print("---not found "+f"{args.load_model} ----")

def initial_process(model):
    print(model)
    util.print_model_parameters(model)
    print("------------------------- Initial training -------------------------------")
    tok="initial" 
    criterion = nn.CrossEntropyLoss().cuda()
    util.initial_train(model, args, train_loader, test_loader, tok, use_cuda=True)
    accuracy = util.validate(args, test_loader, model, criterion)
    torch.save(model, f"{args.save_dir}/model_initial_end.ptmodel")

    util.log(f"{args.save_dir}/{args.log}", f"weight:{args.save_dir}/{args.out_oldweight_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model:{args.save_dir}/model_initial_end.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"initial_accuracy {accuracy}")

    util.layer2torch(model, f"{args.save_dir}/{args.out_oldweight_folder}")
    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_oldweight_folder}", weight_list)

    return model
def pruning_process(model):

    print("------------------------- Before pruning --------------------------------")
    criterion = nn.CrossEntropyLoss().cuda()
    accuracy = util.validate(args, test_loader, model, criterion)
    
    print("------------------------- pruning --------------------------------------")
    if args.model_mode == 'c' or args.model_mode =='a':
        model.prune_by_percentile( ['conv1'], q=100-84.0)
        model.prune_by_percentile( ['conv2'], q=100-38.0)
        model.prune_by_percentile( ['conv3'], q=100-35.0)
        model.prune_by_percentile( ['conv4'], q=100-37.0)
        model.prune_by_percentile( ['conv5'], q=100-37.0)
    if args.model_mode == 'd' or args.model_mode == 'a':
        model.prune_by_percentile( ['fc1'], q=100-9.0)
        model.prune_by_percentile( ['fc2'], q=100-9.0)
        model.prune_by_percentile( ['fc3'], q=100-25.0)
    
    print("------------------------- After pruning --------------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy = util.validate(args, test_loader, model, criterion)
    torch.save(model, f"{args.save_dir}/model_pruned.ptmodel")

    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_pruned_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model\t{args.save_dir}/model_pruned.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"accuracy after pruning\t{accuracy}")

    util.layer2torch(model, f"{args.save_dir}/{args.out_pruned_folder}")
    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_pruned_folder}", weight_list)

    # Retrain
    print("------------------------- Prune and Retrain ----------------------------")
    tok="prune_re"
    util.initial_train(model, args, train_loader, test_loader, tok, use_cuda=True)
    
    print("------------------------- After Retraining -----------------------------")
    util.print_nonzeros(model, f"{args.save_dir}/{args.log}")
    accuracy = util.validate(args, test_loader, model, criterion)
    torch.save(model, f"{args.save_dir}/model_prune_retrain_{args.reepochs}.ptmodel")

    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_pruned_re_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model\t{args.save_dir}/mmodel_prune_retrain_{args.reepochs}.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"accuracy after prune retrain\t{accuracy}")

    util.layer2torch(model, f"{args.save_dir}/{args.out_pruned_re_folder}")
    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_pruned_re_folder}", weight_list)



    return model
def quantize_process(model):

    print('---------------------- Before weight sharing ---------------------------')
    criterion = nn.CrossEntropyLoss().cuda()
    acc = util.validate(args, test_loader, model, criterion)
    util.log(f"{args.save_dir}/{args.log}", f"accuracy before weight sharing\t{acc}")

    # Weight sharing
    old_weight_list, new_weight_list, quantized_index_list, quantized_center_list = apply_weight_sharing(model, args.model_mode, args.bits)
    
    print('----------------------- After weight sharing ---------------------------')
    acc = util.validate(args, test_loader, model, criterion)
    torch.save(model, f"{args.save_dir}/model_quantized.ptmodel")

    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_quantized_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model\t{args.save_dir}/model_quantized.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"accuracy after weight sharing {args.bits}bits\t{acc}")

    util.layer2torch(model, f"{args.save_dir}/{args.out_quantized_folder}")
    util.save_parameters(f"{args.save_dir}/{args.out_quantized_folder}", new_weight_list)

    
    print('----------------------- quantize retrain -------------------------------')
    util.quantized_retrain(model, args, quantized_index_list, quantized_center_list, train_loader, use_cuda)

    acc = util.validate(args, test_loader, model, criterion)
    torch.save(model, f"{args.save_dir}/model_quantized_retrain{args.reepochs}.ptmodel")
    util.layer2torch(model, f"{args.save_dir}/{args.out_quantized_re_folder}")

    util.log(f"{args.save_dir}/{args.log}", f"weight\t{args.save_dir}/{args.out_quantized_re_folder}")
    util.log(f"{args.save_dir}/{args.log}", f"model\t{args.save_dir}/model_quantized_bit{args.bits}_retrain{args.reepochs}.ptmodel")
    util.log(f"{args.save_dir}/{args.log}", f"accuracy retrain after weight sharing\t{acc}")

    weight_list = util.parameters2list(model.children())
    util.save_parameters(f"{args.save_dir}/{args.out_quantized_re_folder}", weight_list)

    return model

if __name__ == '__main__':
    global args, train_loader, test_loader
    args = parser.parse_args()
    # Control Seed
    torch.manual_seed(args.seed)
    # Select Device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    if use_cuda:
        print("Using CUDA!")
        torch.cuda.manual_seed(args.seed)
    else:
        print('Not using CUDA!!!')
    # Loader
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    main()
