import argparse

import torch
import numpy
from numpy import array
from mpd.huffmancoding import huffman_encode_model,huffman_encode_editgraph
import mpd.util as util
import mpd.vgg_mask as vgg_mask

def to_index(fc):
    set_=numpy.unique(fc)
    dict_={}
    count_neg=0
    for i in range(len(set_)):
        if set_[i] <0:
            dict_[set_[i]] = len(set_)-1- i
            count_neg+=1
        elif set_[i] == 0:
            dict_[set_[i]] = 0
        else:
            dict_[set_[i]]=i-count_neg
    return numpy.vectorize(dict_.get)(fc)

def cycledistance(a,b,maxdis):
    distance=abs(a-b)
    if(a>b): #4->2
        if(distance<abs(maxdis-distance)):
            return (-1)*distance
        else:
            return abs(maxdis-distance)
    elif(a<b): #2->4
        if(distance<abs(maxdis-distance)):
            return distance
        else:
            return (-1)*abs(maxdis-distance)
    else:
        return 0


def matrix_cycledistance(array_a,array_b,size_x,size_y,maxdistance):
    distancearray=numpy.zeros(shape=(size_x,size_y))
    for i in range(size_x):
        for j in range(size_y):
            distancearray[i][j]=cycledistance(array_a[i][j],array_b[i][j],maxdistance)
    return distancearray
def conv_editgraph_and_firstblock(fc,x_axis, y_axis, partitionsize, maxdistance):
    edit_histogram=[]
    if partitionsize is not 1:	
        for i in range(partitionsize-1):
            if i==0:
                edit_histogram=matrix_cycledistance(
                     fc[i*(x_axis//partitionsize):(i+1)*(x_axis//partitionsize):1,:],
                     fc[(i+1)*(x_axis//partitionsize):(i+2)*(x_axis//partitionsize):1,:],
                     x_axis//partitionsize,
                     y_axis,
                     maxdistance
                )
            else:
                temp=matrix_cycledistance(
                     fc[i*(x_axis//partitionsize):(i+1)*(x_axis//partitionsize):1,:],
                     fc[(i+1)*(x_axis//partitionsize):(i+2)*(x_axis//partitionsize):1,:],
                     x_axis//partitionsize,
                     y_axis,
                     maxdistance
                )
                edit_histogram=numpy.concatenate((edit_histogram,temp),axis=0)
            
    else: edit_histogram=array(edit_histogram)
    first_block=[]
    for j in range(x_axis//partitionsize):
        for k in range(y_axis):
            first_block.append(fc[j][k])

    return edit_histogram, array(first_block)
def getblocks(fc,x_axis, y_axis, partitionsize, maxdistance):

    block_histogram=[]
    for i in range(partitionsize):
        blocks=[]
        for j in range(x_axis//partitionsize):
            for k in range(y_axis//partitionsize):
                blocks.append(fc[j][k])
        if i==0:
            block_histogram = array(blocks)
        else:
            block_histogram = numpy.concatenate((block_histogram,array(blocks)),axis=0)
    return block_histogram

def editgraph_and_firstblock(fc,x_axis, y_axis, partitionsize, maxdistance):
    edit_histogram=[]
    for i in range(partitionsize-1):
        if i==0:
            edit_histogram=matrix_cycledistance(fc[i*(x_axis//partitionsize):(i+1)*(x_axis//partitionsize):1,i*(y_axis//partitionsize):(i+1)*(y_axis//partitionsize):1],
            fc[(i+1)*(x_axis//partitionsize):(i+2)*(x_axis//partitionsize):1,(i+1)*(y_axis//partitionsize):(i+2)*(y_axis//partitionsize):1],
            x_axis//partitionsize,
            y_axis//partitionsize,
            maxdistance)
        else:
            temp=matrix_cycledistance(fc[i*(x_axis//partitionsize):(i+1)*(x_axis//partitionsize):1,i*(y_axis//partitionsize):(i+1)*(y_axis//partitionsize):1],
            fc[(i+1)*(x_axis//partitionsize):(i+2)*(x_axis//partitionsize):1,(i+1)*(y_axis//partitionsize):(i+2)*(y_axis//partitionsize):1],
            x_axis//partitionsize,
            y_axis//partitionsize,
            maxdistance)
            edit_histogram=numpy.concatenate((edit_histogram,temp),axis=0)
    #for first block
    first_block=[]
    for j in range(x_axis//partitionsize):
        for k in range(y_axis//partitionsize):
            first_block.append(fc[j][k])

    return edit_histogram, array(first_block)
def mpd_huffman_encode(val_loader ,model_path ,args):
    alpha = float(model_path.split("_")[4])
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    org_total = 0
    edit_total = 0
    compressed = 0
    compressed_edit = 0
    size = 0
    fc_d = 0
    fc_t = 0
    fc_compressed_without_edit = 0

    fc_org_total = 0
    fc_compressed = 0
    conv_org_total = 0
    conv_compressed = 0
    edit_distance_fc_list = []
    layer_compressed_dic={}
    layer_org_dic={}
    model = vgg_mask.VGG16('VGG16', args.partition, mask_flag=True).cuda()
    model = util.load_checkpoint(model,  f"{model_path}",args)

    util.log(f"{args.log_detail}", f"\n\n-----------------------------------------")
    util.log(f"{args.log_detail}", f"{model_path}")
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'bn' in name:
            continue
        if 'bias' in name:
            continue
        if 'conv' in name and args.train_mode is not 'd':
            fc_edit_bit=fc_edit_compress=fc_bit=fc_compressed_bit=0
            print(name)
            partition = int(args.partition[name[0:-7]])
            weight = param.data.cpu().numpy()
            fc_index = to_index(weight)
            fc_index = fc_index.reshape(-1,9)

            size += fc_index.size
            print(numpy.unique(fc_index))
            fc_edit, fc_first_block = conv_editgraph_and_firstblock(fc_index,numpy.size(fc_index,0),numpy.size(fc_index,1),partition,2**int(args.bits['conv']))

            fc_bit, fc_compressed_bit, t0, d0=huffman_encode_model(fc_first_block.astype(numpy.int32))
            org_total += fc_bit
            compressed += fc_compressed_bit
            conv_org_total += fc_bit
            conv_compressed += fc_compressed_bit
            util.log(f"{args.log_detail}", f"{name}")
            util.log(f"{args.log_detail}", f"\tfirst original:{fc_bit} bytes\t fist block after:{fc_compressed_bit} bytes")
            if partition is not 1:
                fc_edit_bit, fc_edit_compress, t0, d0=huffman_encode_model(fc_edit.astype(numpy.float32))
                edit_total += fc_edit_bit
                compressed_edit += fc_edit_compress
                conv_org_total += fc_edit_bit
                conv_compressed += fc_edit_compress
                util.log(f"{args.log_detail}", f"\tedit original:{fc_edit_bit} bytes\t edit block after:{fc_edit_compress} bytes")
            util.log(f"{args.log_detail}", f"original:{fc_bit+fc_edit_bit} bytes\t after:{fc_compressed_bit+fc_edit_compress} bytes")
        if 'fc' in name and args.train_mode is not 'c':
            print(name)
            partition = int(args.partition[name[0:3]])
            weight = param.data.cpu().numpy()
            fc_index = to_index(weight)
#----------- compress with editgraph format ------------------------------------------
            size += fc_index.size
            fc_edit, fc_first_block = editgraph_and_firstblock(fc_index,numpy.size(fc_index,0),numpy.size(fc_index,1),partition,2**int(args.bits['fc']))
            print('---------first block-----------')
            fc_bit, fc_compressed_bit, t0, d0=huffman_encode_model(fc_first_block.astype(numpy.int32))
            fc_t += t0
            fc_d += d0
            org_total += fc_bit
            compressed += fc_compressed_bit
            fc_org_total += fc_bit
            fc_compressed += fc_compressed_bit

            util.log(f"{args.log_detail}", f"{name}")
            util.log(f"{args.log_detail}", f"\tfirst original:{fc_bit} bytes\t fist block after:{fc_compressed_bit} bytes")
            print('---------edit block-----------')
            fc_edit_bit, fc_edit_compress, t0, d0=huffman_encode_model(fc_edit.astype(numpy.float32))
            fc_t+=t0
            fc_d+=d0
            edit_total += fc_edit_bit
            compressed_edit += fc_edit_compress
            fc_org_total += fc_edit_bit
            fc_compressed += fc_edit_compress
            util.log(f"{args.log_detail}", f"\tedit original:{fc_edit_bit} bytes\t edit block after:{fc_edit_compress} bytes")
            util.log(f"{args.log_detail}", f"original:{fc_bit+fc_edit_bit} bytes\t after:{fc_compressed_bit+fc_edit_compress} bytes")

#----------- compress without editgraph format ------------------------------------------
            size += fc_index.size
            blocks = getblocks(fc_index,numpy.size(fc_index,0),numpy.size(fc_index,1),partition,2**int(args.bits['fc']))
            print('---------without edit block-----------')
            fc_bit, fc_compressed_bit, t0, d0=huffman_encode_model(blocks.astype(numpy.int32))

            fc_compressed_without_edit += fc_compressed_bit
            layer_org_dic[name[0:3]] = fc_bit

#----------- count edit distance ------------------------------------------
            shape = weight.shape
            edit_distance=0
            for i in range(partition-1):
                for j in range(shape[0]//partition):
                    for k in range(shape[1]//partition):
                        if(numpy.absolute(weight[i*(shape[0]//partition)+j][i*(shape[1]//partition)+k]-weight[(i+1)*(shape[0]//partition)+j][(i+1)*(shape[1]//partition)+k])!=0):
                            edit_distance+=1
            edit_distance_fc_list.append(edit_distance)
#-------------------------------------------------------------------------

    print('first_block bytes org:{}'.format(org_total))
    print('first_block bytes after compression:{}'.format(compressed))
    print('edit_total bytes org:{}'.format(edit_total))
    print('edit_total bytes after compression:{}'.format(compressed_edit))
    
    org_total+=edit_total
    compressed+=compressed_edit
    util.log(f"{args.log}", f"\n\n------------------------------------")
    util.log(f"{args.log}", f"{model_path}")

    util.log(f"{args.log}", f"fc original total: {fc_org_total} bytes")
    util.log(f"{args.log}", f"fc compress total: {fc_compressed} bytes")
    util.log(f"{args.log}", f"conv original total: {conv_org_total} bytes")
    util.log(f"{args.log}", f"conv compress total: {conv_compressed} bytes")
    util.log(f"{args.log}", f"\noriginal total:{org_total} bytes")
    util.log(f"{args.log}", f"compress total:{compressed} bytes")
    util.log(f"{args.log}", f"compressed rate:{(compressed/org_total)*100}")
    util.log(f"{args.log}", f"average bit:{compressed*8/(size)} bits")

    print('compressed rate:{}'.format(compressed/org_total))
    print('original bytes:{}'.format(org_total))
    print('bytes after compression:{}'.format(compressed))
    print('average bit:{}'.format(compressed*8/size))

    acc = util.validate(val_loader, model, alpha)

    return acc, fc_org_total,  fc_compressed, fc_compressed_without_edit, edit_distance_fc_list,fc_t,fc_d, layer_compressed_dic, layer_org_dic
