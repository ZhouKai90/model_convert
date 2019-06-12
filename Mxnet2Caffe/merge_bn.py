import numpy as np  
import sys,os  
# caffe_root = '/home/yaochuanqi/ssd/caffe/'
# sys.path.insert(0, caffe_root + 'python')  
import caffe
import caffe_parser

train_proto = 'model_caffe/symbol_ssh_faceboxes_nodense_nocrelu.prototxt'
train_model = 'model_caffe/symbol_ssh_faceboxes_nodense_nocrelu.caffemodel'  #should be your snapshot caffemodel

deploy_proto = 'model_caffe/symbol_ssh_faceboxes_nodense_nocrelu_nobn.prototxt'
save_model = 'model_caffe/symbol_ssh_faceboxes_nodense_nocrelu_nobn.caffemodel'

layers, names = caffe_parser.read_caffemodel(train_proto, train_model)
layer_iter = caffe_parser.layer_iter(layers, names)

def merge_bn(net, nob):
    '''
    merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
    var = var + scaleFacotr
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''
    # print(net.params.keys)
    param_names = []
    for key in net.params.iterkeys():
        param_names.append(key)
    print(param_names)

    for layer_name, layer_type, layer_blobs in layer_iter:
        print(layer_type, layer_name)
        if layer_type == 'Input' or layer_type == 'ReLU' or layer_type == 'Pooling' or layer_type == 'Split' or layer_type == 'Eltwise' or layer_type == 'Concat' or layer_type == 'Reshape' or layer_type == 'Softmax':
            continue
        elif layer_type == 'BatchNorm' or layer_type == 'Scale':
            continue
        elif layer_type == 'Convolution' and param_names.index(layer_name) <(len(param_names)-2):
            conv = net.params[layer_name]
            print("shape w ", conv[0].data[...].shape)
            bn_layer_name = param_names[param_names.index(layer_name)+1]
            scale_layer_name = param_names[param_names.index(layer_name)+2]
            if bn_layer_name.startswith("bn"):
                bn = net.params[bn_layer_name]
                scale = net.params[scale_layer_name]
                wt = conv[0].data
                channels = wt.shape[0]
                bias = np.zeros(wt.shape[0])
                if len(conv) > 1:
                    bias = conv[1].data
                mean = bn[0].data
                var = bn[1].data
                scalef = bn[2].data

                scales = scale[0].data
                shift = scale[1].data

                # print("mean:",mean)
                # print("var:", var)
                # print("scalef:", scalef)
                if scalef != 0:
                    scalef = 1. / scalef
                mean = mean * scalef
                var = var * scalef
                rstd = 1. / np.sqrt(var + 1e-5)
                rstd1 = rstd.reshape((channels,1,1,1))
                scales1 = scales.reshape((channels,1,1,1))
                wt = wt * rstd1 * scales1
                bias = (bias - mean) * rstd * scales + shift
                
                nob.params[layer_name][0].data[...] = wt
                nob.params[layer_name][1].data[...] = bias
            else:
                for i, w in enumerate(conv):
                    nob.params[layer_name][i].data[...] = w.data
        else:
            conv = net.params[layer_name]
            for i, w in enumerate(conv):
                nob.params[layer_name][i].data[...] = w.data




    # for key in net.params.iterkeys():
    #     print("key -> ",key)
    #     if type(net.params[key]) is caffe._caffe.BlobVec:
    #         print(type(net.params[key]))
    #         if key.startswith("bn") or key.startswith("scale"):
    #             # print("11",key)
    #             continue
    #         else:
    #             print("22",key)
    #             conv = net.params[key]
    #             if not net.params.has_key(key + "/bn"):
    #                 for i, w in enumerate(conv):
    #                     nob.params[key][i].data[...] = w.data
    #             else:
    #                 bn = net.params[key + "/bn"]
    #                 scale = net.params[key + "/scale"]
    #                 wt = conv[0].data
    #                 channels = wt.shape[0]
    #                 bias = np.zeros(wt.shape[0])
    #                 if len(conv) > 1:
    #                     bias = conv[1].data
    #                 mean = bn[0].data
    #                 var = bn[1].data
    #                 scalef = bn[2].data

    #                 scales = scale[0].data
    #                 shift = scale[1].data

    #                 print("mean:",mean)
    #                 print("var:", var)
    #                 print("scalef:", scalef)
    #                 if scalef != 0:
    #                     scalef = 1. / scalef
    #                 mean = mean * scalef
    #                 var = var * scalef
    #                 rstd = 1. / np.sqrt(var + 1e-5)
    #                 rstd1 = rstd.reshape((channels,1,1,1))
    #                 scales1 = scales.reshape((channels,1,1,1))
    #                 wt = wt * rstd1 * scales1
    #                 bias = (bias - mean) * rstd * scales + shift
                    
    #                 nob.params[key][0].data[...] = wt
    #                 nob.params[key][1].data[...] = bias
  

net = caffe.Net(train_proto, train_model, caffe.TRAIN)  
net_deploy = caffe.Net(deploy_proto, caffe.TEST)  

merge_bn(net, net_deploy)
net_deploy.save(save_model)

