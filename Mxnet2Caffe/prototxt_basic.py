# prototxt_basic
import sys
import pprint
from config import config
attrstr = "attrs"
#attrstr = "param"

def data(txt_file, info):
    txt_file.write('name: "mxnet-mdoel"\n')
    txt_file.write('layer {\n')
    txt_file.write('  name: "data"\n')
    txt_file.write('  type: "Input"\n')
    txt_file.write('  top: "data"\n')
    txt_file.write('  input_param {\n')
    txt_file.write('    shape: { dim: %d dim: %d dim: %d dim: %d }\n'
                    % (config.batchSize, config.channels, config.hight, config.width)) # TODO
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')

def fuzzy_haskey(d, key):
    for eachkey in d:
        if key in eachkey:
            return True
    return False
  
def Convolution(txt_file, info):
    #if info['attrs']['no_bias'] == 'True':
        #bias_term = 'false'
    #else:
        #bias_term = 'true'  
    #if info['top'] == 'conv1_1':
        #pprint.pprint(info)  
    if fuzzy_haskey(info['params'], 'bias'):
        bias_term = 'true'  
    elif 'no_bias' in info[attrstr] and info['attrs']['no_bias'] == 'True':
        bias_term = 'false'  
    else:
        bias_term = 'true'

    txt_file.write('layer {\n')
    txt_file.write('	bottom: "%s"\n'       % info['bottom'][0])
    txt_file.write('	top: "%s"\n'          % info['top'])
    txt_file.write('	name: "%s"\n'         % info['top'])
    txt_file.write('	type: "Convolution"\n')
    txt_file.write('	convolution_param {\n')
    txt_file.write('		num_output: %s\n'   % info[attrstr]['num_filter'])
    txt_file.write('		kernel_size: %s\n'  % info[attrstr]['kernel'].split('(')[1].split(',')[0]) # TODO
    if 'pad' in info[attrstr]:
        txt_file.write('		pad: %s\n'          % info[attrstr]['pad'].split('(')[1].split(',')[0]) # TODO
    if 'num_group' in info[attrstr]:
        txt_file.write('		group: %s\n'        % info[attrstr]['num_group'])
    if 'stride' in info[attrstr]:
        txt_file.write('		stride: %s\n'       % info[attrstr]['stride'].split('(')[1].split(',')[0])
    txt_file.write('		bias_term: %s\n'    % bias_term)
    txt_file.write('	}\n')
    if 'share' in info.keys() and info['share']:
        txt_file.write('	param {\n')
        txt_file.write('	  name: "%s"\n'     % info['params'][0])
        txt_file.write('	}\n')
    txt_file.write('}\n')
    txt_file.write('\n')

def ChannelwiseConvolution(txt_file, info):
    Convolution(txt_file, info)
  
def BatchNorm(txt_file, info):
    #pprint.pprint(info)
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
    txt_file.write('  top: "%s"\n'          % info['top'])
    txt_file.write('  name: "%s"\n'         % info['top'])
    txt_file.write('  type: "BatchNorm"\n')
    txt_file.write('  batch_norm_param {\n')
    txt_file.write('    use_global_stats: true\n')        # TODO
    if attrstr in info:
        if 'momentum' in info[attrstr]:
            txt_file.write('    moving_average_fraction: %s\n' % info[attrstr]['momentum'])
        else:
            txt_file.write('    moving_average_fraction: 0.9\n')

        if 'eps' in info[attrstr]:
            txt_file.write('    eps: %s\n' % info[attrstr]['eps'])
        else:
            txt_file.write('    eps: 0.001\n')
            txt_file.write('    eps: 0.001\n')  #?????
    else:
        txt_file.write('    moving_average_fraction: 0.9\n')
    txt_file.write('  }\n')
    txt_file.write('}\n')

    # if info['fix_gamma'] is "False":                    # TODO
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'       % info['top'])
    txt_file.write('  top: "%s"\n'          % info['top'])
    txt_file.write('  name: "%s_scale"\n'   % info['top'])
    txt_file.write('  type: "Scale"\n')
    txt_file.write('  scale_param { bias_term: true }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
    pass

def Activation(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
    txt_file.write('  top: "%s"\n'          % info['top'])
    txt_file.write('  name: "%s"\n'         % info['top'])
    if info[attrstr]['act_type']=='sigmoid':
        txt_file.write('  type: "Sigmoid"\n')
    else:
        txt_file.write('  type: "ReLU"\n')      # TODO
    txt_file.write('}\n')
    txt_file.write('\n')

def sigmoid(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Sigmoid"\n')
    txt_file.write('}\n')
    txt_file.write('\n')
	
def Concat(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n'         % info['top'])
    txt_file.write('  type: "Concat"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n'     % bottom_i)
    txt_file.write('  top: "%s"\n'          % info['top'])
    txt_file.write('}\n')
    txt_file.write('\n')
    pass
  
def ElementWiseSum(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n'         % info['top'])
    txt_file.write('  type: "Eltwise"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n'     % bottom_i)
    txt_file.write('  top: "%s"\n'          % info['top'])
    txt_file.write('  eltwise_param { operation: SUM }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
    pass

def Pooling(txt_file, info):
    pool_type = 'AVE' if info[attrstr]['pool_type'] == 'avg' else 'MAX'
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
    txt_file.write('  top: "%s"\n'          % info['top'])
    txt_file.write('  name: "%s"\n'         % info['top'])
    txt_file.write('  type: "Pooling"\n')
    txt_file.write('  pooling_param {\n')
    txt_file.write('    pool: %s\n'         % pool_type)       # TODO  
    if 'global_pool' in info[attrstr] and info[attrstr]['global_pool'] == 'True':
        txt_file.write('    global_pooling: true\n')
    else:
        txt_file.write('    kernel_size: %s\n'  % info[attrstr]['kernel'].split('(')[1].split(',')[0])

        if 'stride' in info[attrstr]:
            txt_file.write('    stride: %s\n'       % info[attrstr]['stride'].split('(')[1].split(',')[0])
        else:
	        txt_file.write('    stride: 1\n')

        if 'pad' in info[attrstr]:
	        txt_file.write('    pad: %s\n'          % info[attrstr]['pad'].split('(')[1].split(',')[0])

    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
    pass


def FullyConnected(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
    txt_file.write('  top: "%s"\n'        % info['top'])
    txt_file.write('  name: "%s"\n'       % info['top'])
    txt_file.write('  type: "InnerProduct"\n')
    txt_file.write('  inner_product_param {\n')
    txt_file.write('    num_output: %s\n' % info[attrstr]['num_hidden'])
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
    pass

def Flatten(txt_file, info):
    pass
  
def SoftmaxOutput(txt_file, info):
    pass

def LeakyReLU(txt_file, info):
    if info[attrstr]['act_type'] == 'elu':
        txt_file.write('layer {\n')
        txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
        txt_file.write('  top: "%s"\n'        % info['top'])
        txt_file.write('  name: "%s"\n'       % info['top'])
        txt_file.write('  type: "ELU"\n')
        txt_file.write('  elu_param { alpha: 0.25 }\n')
        txt_file.write('}\n')
        txt_file.write('\n')
    elif info[attrstr]['act_type'] == 'prelu':
        txt_file.write('layer {\n')
        txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
        txt_file.write('  top: "%s"\n'        % info['top'])
        txt_file.write('  name: "%s"\n'       % info['top'])
        txt_file.write('  type: "PReLU"\n')
        txt_file.write('}\n')
        txt_file.write('\n')      
    else:
        raise Exception("unsupported Activation")

def Eltwise(txt_file, info, op):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Eltwise"\n')
    txt_file.write('  top: "%s"\n'        % info['top'])
    txt_file.write('  name: "%s"\n'       % info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % btom)
    txt_file.write('  eltwise_param { operation: %s }\n' % op)
    txt_file.write('}\n')
    txt_file.write('\n')  

def LeakyReLU(txt_file, info):
    if info[attrstr]['act_type'] == 'elu':
        txt_file.write('layer {\n')
        txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
        txt_file.write('  top: "%s"\n'        % info['top'])
        txt_file.write('  name: "%s"\n'       % info['top'])
        txt_file.write('  type: "ELU"\n')
        txt_file.write('  elu_param { alpha: 0.25 }\n')
        txt_file.write('}\n')
        txt_file.write('\n')
    elif info[attrstr]['act_type'] == 'prelu':
        txt_file.write('layer {\n')
        txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
        txt_file.write('  top: "%s"\n'        % info['top'])
        txt_file.write('  name: "%s"\n'       % info['top'])
        txt_file.write('  type: "PReLU"\n')
        txt_file.write('}\n')
        txt_file.write('\n')
    else:
        raise Exception("unsupported Activation")

def Eltwise(txt_file, info, op):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Eltwise"\n')
    txt_file.write('  top: "%s"\n'        % info['top'])
    txt_file.write('  name: "%s"\n'       % info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % btom)
    txt_file.write('  eltwise_param { operation: %s }\n' % op)
    txt_file.write('}\n')
    txt_file.write('\n')

def Reshape(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Reshape"\n')
    txt_file.write('  top: "%s"\n'        %info['top'])
    txt_file.write('  name: "%s"\n'        %info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n'        %btom)
    txt_file.write('  reshape_param { shape {')
    for dim in info['attrs']['shape'].replace(',','').strip('(').strip(')').strip('/,').split(' '):
        txt_file.write('  dim: %s' %dim)
    txt_file.write('} }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
def SoftmaxActivation(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Softmax"\n')
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % btom)
    txt_file.write('}\n')
    txt_file.write('\n')
def Crop(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Crop"\n')
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % btom)
    txt_file.write('  crop_param { axis: 2 offset: 0}\n')
    txt_file.write('}\n')
    txt_file.write('\n')
def Deconvolution(txt_file, info):
    #if info['attrs']['no_bias'] == 'True':
    #bias_term = 'false'
    #else:
    #bias_term = 'true'
    #if info['top'] == 'conv1_1':
    #pprint.pprint(info)
    if fuzzy_haskey(info['params'], 'bias'):
        bias_term = 'true'
    elif info[attrstr]['no_bias'] and info['attrs']['no_bias'] == 'True':
        bias_term = 'false'
    else:
        bias_term = 'true'
    txt_file.write('layer {\n')
    txt_file.write('	bottom: "%s"\n'       % info['bottom'][0])
    txt_file.write('	top: "%s"\n'          % info['top'])
    txt_file.write('	name: "%s"\n'         % info['top'])
    txt_file.write('	type: "Convolution"\n')
    txt_file.write('	convolution_param {\n')
    txt_file.write('		num_output: %s\n'   % info[attrstr]['num_filter'])
    txt_file.write('		kernel_size: %s\n'  % info[attrstr]['kernel'].split('(')[1].split(',')[0]) # TODO
    if info[attrstr]['pad']:
        txt_file.write('		pad: %s\n'          % info[attrstr]['pad'].split('(')[1].split(',')[0]) # TODO
    if info[attrstr]['num_group']:
        txt_file.write('		group: %s\n'        % info[attrstr]['num_group'])
    if info[attrstr]['stride']:
        txt_file.write('		stride: %s\n'       % info[attrstr]['stride'].split('(')[1].split(',')[0])
    txt_file.write('		bias_term: %s\n'    % bias_term)
    txt_file.write('	}\n')
    if 'share' in info.keys() and info['share']:
        txt_file.write('	param {\n')
        txt_file.write('	  name: "%s"\n'     % info['params'][0])
        txt_file.write('	}\n')
    txt_file.write('}\n')
    txt_file.write('\n')
def Dropout(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Dropout"\n')
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % btom)
    txt_file.write('  dropout_param { dropout_ratio: %s}\n'%info[attrstr]["p"])
    txt_file.write('}\n')
    txt_file.write('\n')
def L2Normalization(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Normalize"\n')
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % btom)
    if info[attrstr]["mode"]:
        if info[attrstr]["mode"] == "instance":
            txt_file.write('  norm_param {\n')
            txt_file.write('    across_spatial: True\n')
            txt_file.write('    channel_shared: False\n')
            txt_file.write('  }\n')
        else:
            print("bn Normalization with %s is not supported\n")


    txt_file.write('}\n')
    txt_file.write('\n')
# ----------------------------------------------------------------
def write_node(txt_file, info):
    if 'label' in info['name']:
        return        
    if info['op'] == 'null' and info['name'] == 'data':
        data(txt_file, info)
    elif info['op'] == 'Convolution':
        Convolution(txt_file, info)
    elif info['op'] == 'ChannelwiseConvolution':
        ChannelwiseConvolution(txt_file, info)
    elif info['op'] == 'BatchNorm':
        BatchNorm(txt_file, info)
    elif info['op'] == 'Activation':
        Activation(txt_file, info)
    elif info['op'] == 'sigmoid':
        sigmoid(txt_file, info)
    elif info['op'] == 'ElementWiseSum':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'add_n':
        ElementWiseSum(txt_file, info)
    elif info['op'] == '_Plus':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'Concat':
        Concat(txt_file, info)
    elif info['op'] == 'Pooling':
        Pooling(txt_file, info)
    elif info['op'] == 'Flatten':
        Flatten(txt_file, info)
    elif info['op'] == 'FullyConnected':
        FullyConnected(txt_file, info)
    elif info['op'] == 'SoftmaxOutput':
        SoftmaxOutput(txt_file, info)
    elif info['op'] == 'LeakyReLU':
        LeakyReLU(txt_file, info)
    elif info['op'] == 'elemwise_add':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'Reshape':
        Reshape(txt_file, info)
    elif info['op'] == 'SoftmaxActivation':
        SoftmaxActivation(txt_file, info)
    elif info['op'] == "Crop":
        Crop(txt_file, info)
    elif info['op'] == "Deconvolution":
        Deconvolution(txt_file, info)
    elif info['op'] == "Dropout":
        Dropout(txt_file, info)
    elif info['op'] == "L2Normalization":
        L2Normalization(txt_file, info)
    else:
        #pprint.pprint(info)
        #sys.exit("Warning!  Unknown mxnet op:{}".format(info['op']))
        print ("Warning! Skip Unknown mxnet op:{}".format(info['op']))




