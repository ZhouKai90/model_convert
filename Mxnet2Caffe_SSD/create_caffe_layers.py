# Copyright 2018 Argo AI, LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Generate caffe layer according to mxnet config.
"""

import constants

from ast import literal_eval
from caffe import layers
from caffe import params


def make_list(str_inp):
    """Create a list from a string of numbers.

    Args:
        str_inp (str): Expression to convert to list

    Returns:
        list: Converted list
    """
    val = literal_eval(str_inp)
    if type(val) is not tuple:
        val = [val]
    return list(val)


def get_caffe_layer(node, net, input_dims):
    """Generate caffe layer for corresponding mxnet op.

    Args:
        node (iterable from MxnetParser): Mxnet op summary generated by MxnetParser
        net (caffe.net): Caffe netspec object

    Returns:
        caffe.layers: Equivalent caffe layer
    """
    print(node)
    if node['type'] == 'Convolution':
        assert len(node['inputs']) == 1, \
            'Convolution layers can have only one input'
        conv_params = node['attrs']
        kernel_size = make_list(conv_params['kernel'])[0]
        num_filters = make_list(conv_params['num_filter'])[0]
        if 'stride' in conv_params:
            stride = make_list(conv_params['stride'])[0]
        else:
            stride = 1
        padding = make_list(conv_params['pad'])[0]
        if 'dilate' in conv_params:
            dilation = make_list(conv_params['dilate'])[0]
        else:
            dilation = 1
        convolution_param = {'pad': padding,
                             'kernel_size': kernel_size,
                             'num_output': num_filters,
                             'stride': stride,
                             'dilation': dilation}
        return layers.Convolution(net[node['inputs'][0]],
                                  convolution_param=convolution_param)
    if node['type'] == 'Activation':
        assert len(node['inputs']) == 1, \
            'Activation layers can have only one input'
        assert node['attrs']['act_type'] == 'relu'
        return layers.ReLU(net[node['inputs'][0]])

    if node['type'] == 'Pooling':
        assert len(node['inputs']) == 1, \
            'Pooling layers can have only one input'
        kernel_size = make_list(node['attrs']['kernel'])
        stride = make_list(node['attrs']['stride'])
        pooling_type = node['attrs']['pool_type']
        if 'pad' in node['attrs']:
            padding = make_list(node['attrs']['pad'])
        else:
            padding = [0]
        if pooling_type == 'max':
            pooling = params.Pooling.MAX
        elif pooling_type == 'avg':
            pooling = params.Pooling.AVE
        pooling_param = {'pool': pooling, 'pad': padding[0],
                         'kernel_size': kernel_size[0], 'stride': stride[0]}
        return layers.Pooling(net[node['inputs'][0]],
                              pooling_param=pooling_param)

    if node['type'] == 'L2Normalization':
        across_spatial = node['attrs']['mode'] != 'channel'
        channel_shared = False
        scale_filler = {
            'type': "constant",
            'value': constants.NORMALIZATION_FACTOR
        }
        norm_param = {'across_spatial': across_spatial,
                      'scale_filler': scale_filler,
                      'channel_shared': channel_shared}
        return layers.Normalize(net[node['inputs'][0]],
                                norm_param=norm_param)

    if node['type'] == 'BatchNorm':
        bn_param = {
                    'moving_average_fraction': 0.90,
                    'use_global_stats': True,
                    'eps': 1e-5
        }
        return layers.BatchNorm(net[node['inputs'][0]],
                                in_place=True, **bn_param)

    # Note - this layer has been implemented
    # only in WeiLiu's ssd branch of caffe not in caffe master
    if node['type'] == 'transpose':
        order = make_list(node['attrs']['axes'])
        return layers.Permute(net[node['inputs'][0]],
                              permute_param={'order': order})

    if node['type'] == 'Flatten':
        if node['inputs'][0].endswith('anchors'):
            axis = 2
        else:
            axis = 1
        return layers.Flatten(net[node['inputs'][0]],
                              flatten_param={'axis': axis})

    if node['type'] == 'Concat':
        # In the ssd model, always concatenate along last axis,
        # since anchor boxes have an extra dimension in caffe (that includes variance).
        axis = -1
        concat_inputs = [net[inp] for inp in node['inputs']]
        return layers.Concat(*concat_inputs, concat_param={'axis': axis})

    if node['type'] == 'Reshape':
        if node['name'] == 'multibox_anchors':
            reshape_dims = [1, 2, -1]
        else:
            reshape_dims = make_list(node['attrs']['shape'])
        return layers.Reshape(net[node['inputs'][0]],
                              reshape_param={'shape': {'dim': reshape_dims}})

    if node['type'] == '_contrib_MultiBoxPrior':
        priorbox_inputs = [net[inp] for inp in node['inputs']] + [net["data"]]
        sizes = make_list(node["attrs"]["sizes"])
        min_size = sizes[0] * input_dims[0]
        max_size = int(round((sizes[1] * input_dims[0]) ** 2 / min_size))
        aspect_ratio = make_list(node["attrs"]["ratios"])
        steps = make_list(node["attrs"]["steps"])
        param = {'clip': node["attrs"]["clip"] == "true",
                 'flip': False,
                 'min_size': int(round(min_size)),
                 'max_size': int(round(max_size)),
                 'aspect_ratio': aspect_ratio,
                 'variance': [.1, .1, .2, .2],
                 'step': int(round(steps[0] * input_dims[0])),
                 }
        return layers.PriorBox(*priorbox_inputs, prior_box_param=param)

    if node['type'] == '_contrib_MultiBoxDetection':
        multibox_inputs = [net[inp] for inp in node['inputs']]
        bottom_order = [1, 0, 2]
        multibox_inputs = [multibox_inputs[i] for i in bottom_order]
        param = {
            'num_classes': constants.NUM_CLASSES,
            'share_location': True,
            'background_label_id': 0,
            'nms_param': {
                'nms_threshold': float(node['attrs']['nms_threshold']),
                'top_k': int(node['attrs']['nms_topk'])
            },
            'keep_top_k': make_list(node['attrs']['nms_topk'])[0],
            'confidence_threshold': 0.01,
            'code_type': params.PriorBox.CENTER_SIZE,
        }
        return layers.DetectionOutput(*multibox_inputs, detection_output_param=param)

    if node['type'] in ['SoftmaxActivation', 'SoftmaxOutput']:
        if 'mode' not in node['attrs']:
            axis = 1
        elif node['attrs']['mode'] == 'channel':
            axis = 1
        else:
            axis = 0
        # note: caffe expects confidence scores to be flattened before detection output layer receives it
        return layers.Flatten(layers.Permute(layers.Softmax(net[node['inputs'][0]],
                                                            axis=axis),
                                             permute_param={'order': [0, 2, 1]}),
                              flatten_param={'axis': 1})
