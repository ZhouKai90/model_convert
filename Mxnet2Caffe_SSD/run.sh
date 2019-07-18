# !/bin/bash
python mxnet_to_caffe.py \
    --prefix mxnet_models/fa-resnet18-half \
    --caffe_prototxt caffe_models/fa-resnet18-half.prototxt \
    --caffemodel_name caffe_models/fa-resnet18-half.caffemodel