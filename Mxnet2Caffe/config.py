from easydict import EasyDict as edict

config = edict()
rootPath = '/kyle/workspace/project/tools/MXNet2Caffe/'
config.mxnetJson = rootPath + 'model_mxnet/resnet.json'
config.mxnetParam = rootPath + 'model_mxnet/fa-vgg16/fa-vgg16-nobn'

config.mxnetEpoch = 0
config.caffePrototxt = rootPath + 'model_caffe/resnet18.prototxt'
config.caffeModel = rootPath + 'model_caffe/fa-vgg16/fa-vgg16-nobn.caffemodel'

config.width = 224
config.hight = 224
config.channels = 3
config.batchSize = 1
