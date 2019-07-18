from easydict import EasyDict as edict

config = edict()
rootPath = '/xx/model_transform/Mxnet2Caffe/'
config.mxnetJson = rootPath + 'model_mxnet/deploy_ssd_vgg16_reduced_512-symbol.json'
config.mxnetParam = rootPath + 'model_mxnet/deploy_ssd_vgg16_reduced_512'

config.mxnetEpoch = 0
config.caffePrototxt = rootPath + 'model_caffe/ssd_vgg16_512.prototxt'
config.caffeModel = rootPath + 'model_caffe/ssd_vgg16_512.caffeModel'

config.width = 512
config.hight = 512
config.channels = 3
config.batchSize = 1
