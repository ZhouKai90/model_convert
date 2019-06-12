import caffe
import numpy as np
import math
import cv2

caffe.set_device(0)
caffe.set_mode_gpu()
model_def = 'model_caffe/insightface_r34.prototxt'
model_weights = 'model_caffe/insightface_r34.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

Image_Path = 'test_model.jpg'
image = caffe.io.load_image(Image_Path)
heigh = image.shape[0]
width = image.shape[1]

net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([127.5, 127.5, 127.5]))
# transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_raw_scale('data', 255)
transformer.set_input_scale('data', 0.0078125)
# transformer.set_input_scale('data', 0.00390625)
# transformer.set_channel_swap('data', (2, 1, 0))
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

print("caffe input data:\n", transformed_image)

#print("start forword !")
net.forward()
#detections = net.blobs['detection_out'].data[...]
#print("xx")
detections = net.forward()['fc1']
sum = 0.
for i in detections[0]:
    sum = sum + i*i
sum = math.sqrt(sum)
sum = 1
print(detections/sum)