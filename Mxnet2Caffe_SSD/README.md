Mxnet2Caffe_SSD
================

将mxnet上训练的ssd模型转换为caffe模型

环境
-----------
* Mxnet : 1.4.0，需要编译安装Python接口，并且在`mxnet_to_caffe.py`文件中指定`caffe_converter`路径

* Caffe : 1.0 ，需要编译ssd分支Python接口，并设置PYTHONPATH中caffe的pthon路径。

  `git clone https://github.com/weiliu89/caffe.git`

  `cd caffe`

  `git checkout ssd`

  

运行
-----

拷贝mxnet模型到`mxnet_models`路径下，修改`run.sh`脚本






