nohup: ignoring input
I1118 03:13:19.409128  8554 caffe.cpp:99] Use GPU with device ID 0
I1118 03:13:20.531024  8554 caffe.cpp:107] Starting Optimization
I1118 03:13:20.531133  8554 solver.cpp:32] Initializing solver from parameters: 
test_iter: 121
test_interval: 50
base_lr: 1e-05
display: 1
max_iter: 5000
lr_policy: "fixed"
momentum: 0.9
weight_decay: 0.0005
snapshot: 1000
snapshot_prefix: "task/soil_high_o/none/"
solver_mode: GPU
test_compute_loss: true
net: "task/soil_high_o/train_val.prototxt"
I1118 03:13:20.531163  8554 solver.cpp:67] Creating training net from net file: task/soil_high_o/train_val.prototxt
I1118 03:13:20.531952  8554 net.cpp:275] The NetState phase (0) differed from the phase (1) specified by a rule in layer data
I1118 03:13:20.531996  8554 net.cpp:275] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy
I1118 03:13:20.532192  8554 net.cpp:39] Initializing net from parameters: 
name: "small"
layers {
  top: "data"
  top: "label"
  name: "data"
  type: IMAGE_DATA
  image_data_param {
    source: "/data/ad6813/devCaffe/caffe/data/soil_high_o/train.txt"
    batch_size: 32
    new_height: 256
    new_width: 256
  }
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 224
    mean_value: 86.752
    mean_value: 101.46
    mean_value: 104.6
  }
}
layers {
  bottom: "data"
  top: "conv1_1"
  name: "conv1_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv1_1"
  top: "conv1_1"
  name: "relu1_1"
  type: RELU
}
layers {
  bottom: "conv1_1"
  top: "conv1_2"
  name: "conv1_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv1_2"
  top: "conv1_2"
  name: "relu1_2"
  type: RELU
}
layers {
  bottom: "conv1_2"
  top: "pool1"
  name: "pool1"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool1"
  top: "conv2_1"
  name: "conv2_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv2_1"
  top: "conv2_1"
  name: "relu2_1"
  type: RELU
}
layers {
  bottom: "conv2_1"
  top: "conv2_2"
  name: "conv2_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv2_2"
  top: "conv2_2"
  name: "relu2_2"
  type: RELU
}
layers {
  bottom: "conv2_2"
  top: "pool2"
  name: "pool2"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool2"
  top: "conv3_1"
  name: "conv3_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv3_1"
  top: "conv3_1"
  name: "relu3_1"
  type: RELU
}
layers {
  bottom: "conv3_1"
  top: "conv3_2"
  name: "conv3_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv3_2"
  top: "conv3_2"
  name: "relu3_2"
  type: RELU
}
layers {
  bottom: "conv3_2"
  top: "conv3_3"
  name: "conv3_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv3_3"
  top: "conv3_3"
  name: "relu3_3"
  type: RELU
}
layers {
  bottom: "conv3_3"
  top: "pool3"
  name: "pool3"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool3"
  top: "conv4_1"
  name: "conv4_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv4_1"
  top: "conv4_1"
  name: "relu4_1"
  type: RELU
}
layers {
  bottom: "conv4_1"
  top: "conv4_2"
  name: "conv4_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv4_2"
  top: "conv4_2"
  name: "relu4_2"
  type: RELU
}
layers {
  bottom: "conv4_2"
  top: "conv4_3"
  name: "conv4_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv4_3"
  top: "conv4_3"
  name: "relu4_3"
  type: RELU
}
layers {
  bottom: "conv4_3"
  top: "pool4"
  name: "pool4"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool4"
  top: "conv5_1"
  name: "conv5_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv5_1"
  top: "conv5_1"
  name: "relu5_1"
  type: RELU
}
layers {
  bottom: "conv5_1"
  top: "conv5_2"
  name: "conv5_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv5_2"
  top: "conv5_2"
  name: "relu5_2"
  type: RELU
}
layers {
  bottom: "conv5_2"
  top: "conv5_3"
  name: "conv5_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv5_3"
  top: "conv5_3"
  name: "relu5_3"
  type: RELU
}
layers {
  bottom: "conv5_3"
  top: "pool5"
  name: "pool5"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool5"
  top: "fc6"
  name: "fc6"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "fc6"
  top: "fc6"
  name: "relu6"
  type: RELU
}
layers {
  bottom: "fc6"
  top: "fc6"
  name: "drop6"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc6"
  top: "fc7"
  name: "fc7"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "fc7"
  top: "fc7"
  name: "relu7"
  type: RELU
}
layers {
  bottom: "fc7"
  top: "fc7"
  name: "drop7"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc7"
  top: "fc8_2"
  name: "fc8_2"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "fc8_2"
  bottom: "label"
  name: "loss"
  type: SOFTMAX_LOSS
}
state {
  phase: TRAIN
}
I1118 03:13:20.532364  8554 layer_factory.hpp:78] Creating layer data
I1118 03:13:20.532388  8554 net.cpp:67] Creating Layer data
I1118 03:13:20.532403  8554 net.cpp:356] data -> data
I1118 03:13:20.532428  8554 net.cpp:356] data -> label
I1118 03:13:20.532445  8554 net.cpp:96] Setting up data
I1118 03:13:20.532457  8554 image_data_layer.cpp:34] Opening file /data/ad6813/devCaffe/caffe/data/soil_high_o/train.txt
I1118 03:13:20.600529  8554 image_data_layer.cpp:49] A total of 120113 images.
I1118 03:13:20.606034  8554 image_data_layer.cpp:78] output data size: 32,3,224,224
I1118 03:13:20.608945  8554 net.cpp:103] Top shape: 32 3 224 224 (4816896)
I1118 03:13:20.608971  8554 net.cpp:103] Top shape: 32 1 1 1 (32)
I1118 03:13:20.608976  8554 layer_factory.hpp:78] Creating layer conv1_1
I1118 03:13:20.608991  8554 net.cpp:67] Creating Layer conv1_1
I1118 03:13:20.608995  8554 net.cpp:394] conv1_1 <- data
I1118 03:13:20.609009  8554 net.cpp:356] conv1_1 -> conv1_1
I1118 03:13:20.609019  8554 net.cpp:96] Setting up conv1_1
I1118 03:13:20.800297  8554 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 03:13:20.800338  8554 layer_factory.hpp:78] Creating layer relu1_1
I1118 03:13:20.800348  8554 net.cpp:67] Creating Layer relu1_1
I1118 03:13:20.800354  8554 net.cpp:394] relu1_1 <- conv1_1
I1118 03:13:20.800360  8554 net.cpp:345] relu1_1 -> conv1_1 (in-place)
I1118 03:13:20.800369  8554 net.cpp:96] Setting up relu1_1
I1118 03:13:20.800376  8554 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 03:13:20.800380  8554 layer_factory.hpp:78] Creating layer conv1_2
I1118 03:13:20.800386  8554 net.cpp:67] Creating Layer conv1_2
I1118 03:13:20.800390  8554 net.cpp:394] conv1_2 <- conv1_1
I1118 03:13:20.800395  8554 net.cpp:356] conv1_2 -> conv1_2
I1118 03:13:20.800401  8554 net.cpp:96] Setting up conv1_2
I1118 03:13:20.801656  8554 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 03:13:20.801667  8554 layer_factory.hpp:78] Creating layer relu1_2
I1118 03:13:20.801676  8554 net.cpp:67] Creating Layer relu1_2
I1118 03:13:20.801678  8554 net.cpp:394] relu1_2 <- conv1_2
I1118 03:13:20.801683  8554 net.cpp:345] relu1_2 -> conv1_2 (in-place)
I1118 03:13:20.801688  8554 net.cpp:96] Setting up relu1_2
I1118 03:13:20.801693  8554 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 03:13:20.801697  8554 layer_factory.hpp:78] Creating layer pool1
I1118 03:13:20.801704  8554 net.cpp:67] Creating Layer pool1
I1118 03:13:20.801707  8554 net.cpp:394] pool1 <- conv1_2
I1118 03:13:20.801712  8554 net.cpp:356] pool1 -> pool1
I1118 03:13:20.801717  8554 net.cpp:96] Setting up pool1
I1118 03:13:20.801734  8554 net.cpp:103] Top shape: 32 64 112 112 (25690112)
I1118 03:13:20.801740  8554 layer_factory.hpp:78] Creating layer conv2_1
I1118 03:13:20.801746  8554 net.cpp:67] Creating Layer conv2_1
I1118 03:13:20.801749  8554 net.cpp:394] conv2_1 <- pool1
I1118 03:13:20.801755  8554 net.cpp:356] conv2_1 -> conv2_1
I1118 03:13:20.801761  8554 net.cpp:96] Setting up conv2_1
I1118 03:13:20.804059  8554 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 03:13:20.804071  8554 layer_factory.hpp:78] Creating layer relu2_1
I1118 03:13:20.804076  8554 net.cpp:67] Creating Layer relu2_1
I1118 03:13:20.804080  8554 net.cpp:394] relu2_1 <- conv2_1
I1118 03:13:20.804085  8554 net.cpp:345] relu2_1 -> conv2_1 (in-place)
I1118 03:13:20.804090  8554 net.cpp:96] Setting up relu2_1
I1118 03:13:20.804095  8554 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 03:13:20.804098  8554 layer_factory.hpp:78] Creating layer conv2_2
I1118 03:13:20.804105  8554 net.cpp:67] Creating Layer conv2_2
I1118 03:13:20.804107  8554 net.cpp:394] conv2_2 <- conv2_1
I1118 03:13:20.804112  8554 net.cpp:356] conv2_2 -> conv2_2
I1118 03:13:20.804118  8554 net.cpp:96] Setting up conv2_2
I1118 03:13:20.808634  8554 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 03:13:20.808645  8554 layer_factory.hpp:78] Creating layer relu2_2
I1118 03:13:20.808650  8554 net.cpp:67] Creating Layer relu2_2
I1118 03:13:20.808652  8554 net.cpp:394] relu2_2 <- conv2_2
I1118 03:13:20.808658  8554 net.cpp:345] relu2_2 -> conv2_2 (in-place)
I1118 03:13:20.808663  8554 net.cpp:96] Setting up relu2_2
I1118 03:13:20.808668  8554 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 03:13:20.808678  8554 layer_factory.hpp:78] Creating layer pool2
I1118 03:13:20.808683  8554 net.cpp:67] Creating Layer pool2
I1118 03:13:20.808686  8554 net.cpp:394] pool2 <- conv2_2
I1118 03:13:20.808691  8554 net.cpp:356] pool2 -> pool2
I1118 03:13:20.808696  8554 net.cpp:96] Setting up pool2
I1118 03:13:20.808701  8554 net.cpp:103] Top shape: 32 128 56 56 (12845056)
I1118 03:13:20.808704  8554 layer_factory.hpp:78] Creating layer conv3_1
I1118 03:13:20.808711  8554 net.cpp:67] Creating Layer conv3_1
I1118 03:13:20.808713  8554 net.cpp:394] conv3_1 <- pool2
I1118 03:13:20.808719  8554 net.cpp:356] conv3_1 -> conv3_1
I1118 03:13:20.808725  8554 net.cpp:96] Setting up conv3_1
I1118 03:13:20.817695  8554 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 03:13:20.817709  8554 layer_factory.hpp:78] Creating layer relu3_1
I1118 03:13:20.817715  8554 net.cpp:67] Creating Layer relu3_1
I1118 03:13:20.817718  8554 net.cpp:394] relu3_1 <- conv3_1
I1118 03:13:20.817723  8554 net.cpp:345] relu3_1 -> conv3_1 (in-place)
I1118 03:13:20.817729  8554 net.cpp:96] Setting up relu3_1
I1118 03:13:20.817734  8554 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 03:13:20.817737  8554 layer_factory.hpp:78] Creating layer conv3_2
I1118 03:13:20.817742  8554 net.cpp:67] Creating Layer conv3_2
I1118 03:13:20.817745  8554 net.cpp:394] conv3_2 <- conv3_1
I1118 03:13:20.817750  8554 net.cpp:356] conv3_2 -> conv3_2
I1118 03:13:20.817755  8554 net.cpp:96] Setting up conv3_2
I1118 03:13:20.835741  8554 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 03:13:20.835753  8554 layer_factory.hpp:78] Creating layer relu3_2
I1118 03:13:20.835758  8554 net.cpp:67] Creating Layer relu3_2
I1118 03:13:20.835762  8554 net.cpp:394] relu3_2 <- conv3_2
I1118 03:13:20.835768  8554 net.cpp:345] relu3_2 -> conv3_2 (in-place)
I1118 03:13:20.835774  8554 net.cpp:96] Setting up relu3_2
I1118 03:13:20.835779  8554 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 03:13:20.835783  8554 layer_factory.hpp:78] Creating layer conv3_3
I1118 03:13:20.835788  8554 net.cpp:67] Creating Layer conv3_3
I1118 03:13:20.835790  8554 net.cpp:394] conv3_3 <- conv3_2
I1118 03:13:20.835795  8554 net.cpp:356] conv3_3 -> conv3_3
I1118 03:13:20.835800  8554 net.cpp:96] Setting up conv3_3
I1118 03:13:20.853720  8554 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 03:13:20.853739  8554 layer_factory.hpp:78] Creating layer relu3_3
I1118 03:13:20.853749  8554 net.cpp:67] Creating Layer relu3_3
I1118 03:13:20.853754  8554 net.cpp:394] relu3_3 <- conv3_3
I1118 03:13:20.853760  8554 net.cpp:345] relu3_3 -> conv3_3 (in-place)
I1118 03:13:20.853767  8554 net.cpp:96] Setting up relu3_3
I1118 03:13:20.853773  8554 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 03:13:20.853776  8554 layer_factory.hpp:78] Creating layer pool3
I1118 03:13:20.853782  8554 net.cpp:67] Creating Layer pool3
I1118 03:13:20.853785  8554 net.cpp:394] pool3 <- conv3_3
I1118 03:13:20.853790  8554 net.cpp:356] pool3 -> pool3
I1118 03:13:20.853796  8554 net.cpp:96] Setting up pool3
I1118 03:13:20.853803  8554 net.cpp:103] Top shape: 32 256 28 28 (6422528)
I1118 03:13:20.853806  8554 layer_factory.hpp:78] Creating layer conv4_1
I1118 03:13:20.853811  8554 net.cpp:67] Creating Layer conv4_1
I1118 03:13:20.853814  8554 net.cpp:394] conv4_1 <- pool3
I1118 03:13:20.853821  8554 net.cpp:356] conv4_1 -> conv4_1
I1118 03:13:20.853826  8554 net.cpp:96] Setting up conv4_1
I1118 03:13:20.889020  8554 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 03:13:20.889039  8554 layer_factory.hpp:78] Creating layer relu4_1
I1118 03:13:20.889047  8554 net.cpp:67] Creating Layer relu4_1
I1118 03:13:20.889050  8554 net.cpp:394] relu4_1 <- conv4_1
I1118 03:13:20.889056  8554 net.cpp:345] relu4_1 -> conv4_1 (in-place)
I1118 03:13:20.889061  8554 net.cpp:96] Setting up relu4_1
I1118 03:13:20.889067  8554 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 03:13:20.889070  8554 layer_factory.hpp:78] Creating layer conv4_2
I1118 03:13:20.889075  8554 net.cpp:67] Creating Layer conv4_2
I1118 03:13:20.889078  8554 net.cpp:394] conv4_2 <- conv4_1
I1118 03:13:20.889092  8554 net.cpp:356] conv4_2 -> conv4_2
I1118 03:13:20.889098  8554 net.cpp:96] Setting up conv4_2
I1118 03:13:20.959712  8554 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 03:13:20.959739  8554 layer_factory.hpp:78] Creating layer relu4_2
I1118 03:13:20.959748  8554 net.cpp:67] Creating Layer relu4_2
I1118 03:13:20.959753  8554 net.cpp:394] relu4_2 <- conv4_2
I1118 03:13:20.959760  8554 net.cpp:345] relu4_2 -> conv4_2 (in-place)
I1118 03:13:20.959767  8554 net.cpp:96] Setting up relu4_2
I1118 03:13:20.959772  8554 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 03:13:20.959776  8554 layer_factory.hpp:78] Creating layer conv4_3
I1118 03:13:20.959782  8554 net.cpp:67] Creating Layer conv4_3
I1118 03:13:20.959785  8554 net.cpp:394] conv4_3 <- conv4_2
I1118 03:13:20.959791  8554 net.cpp:356] conv4_3 -> conv4_3
I1118 03:13:20.959797  8554 net.cpp:96] Setting up conv4_3
I1118 03:13:21.030145  8554 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 03:13:21.030169  8554 layer_factory.hpp:78] Creating layer relu4_3
I1118 03:13:21.030177  8554 net.cpp:67] Creating Layer relu4_3
I1118 03:13:21.030181  8554 net.cpp:394] relu4_3 <- conv4_3
I1118 03:13:21.030189  8554 net.cpp:345] relu4_3 -> conv4_3 (in-place)
I1118 03:13:21.030195  8554 net.cpp:96] Setting up relu4_3
I1118 03:13:21.030200  8554 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 03:13:21.030205  8554 layer_factory.hpp:78] Creating layer pool4
I1118 03:13:21.030210  8554 net.cpp:67] Creating Layer pool4
I1118 03:13:21.030213  8554 net.cpp:394] pool4 <- conv4_3
I1118 03:13:21.030218  8554 net.cpp:356] pool4 -> pool4
I1118 03:13:21.030225  8554 net.cpp:96] Setting up pool4
I1118 03:13:21.030231  8554 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 03:13:21.030235  8554 layer_factory.hpp:78] Creating layer conv5_1
I1118 03:13:21.030241  8554 net.cpp:67] Creating Layer conv5_1
I1118 03:13:21.030243  8554 net.cpp:394] conv5_1 <- pool4
I1118 03:13:21.030249  8554 net.cpp:356] conv5_1 -> conv5_1
I1118 03:13:21.030257  8554 net.cpp:96] Setting up conv5_1
I1118 03:13:21.100960  8554 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 03:13:21.100983  8554 layer_factory.hpp:78] Creating layer relu5_1
I1118 03:13:21.100991  8554 net.cpp:67] Creating Layer relu5_1
I1118 03:13:21.100996  8554 net.cpp:394] relu5_1 <- conv5_1
I1118 03:13:21.101003  8554 net.cpp:345] relu5_1 -> conv5_1 (in-place)
I1118 03:13:21.101009  8554 net.cpp:96] Setting up relu5_1
I1118 03:13:21.101016  8554 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 03:13:21.101018  8554 layer_factory.hpp:78] Creating layer conv5_2
I1118 03:13:21.101025  8554 net.cpp:67] Creating Layer conv5_2
I1118 03:13:21.101028  8554 net.cpp:394] conv5_2 <- conv5_1
I1118 03:13:21.101033  8554 net.cpp:356] conv5_2 -> conv5_2
I1118 03:13:21.101039  8554 net.cpp:96] Setting up conv5_2
I1118 03:13:21.171389  8554 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 03:13:21.171413  8554 layer_factory.hpp:78] Creating layer relu5_2
I1118 03:13:21.171422  8554 net.cpp:67] Creating Layer relu5_2
I1118 03:13:21.171427  8554 net.cpp:394] relu5_2 <- conv5_2
I1118 03:13:21.171434  8554 net.cpp:345] relu5_2 -> conv5_2 (in-place)
I1118 03:13:21.171442  8554 net.cpp:96] Setting up relu5_2
I1118 03:13:21.171447  8554 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 03:13:21.171450  8554 layer_factory.hpp:78] Creating layer conv5_3
I1118 03:13:21.171455  8554 net.cpp:67] Creating Layer conv5_3
I1118 03:13:21.171463  8554 net.cpp:394] conv5_3 <- conv5_2
I1118 03:13:21.171469  8554 net.cpp:356] conv5_3 -> conv5_3
I1118 03:13:21.171476  8554 net.cpp:96] Setting up conv5_3
I1118 03:13:21.242079  8554 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 03:13:21.242102  8554 layer_factory.hpp:78] Creating layer relu5_3
I1118 03:13:21.242110  8554 net.cpp:67] Creating Layer relu5_3
I1118 03:13:21.242115  8554 net.cpp:394] relu5_3 <- conv5_3
I1118 03:13:21.242121  8554 net.cpp:345] relu5_3 -> conv5_3 (in-place)
I1118 03:13:21.242127  8554 net.cpp:96] Setting up relu5_3
I1118 03:13:21.242133  8554 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 03:13:21.242146  8554 layer_factory.hpp:78] Creating layer pool5
I1118 03:13:21.242151  8554 net.cpp:67] Creating Layer pool5
I1118 03:13:21.242153  8554 net.cpp:394] pool5 <- conv5_3
I1118 03:13:21.242159  8554 net.cpp:356] pool5 -> pool5
I1118 03:13:21.242166  8554 net.cpp:96] Setting up pool5
I1118 03:13:21.242172  8554 net.cpp:103] Top shape: 32 512 7 7 (802816)
I1118 03:13:21.242177  8554 layer_factory.hpp:78] Creating layer fc6
I1118 03:13:21.242189  8554 net.cpp:67] Creating Layer fc6
I1118 03:13:21.242192  8554 net.cpp:394] fc6 <- pool5
I1118 03:13:21.242197  8554 net.cpp:356] fc6 -> fc6
I1118 03:13:21.242202  8554 net.cpp:96] Setting up fc6
I1118 03:13:23.795735  8554 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 03:13:23.795775  8554 layer_factory.hpp:78] Creating layer relu6
I1118 03:13:23.795784  8554 net.cpp:67] Creating Layer relu6
I1118 03:13:23.795789  8554 net.cpp:394] relu6 <- fc6
I1118 03:13:23.795796  8554 net.cpp:345] relu6 -> fc6 (in-place)
I1118 03:13:23.795804  8554 net.cpp:96] Setting up relu6
I1118 03:13:23.795817  8554 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 03:13:23.795820  8554 layer_factory.hpp:78] Creating layer drop6
I1118 03:13:23.795827  8554 net.cpp:67] Creating Layer drop6
I1118 03:13:23.795830  8554 net.cpp:394] drop6 <- fc6
I1118 03:13:23.795835  8554 net.cpp:345] drop6 -> fc6 (in-place)
I1118 03:13:23.795840  8554 net.cpp:96] Setting up drop6
I1118 03:13:23.795843  8554 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 03:13:23.795846  8554 layer_factory.hpp:78] Creating layer fc7
I1118 03:13:23.795851  8554 net.cpp:67] Creating Layer fc7
I1118 03:13:23.795855  8554 net.cpp:394] fc7 <- fc6
I1118 03:13:23.795861  8554 net.cpp:356] fc7 -> fc7
I1118 03:13:23.795866  8554 net.cpp:96] Setting up fc7
I1118 03:13:24.208925  8554 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 03:13:24.208964  8554 layer_factory.hpp:78] Creating layer relu7
I1118 03:13:24.208972  8554 net.cpp:67] Creating Layer relu7
I1118 03:13:24.208977  8554 net.cpp:394] relu7 <- fc7
I1118 03:13:24.208986  8554 net.cpp:345] relu7 -> fc7 (in-place)
I1118 03:13:24.208993  8554 net.cpp:96] Setting up relu7
I1118 03:13:24.209007  8554 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 03:13:24.209010  8554 layer_factory.hpp:78] Creating layer drop7
I1118 03:13:24.209015  8554 net.cpp:67] Creating Layer drop7
I1118 03:13:24.209018  8554 net.cpp:394] drop7 <- fc7
I1118 03:13:24.209022  8554 net.cpp:345] drop7 -> fc7 (in-place)
I1118 03:13:24.209027  8554 net.cpp:96] Setting up drop7
I1118 03:13:24.209030  8554 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 03:13:24.209033  8554 layer_factory.hpp:78] Creating layer fc8_2
I1118 03:13:24.209039  8554 net.cpp:67] Creating Layer fc8_2
I1118 03:13:24.209043  8554 net.cpp:394] fc8_2 <- fc7
I1118 03:13:24.209048  8554 net.cpp:356] fc8_2 -> fc8_2
I1118 03:13:24.209053  8554 net.cpp:96] Setting up fc8_2
I1118 03:13:24.209283  8554 net.cpp:103] Top shape: 32 2 1 1 (64)
I1118 03:13:24.209292  8554 layer_factory.hpp:78] Creating layer loss
I1118 03:13:24.209297  8554 net.cpp:67] Creating Layer loss
I1118 03:13:24.209300  8554 net.cpp:394] loss <- fc8_2
I1118 03:13:24.209305  8554 net.cpp:394] loss <- label
I1118 03:13:24.209314  8554 net.cpp:356] loss -> (automatic)
I1118 03:13:24.209319  8554 net.cpp:96] Setting up loss
I1118 03:13:24.209329  8554 net.cpp:103] Top shape: 1 1 1 1 (1)
I1118 03:13:24.209333  8554 net.cpp:109]     with loss weight 1
I1118 03:13:24.209362  8554 net.cpp:170] loss needs backward computation.
I1118 03:13:24.209367  8554 net.cpp:170] fc8_2 needs backward computation.
I1118 03:13:24.209368  8554 net.cpp:170] drop7 needs backward computation.
I1118 03:13:24.209372  8554 net.cpp:170] relu7 needs backward computation.
I1118 03:13:24.209373  8554 net.cpp:170] fc7 needs backward computation.
I1118 03:13:24.209377  8554 net.cpp:170] drop6 needs backward computation.
I1118 03:13:24.209379  8554 net.cpp:170] relu6 needs backward computation.
I1118 03:13:24.209381  8554 net.cpp:170] fc6 needs backward computation.
I1118 03:13:24.209393  8554 net.cpp:170] pool5 needs backward computation.
I1118 03:13:24.209396  8554 net.cpp:170] relu5_3 needs backward computation.
I1118 03:13:24.209398  8554 net.cpp:170] conv5_3 needs backward computation.
I1118 03:13:24.209401  8554 net.cpp:170] relu5_2 needs backward computation.
I1118 03:13:24.209404  8554 net.cpp:170] conv5_2 needs backward computation.
I1118 03:13:24.209408  8554 net.cpp:170] relu5_1 needs backward computation.
I1118 03:13:24.209410  8554 net.cpp:170] conv5_1 needs backward computation.
I1118 03:13:24.209414  8554 net.cpp:170] pool4 needs backward computation.
I1118 03:13:24.209415  8554 net.cpp:170] relu4_3 needs backward computation.
I1118 03:13:24.209419  8554 net.cpp:170] conv4_3 needs backward computation.
I1118 03:13:24.209421  8554 net.cpp:170] relu4_2 needs backward computation.
I1118 03:13:24.209424  8554 net.cpp:170] conv4_2 needs backward computation.
I1118 03:13:24.209427  8554 net.cpp:170] relu4_1 needs backward computation.
I1118 03:13:24.209429  8554 net.cpp:170] conv4_1 needs backward computation.
I1118 03:13:24.209432  8554 net.cpp:170] pool3 needs backward computation.
I1118 03:13:24.209435  8554 net.cpp:170] relu3_3 needs backward computation.
I1118 03:13:24.209439  8554 net.cpp:170] conv3_3 needs backward computation.
I1118 03:13:24.209441  8554 net.cpp:170] relu3_2 needs backward computation.
I1118 03:13:24.209444  8554 net.cpp:170] conv3_2 needs backward computation.
I1118 03:13:24.209446  8554 net.cpp:170] relu3_1 needs backward computation.
I1118 03:13:24.209450  8554 net.cpp:170] conv3_1 needs backward computation.
I1118 03:13:24.209452  8554 net.cpp:170] pool2 needs backward computation.
I1118 03:13:24.209455  8554 net.cpp:170] relu2_2 needs backward computation.
I1118 03:13:24.209457  8554 net.cpp:170] conv2_2 needs backward computation.
I1118 03:13:24.209460  8554 net.cpp:170] relu2_1 needs backward computation.
I1118 03:13:24.209463  8554 net.cpp:170] conv2_1 needs backward computation.
I1118 03:13:24.209466  8554 net.cpp:170] pool1 needs backward computation.
I1118 03:13:24.209468  8554 net.cpp:170] relu1_2 needs backward computation.
I1118 03:13:24.209471  8554 net.cpp:170] conv1_2 needs backward computation.
I1118 03:13:24.209475  8554 net.cpp:170] relu1_1 needs backward computation.
I1118 03:13:24.209476  8554 net.cpp:170] conv1_1 needs backward computation.
I1118 03:13:24.209480  8554 net.cpp:172] data does not need backward computation.
I1118 03:13:24.209497  8554 net.cpp:467] Collecting Learning Rate and Weight Decay.
I1118 03:13:24.209508  8554 net.cpp:219] Network initialization done.
I1118 03:13:24.209511  8554 net.cpp:220] Memory required for data: 3686465924
I1118 03:13:24.210307  8554 solver.cpp:151] Creating test net (#0) specified by net file: task/soil_high_o/train_val.prototxt
I1118 03:13:24.210355  8554 net.cpp:275] The NetState phase (1) differed from the phase (0) specified by a rule in layer data
I1118 03:13:24.210558  8554 net.cpp:39] Initializing net from parameters: 
name: "small"
layers {
  top: "data"
  top: "label"
  name: "data"
  type: IMAGE_DATA
  image_data_param {
    source: "/data/ad6813/devCaffe/caffe/data/soil_high_o/val.txt"
    batch_size: 8
    new_height: 256
    new_width: 256
  }
  include {
    phase: TEST
  }
  transform_param {
    mirror: true
    crop_size: 224
    mean_value: 86.752
    mean_value: 101.46
    mean_value: 104.6
  }
}
layers {
  bottom: "data"
  top: "conv1_1"
  name: "conv1_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv1_1"
  top: "conv1_1"
  name: "relu1_1"
  type: RELU
}
layers {
  bottom: "conv1_1"
  top: "conv1_2"
  name: "conv1_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv1_2"
  top: "conv1_2"
  name: "relu1_2"
  type: RELU
}
layers {
  bottom: "conv1_2"
  top: "pool1"
  name: "pool1"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool1"
  top: "conv2_1"
  name: "conv2_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv2_1"
  top: "conv2_1"
  name: "relu2_1"
  type: RELU
}
layers {
  bottom: "conv2_1"
  top: "conv2_2"
  name: "conv2_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv2_2"
  top: "conv2_2"
  name: "relu2_2"
  type: RELU
}
layers {
  bottom: "conv2_2"
  top: "pool2"
  name: "pool2"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool2"
  top: "conv3_1"
  name: "conv3_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv3_1"
  top: "conv3_1"
  name: "relu3_1"
  type: RELU
}
layers {
  bottom: "conv3_1"
  top: "conv3_2"
  name: "conv3_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv3_2"
  top: "conv3_2"
  name: "relu3_2"
  type: RELU
}
layers {
  bottom: "conv3_2"
  top: "conv3_3"
  name: "conv3_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv3_3"
  top: "conv3_3"
  name: "relu3_3"
  type: RELU
}
layers {
  bottom: "conv3_3"
  top: "pool3"
  name: "pool3"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool3"
  top: "conv4_1"
  name: "conv4_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv4_1"
  top: "conv4_1"
  name: "relu4_1"
  type: RELU
}
layers {
  bottom: "conv4_1"
  top: "conv4_2"
  name: "conv4_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv4_2"
  top: "conv4_2"
  name: "relu4_2"
  type: RELU
}
layers {
  bottom: "conv4_2"
  top: "conv4_3"
  name: "conv4_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv4_3"
  top: "conv4_3"
  name: "relu4_3"
  type: RELU
}
layers {
  bottom: "conv4_3"
  top: "pool4"
  name: "pool4"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool4"
  top: "conv5_1"
  name: "conv5_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv5_1"
  top: "conv5_1"
  name: "relu5_1"
  type: RELU
}
layers {
  bottom: "conv5_1"
  top: "conv5_2"
  name: "conv5_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv5_2"
  top: "conv5_2"
  name: "relu5_2"
  type: RELU
}
layers {
  bottom: "conv5_2"
  top: "conv5_3"
  name: "conv5_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "conv5_3"
  top: "conv5_3"
  name: "relu5_3"
  type: RELU
}
layers {
  bottom: "conv5_3"
  top: "pool5"
  name: "pool5"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "pool5"
  top: "fc6"
  name: "fc6"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "fc6"
  top: "fc6"
  name: "relu6"
  type: RELU
}
layers {
  bottom: "fc6"
  top: "fc6"
  name: "drop6"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc6"
  top: "fc7"
  name: "fc7"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "fc7"
  top: "fc7"
  name: "relu7"
  type: RELU
}
layers {
  bottom: "fc7"
  top: "fc7"
  name: "drop7"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc7"
  top: "fc8_2"
  name: "fc8_2"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "fc8_2"
  bottom: "label"
  name: "loss"
  type: SOFTMAX_LOSS
}
layers {
  bottom: "fc8_2"
  bottom: "label"
  top: "accuracy"
  name: "accuracy"
  type: PER_CLASS_ACCURACY
  include {
    phase: TEST
  }
}
state {
  phase: TEST
}
I1118 03:13:24.210683  8554 layer_factory.hpp:78] Creating layer data
I1118 03:13:24.210692  8554 net.cpp:67] Creating Layer data
I1118 03:13:24.210697  8554 net.cpp:356] data -> data
I1118 03:13:24.210705  8554 net.cpp:356] data -> label
I1118 03:13:24.210711  8554 net.cpp:96] Setting up data
I1118 03:13:24.210713  8554 image_data_layer.cpp:34] Opening file /data/ad6813/devCaffe/caffe/data/soil_high_o/val.txt
I1118 03:13:24.211071  8554 image_data_layer.cpp:49] A total of 967 images.
I1118 03:13:24.231654  8554 image_data_layer.cpp:78] output data size: 8,3,224,224
I1118 03:13:24.232189  8554 net.cpp:103] Top shape: 8 3 224 224 (1204224)
I1118 03:13:24.232210  8554 net.cpp:103] Top shape: 8 1 1 1 (8)
I1118 03:13:24.232214  8554 layer_factory.hpp:78] Creating layer label_data_1_split
I1118 03:13:24.232224  8554 net.cpp:67] Creating Layer label_data_1_split
I1118 03:13:24.232228  8554 net.cpp:394] label_data_1_split <- label
I1118 03:13:24.232234  8554 net.cpp:356] label_data_1_split -> label_data_1_split_0
I1118 03:13:24.232241  8554 net.cpp:356] label_data_1_split -> label_data_1_split_1
I1118 03:13:24.232246  8554 net.cpp:96] Setting up label_data_1_split
I1118 03:13:24.232251  8554 net.cpp:103] Top shape: 8 1 1 1 (8)
I1118 03:13:24.232254  8554 net.cpp:103] Top shape: 8 1 1 1 (8)
I1118 03:13:24.232257  8554 layer_factory.hpp:78] Creating layer conv1_1
I1118 03:13:24.232262  8554 net.cpp:67] Creating Layer conv1_1
I1118 03:13:24.232265  8554 net.cpp:394] conv1_1 <- data
I1118 03:13:24.232270  8554 net.cpp:356] conv1_1 -> conv1_1
I1118 03:13:24.232277  8554 net.cpp:96] Setting up conv1_1
I1118 03:13:24.232455  8554 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 03:13:24.232468  8554 layer_factory.hpp:78] Creating layer relu1_1
I1118 03:13:24.232476  8554 net.cpp:67] Creating Layer relu1_1
I1118 03:13:24.232480  8554 net.cpp:394] relu1_1 <- conv1_1
I1118 03:13:24.232484  8554 net.cpp:345] relu1_1 -> conv1_1 (in-place)
I1118 03:13:24.232494  8554 net.cpp:96] Setting up relu1_1
I1118 03:13:24.232499  8554 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 03:13:24.232502  8554 layer_factory.hpp:78] Creating layer conv1_2
I1118 03:13:24.232507  8554 net.cpp:67] Creating Layer conv1_2
I1118 03:13:24.232511  8554 net.cpp:394] conv1_2 <- conv1_1
I1118 03:13:24.232516  8554 net.cpp:356] conv1_2 -> conv1_2
I1118 03:13:24.232522  8554 net.cpp:96] Setting up conv1_2
I1118 03:13:24.233585  8554 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 03:13:24.233597  8554 layer_factory.hpp:78] Creating layer relu1_2
I1118 03:13:24.233603  8554 net.cpp:67] Creating Layer relu1_2
I1118 03:13:24.233605  8554 net.cpp:394] relu1_2 <- conv1_2
I1118 03:13:24.233609  8554 net.cpp:345] relu1_2 -> conv1_2 (in-place)
I1118 03:13:24.233614  8554 net.cpp:96] Setting up relu1_2
I1118 03:13:24.233619  8554 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 03:13:24.233623  8554 layer_factory.hpp:78] Creating layer pool1
I1118 03:13:24.233628  8554 net.cpp:67] Creating Layer pool1
I1118 03:13:24.233630  8554 net.cpp:394] pool1 <- conv1_2
I1118 03:13:24.233634  8554 net.cpp:356] pool1 -> pool1
I1118 03:13:24.233639  8554 net.cpp:96] Setting up pool1
I1118 03:13:24.233646  8554 net.cpp:103] Top shape: 8 64 112 112 (6422528)
I1118 03:13:24.233649  8554 layer_factory.hpp:78] Creating layer conv2_1
I1118 03:13:24.233654  8554 net.cpp:67] Creating Layer conv2_1
I1118 03:13:24.233656  8554 net.cpp:394] conv2_1 <- pool1
I1118 03:13:24.233661  8554 net.cpp:356] conv2_1 -> conv2_1
I1118 03:13:24.233665  8554 net.cpp:96] Setting up conv2_1
I1118 03:13:24.235841  8554 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 03:13:24.235854  8554 layer_factory.hpp:78] Creating layer relu2_1
I1118 03:13:24.235860  8554 net.cpp:67] Creating Layer relu2_1
I1118 03:13:24.235863  8554 net.cpp:394] relu2_1 <- conv2_1
I1118 03:13:24.235868  8554 net.cpp:345] relu2_1 -> conv2_1 (in-place)
I1118 03:13:24.235873  8554 net.cpp:96] Setting up relu2_1
I1118 03:13:24.235878  8554 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 03:13:24.235882  8554 layer_factory.hpp:78] Creating layer conv2_2
I1118 03:13:24.235887  8554 net.cpp:67] Creating Layer conv2_2
I1118 03:13:24.235889  8554 net.cpp:394] conv2_2 <- conv2_1
I1118 03:13:24.235895  8554 net.cpp:356] conv2_2 -> conv2_2
I1118 03:13:24.235901  8554 net.cpp:96] Setting up conv2_2
I1118 03:13:24.239949  8554 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 03:13:24.239960  8554 layer_factory.hpp:78] Creating layer relu2_2
I1118 03:13:24.239965  8554 net.cpp:67] Creating Layer relu2_2
I1118 03:13:24.239969  8554 net.cpp:394] relu2_2 <- conv2_2
I1118 03:13:24.239974  8554 net.cpp:345] relu2_2 -> conv2_2 (in-place)
I1118 03:13:24.239977  8554 net.cpp:96] Setting up relu2_2
I1118 03:13:24.239982  8554 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 03:13:24.239986  8554 layer_factory.hpp:78] Creating layer pool2
I1118 03:13:24.239991  8554 net.cpp:67] Creating Layer pool2
I1118 03:13:24.239995  8554 net.cpp:394] pool2 <- conv2_2
I1118 03:13:24.240000  8554 net.cpp:356] pool2 -> pool2
I1118 03:13:24.240005  8554 net.cpp:96] Setting up pool2
I1118 03:13:24.240010  8554 net.cpp:103] Top shape: 8 128 56 56 (3211264)
I1118 03:13:24.240013  8554 layer_factory.hpp:78] Creating layer conv3_1
I1118 03:13:24.240020  8554 net.cpp:67] Creating Layer conv3_1
I1118 03:13:24.240022  8554 net.cpp:394] conv3_1 <- pool2
I1118 03:13:24.240027  8554 net.cpp:356] conv3_1 -> conv3_1
I1118 03:13:24.240032  8554 net.cpp:96] Setting up conv3_1
I1118 03:13:24.248020  8554 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 03:13:24.248038  8554 layer_factory.hpp:78] Creating layer relu3_1
I1118 03:13:24.248042  8554 net.cpp:67] Creating Layer relu3_1
I1118 03:13:24.248045  8554 net.cpp:394] relu3_1 <- conv3_1
I1118 03:13:24.248050  8554 net.cpp:345] relu3_1 -> conv3_1 (in-place)
I1118 03:13:24.248054  8554 net.cpp:96] Setting up relu3_1
I1118 03:13:24.248060  8554 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 03:13:24.248064  8554 layer_factory.hpp:78] Creating layer conv3_2
I1118 03:13:24.248075  8554 net.cpp:67] Creating Layer conv3_2
I1118 03:13:24.248077  8554 net.cpp:394] conv3_2 <- conv3_1
I1118 03:13:24.248083  8554 net.cpp:356] conv3_2 -> conv3_2
I1118 03:13:24.248090  8554 net.cpp:96] Setting up conv3_2
I1118 03:13:24.263787  8554 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 03:13:24.263805  8554 layer_factory.hpp:78] Creating layer relu3_2
I1118 03:13:24.263811  8554 net.cpp:67] Creating Layer relu3_2
I1118 03:13:24.263815  8554 net.cpp:394] relu3_2 <- conv3_2
I1118 03:13:24.263820  8554 net.cpp:345] relu3_2 -> conv3_2 (in-place)
I1118 03:13:24.263839  8554 net.cpp:96] Setting up relu3_2
I1118 03:13:24.263844  8554 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 03:13:24.263847  8554 layer_factory.hpp:78] Creating layer conv3_3
I1118 03:13:24.263856  8554 net.cpp:67] Creating Layer conv3_3
I1118 03:13:24.263860  8554 net.cpp:394] conv3_3 <- conv3_2
I1118 03:13:24.263866  8554 net.cpp:356] conv3_3 -> conv3_3
I1118 03:13:24.263871  8554 net.cpp:96] Setting up conv3_3
I1118 03:13:24.279649  8554 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 03:13:24.279664  8554 layer_factory.hpp:78] Creating layer relu3_3
I1118 03:13:24.279670  8554 net.cpp:67] Creating Layer relu3_3
I1118 03:13:24.279674  8554 net.cpp:394] relu3_3 <- conv3_3
I1118 03:13:24.279680  8554 net.cpp:345] relu3_3 -> conv3_3 (in-place)
I1118 03:13:24.279685  8554 net.cpp:96] Setting up relu3_3
I1118 03:13:24.279690  8554 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 03:13:24.279695  8554 layer_factory.hpp:78] Creating layer pool3
I1118 03:13:24.279700  8554 net.cpp:67] Creating Layer pool3
I1118 03:13:24.279701  8554 net.cpp:394] pool3 <- conv3_3
I1118 03:13:24.279706  8554 net.cpp:356] pool3 -> pool3
I1118 03:13:24.279711  8554 net.cpp:96] Setting up pool3
I1118 03:13:24.279718  8554 net.cpp:103] Top shape: 8 256 28 28 (1605632)
I1118 03:13:24.279721  8554 layer_factory.hpp:78] Creating layer conv4_1
I1118 03:13:24.279728  8554 net.cpp:67] Creating Layer conv4_1
I1118 03:13:24.279731  8554 net.cpp:394] conv4_1 <- pool3
I1118 03:13:24.279736  8554 net.cpp:356] conv4_1 -> conv4_1
I1118 03:13:24.279743  8554 net.cpp:96] Setting up conv4_1
I1118 03:13:24.311238  8554 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 03:13:24.311261  8554 layer_factory.hpp:78] Creating layer relu4_1
I1118 03:13:24.311269  8554 net.cpp:67] Creating Layer relu4_1
I1118 03:13:24.311275  8554 net.cpp:394] relu4_1 <- conv4_1
I1118 03:13:24.311283  8554 net.cpp:345] relu4_1 -> conv4_1 (in-place)
I1118 03:13:24.311290  8554 net.cpp:96] Setting up relu4_1
I1118 03:13:24.311296  8554 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 03:13:24.311300  8554 layer_factory.hpp:78] Creating layer conv4_2
I1118 03:13:24.311305  8554 net.cpp:67] Creating Layer conv4_2
I1118 03:13:24.311308  8554 net.cpp:394] conv4_2 <- conv4_1
I1118 03:13:24.311312  8554 net.cpp:356] conv4_2 -> conv4_2
I1118 03:13:24.311318  8554 net.cpp:96] Setting up conv4_2
I1118 03:13:24.371323  8554 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 03:13:24.371366  8554 layer_factory.hpp:78] Creating layer relu4_2
I1118 03:13:24.371376  8554 net.cpp:67] Creating Layer relu4_2
I1118 03:13:24.371381  8554 net.cpp:394] relu4_2 <- conv4_2
I1118 03:13:24.371387  8554 net.cpp:345] relu4_2 -> conv4_2 (in-place)
I1118 03:13:24.371393  8554 net.cpp:96] Setting up relu4_2
I1118 03:13:24.371399  8554 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 03:13:24.371402  8554 layer_factory.hpp:78] Creating layer conv4_3
I1118 03:13:24.371409  8554 net.cpp:67] Creating Layer conv4_3
I1118 03:13:24.371412  8554 net.cpp:394] conv4_3 <- conv4_2
I1118 03:13:24.371417  8554 net.cpp:356] conv4_3 -> conv4_3
I1118 03:13:24.371425  8554 net.cpp:96] Setting up conv4_3
I1118 03:13:24.430866  8554 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 03:13:24.430903  8554 layer_factory.hpp:78] Creating layer relu4_3
I1118 03:13:24.430912  8554 net.cpp:67] Creating Layer relu4_3
I1118 03:13:24.430915  8554 net.cpp:394] relu4_3 <- conv4_3
I1118 03:13:24.430922  8554 net.cpp:345] relu4_3 -> conv4_3 (in-place)
I1118 03:13:24.430937  8554 net.cpp:96] Setting up relu4_3
I1118 03:13:24.430943  8554 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 03:13:24.430946  8554 layer_factory.hpp:78] Creating layer pool4
I1118 03:13:24.430953  8554 net.cpp:67] Creating Layer pool4
I1118 03:13:24.430955  8554 net.cpp:394] pool4 <- conv4_3
I1118 03:13:24.430960  8554 net.cpp:356] pool4 -> pool4
I1118 03:13:24.430966  8554 net.cpp:96] Setting up pool4
I1118 03:13:24.430974  8554 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 03:13:24.430977  8554 layer_factory.hpp:78] Creating layer conv5_1
I1118 03:13:24.430984  8554 net.cpp:67] Creating Layer conv5_1
I1118 03:13:24.430987  8554 net.cpp:394] conv5_1 <- pool4
I1118 03:13:24.430991  8554 net.cpp:356] conv5_1 -> conv5_1
I1118 03:13:24.430997  8554 net.cpp:96] Setting up conv5_1
I1118 03:13:24.490877  8554 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 03:13:24.490914  8554 layer_factory.hpp:78] Creating layer relu5_1
I1118 03:13:24.490921  8554 net.cpp:67] Creating Layer relu5_1
I1118 03:13:24.490926  8554 net.cpp:394] relu5_1 <- conv5_1
I1118 03:13:24.490932  8554 net.cpp:345] relu5_1 -> conv5_1 (in-place)
I1118 03:13:24.490938  8554 net.cpp:96] Setting up relu5_1
I1118 03:13:24.490944  8554 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 03:13:24.490947  8554 layer_factory.hpp:78] Creating layer conv5_2
I1118 03:13:24.490959  8554 net.cpp:67] Creating Layer conv5_2
I1118 03:13:24.490963  8554 net.cpp:394] conv5_2 <- conv5_1
I1118 03:13:24.490972  8554 net.cpp:356] conv5_2 -> conv5_2
I1118 03:13:24.490978  8554 net.cpp:96] Setting up conv5_2
I1118 03:13:24.550397  8554 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 03:13:24.550437  8554 layer_factory.hpp:78] Creating layer relu5_2
I1118 03:13:24.550446  8554 net.cpp:67] Creating Layer relu5_2
I1118 03:13:24.550451  8554 net.cpp:394] relu5_2 <- conv5_2
I1118 03:13:24.550457  8554 net.cpp:345] relu5_2 -> conv5_2 (in-place)
I1118 03:13:24.550462  8554 net.cpp:96] Setting up relu5_2
I1118 03:13:24.550468  8554 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 03:13:24.550472  8554 layer_factory.hpp:78] Creating layer conv5_3
I1118 03:13:24.550478  8554 net.cpp:67] Creating Layer conv5_3
I1118 03:13:24.550482  8554 net.cpp:394] conv5_3 <- conv5_2
I1118 03:13:24.550485  8554 net.cpp:356] conv5_3 -> conv5_3
I1118 03:13:24.550492  8554 net.cpp:96] Setting up conv5_3
I1118 03:13:24.610651  8554 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 03:13:24.610692  8554 layer_factory.hpp:78] Creating layer relu5_3
I1118 03:13:24.610702  8554 net.cpp:67] Creating Layer relu5_3
I1118 03:13:24.610707  8554 net.cpp:394] relu5_3 <- conv5_3
I1118 03:13:24.610713  8554 net.cpp:345] relu5_3 -> conv5_3 (in-place)
I1118 03:13:24.610719  8554 net.cpp:96] Setting up relu5_3
I1118 03:13:24.610724  8554 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 03:13:24.610728  8554 layer_factory.hpp:78] Creating layer pool5
I1118 03:13:24.610738  8554 net.cpp:67] Creating Layer pool5
I1118 03:13:24.610741  8554 net.cpp:394] pool5 <- conv5_3
I1118 03:13:24.610746  8554 net.cpp:356] pool5 -> pool5
I1118 03:13:24.610752  8554 net.cpp:96] Setting up pool5
I1118 03:13:24.610759  8554 net.cpp:103] Top shape: 8 512 7 7 (200704)
I1118 03:13:24.610762  8554 layer_factory.hpp:78] Creating layer fc6
I1118 03:13:24.610770  8554 net.cpp:67] Creating Layer fc6
I1118 03:13:24.610774  8554 net.cpp:394] fc6 <- pool5
I1118 03:13:24.610779  8554 net.cpp:356] fc6 -> fc6
I1118 03:13:24.610784  8554 net.cpp:96] Setting up fc6
I1118 03:13:27.197721  8554 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 03:13:27.197751  8554 layer_factory.hpp:78] Creating layer relu6
I1118 03:13:27.197758  8554 net.cpp:67] Creating Layer relu6
I1118 03:13:27.197763  8554 net.cpp:394] relu6 <- fc6
I1118 03:13:27.197770  8554 net.cpp:345] relu6 -> fc6 (in-place)
I1118 03:13:27.197777  8554 net.cpp:96] Setting up relu6
I1118 03:13:27.197792  8554 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 03:13:27.197795  8554 layer_factory.hpp:78] Creating layer drop6
I1118 03:13:27.197811  8554 net.cpp:67] Creating Layer drop6
I1118 03:13:27.197814  8554 net.cpp:394] drop6 <- fc6
I1118 03:13:27.197823  8554 net.cpp:345] drop6 -> fc6 (in-place)
I1118 03:13:27.197828  8554 net.cpp:96] Setting up drop6
I1118 03:13:27.197831  8554 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 03:13:27.197834  8554 layer_factory.hpp:78] Creating layer fc7
I1118 03:13:27.197839  8554 net.cpp:67] Creating Layer fc7
I1118 03:13:27.197842  8554 net.cpp:394] fc7 <- fc6
I1118 03:13:27.197846  8554 net.cpp:356] fc7 -> fc7
I1118 03:13:27.197852  8554 net.cpp:96] Setting up fc7
I1118 03:13:27.631762  8554 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 03:13:27.631790  8554 layer_factory.hpp:78] Creating layer relu7
I1118 03:13:27.631798  8554 net.cpp:67] Creating Layer relu7
I1118 03:13:27.631803  8554 net.cpp:394] relu7 <- fc7
I1118 03:13:27.631809  8554 net.cpp:345] relu7 -> fc7 (in-place)
I1118 03:13:27.631815  8554 net.cpp:96] Setting up relu7
I1118 03:13:27.631829  8554 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 03:13:27.631832  8554 layer_factory.hpp:78] Creating layer drop7
I1118 03:13:27.631839  8554 net.cpp:67] Creating Layer drop7
I1118 03:13:27.631842  8554 net.cpp:394] drop7 <- fc7
I1118 03:13:27.631846  8554 net.cpp:345] drop7 -> fc7 (in-place)
I1118 03:13:27.631850  8554 net.cpp:96] Setting up drop7
I1118 03:13:27.631855  8554 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 03:13:27.631857  8554 layer_factory.hpp:78] Creating layer fc8_2
I1118 03:13:27.631863  8554 net.cpp:67] Creating Layer fc8_2
I1118 03:13:27.631866  8554 net.cpp:394] fc8_2 <- fc7
I1118 03:13:27.631871  8554 net.cpp:356] fc8_2 -> fc8_2
I1118 03:13:27.631876  8554 net.cpp:96] Setting up fc8_2
I1118 03:13:27.632100  8554 net.cpp:103] Top shape: 8 2 1 1 (16)
I1118 03:13:27.632108  8554 layer_factory.hpp:78] Creating layer fc8_2_fc8_2_0_split
I1118 03:13:27.632115  8554 net.cpp:67] Creating Layer fc8_2_fc8_2_0_split
I1118 03:13:27.632118  8554 net.cpp:394] fc8_2_fc8_2_0_split <- fc8_2
I1118 03:13:27.632122  8554 net.cpp:356] fc8_2_fc8_2_0_split -> fc8_2_fc8_2_0_split_0
I1118 03:13:27.632128  8554 net.cpp:356] fc8_2_fc8_2_0_split -> fc8_2_fc8_2_0_split_1
I1118 03:13:27.632133  8554 net.cpp:96] Setting up fc8_2_fc8_2_0_split
I1118 03:13:27.632141  8554 net.cpp:103] Top shape: 8 2 1 1 (16)
I1118 03:13:27.632144  8554 net.cpp:103] Top shape: 8 2 1 1 (16)
I1118 03:13:27.632148  8554 layer_factory.hpp:78] Creating layer loss
I1118 03:13:27.632153  8554 net.cpp:67] Creating Layer loss
I1118 03:13:27.632156  8554 net.cpp:394] loss <- fc8_2_fc8_2_0_split_0
I1118 03:13:27.632160  8554 net.cpp:394] loss <- label_data_1_split_0
I1118 03:13:27.632164  8554 net.cpp:356] loss -> (automatic)
I1118 03:13:27.632169  8554 net.cpp:96] Setting up loss
I1118 03:13:27.632174  8554 net.cpp:103] Top shape: 1 1 1 1 (1)
I1118 03:13:27.632177  8554 net.cpp:109]     with loss weight 1
I1118 03:13:27.632192  8554 layer_factory.hpp:78] Creating layer accuracy
I1118 03:13:27.632199  8554 net.cpp:67] Creating Layer accuracy
I1118 03:13:27.632201  8554 net.cpp:394] accuracy <- fc8_2_fc8_2_0_split_1
I1118 03:13:27.632205  8554 net.cpp:394] accuracy <- label_data_1_split_1
I1118 03:13:27.632210  8554 net.cpp:356] accuracy -> accuracy
I1118 03:13:27.632215  8554 net.cpp:96] Setting up accuracy
I1118 03:13:27.632225  8554 net.cpp:103] Top shape: 1 1 1 4 (4)
I1118 03:13:27.632228  8554 net.cpp:172] accuracy does not need backward computation.
I1118 03:13:27.632231  8554 net.cpp:170] loss needs backward computation.
I1118 03:13:27.632235  8554 net.cpp:170] fc8_2_fc8_2_0_split needs backward computation.
I1118 03:13:27.632237  8554 net.cpp:170] fc8_2 needs backward computation.
I1118 03:13:27.632239  8554 net.cpp:170] drop7 needs backward computation.
I1118 03:13:27.632242  8554 net.cpp:170] relu7 needs backward computation.
I1118 03:13:27.632244  8554 net.cpp:170] fc7 needs backward computation.
I1118 03:13:27.632247  8554 net.cpp:170] drop6 needs backward computation.
I1118 03:13:27.632249  8554 net.cpp:170] relu6 needs backward computation.
I1118 03:13:27.632261  8554 net.cpp:170] fc6 needs backward computation.
I1118 03:13:27.632263  8554 net.cpp:170] pool5 needs backward computation.
I1118 03:13:27.632266  8554 net.cpp:170] relu5_3 needs backward computation.
I1118 03:13:27.632269  8554 net.cpp:170] conv5_3 needs backward computation.
I1118 03:13:27.632272  8554 net.cpp:170] relu5_2 needs backward computation.
I1118 03:13:27.632275  8554 net.cpp:170] conv5_2 needs backward computation.
I1118 03:13:27.632277  8554 net.cpp:170] relu5_1 needs backward computation.
I1118 03:13:27.632280  8554 net.cpp:170] conv5_1 needs backward computation.
I1118 03:13:27.632283  8554 net.cpp:170] pool4 needs backward computation.
I1118 03:13:27.632287  8554 net.cpp:170] relu4_3 needs backward computation.
I1118 03:13:27.632288  8554 net.cpp:170] conv4_3 needs backward computation.
I1118 03:13:27.632292  8554 net.cpp:170] relu4_2 needs backward computation.
I1118 03:13:27.632294  8554 net.cpp:170] conv4_2 needs backward computation.
I1118 03:13:27.632297  8554 net.cpp:170] relu4_1 needs backward computation.
I1118 03:13:27.632299  8554 net.cpp:170] conv4_1 needs backward computation.
I1118 03:13:27.632302  8554 net.cpp:170] pool3 needs backward computation.
I1118 03:13:27.632305  8554 net.cpp:170] relu3_3 needs backward computation.
I1118 03:13:27.632308  8554 net.cpp:170] conv3_3 needs backward computation.
I1118 03:13:27.632310  8554 net.cpp:170] relu3_2 needs backward computation.
I1118 03:13:27.632313  8554 net.cpp:170] conv3_2 needs backward computation.
I1118 03:13:27.632315  8554 net.cpp:170] relu3_1 needs backward computation.
I1118 03:13:27.632318  8554 net.cpp:170] conv3_1 needs backward computation.
I1118 03:13:27.632321  8554 net.cpp:170] pool2 needs backward computation.
I1118 03:13:27.632323  8554 net.cpp:170] relu2_2 needs backward computation.
I1118 03:13:27.632326  8554 net.cpp:170] conv2_2 needs backward computation.
I1118 03:13:27.632329  8554 net.cpp:170] relu2_1 needs backward computation.
I1118 03:13:27.632331  8554 net.cpp:170] conv2_1 needs backward computation.
I1118 03:13:27.632334  8554 net.cpp:170] pool1 needs backward computation.
I1118 03:13:27.632338  8554 net.cpp:170] relu1_2 needs backward computation.
I1118 03:13:27.632339  8554 net.cpp:170] conv1_2 needs backward computation.
I1118 03:13:27.632343  8554 net.cpp:170] relu1_1 needs backward computation.
I1118 03:13:27.632344  8554 net.cpp:170] conv1_1 needs backward computation.
I1118 03:13:27.632347  8554 net.cpp:172] label_data_1_split does not need backward computation.
I1118 03:13:27.632350  8554 net.cpp:172] data does not need backward computation.
I1118 03:13:27.632352  8554 net.cpp:208] This network produces output accuracy
I1118 03:13:27.632374  8554 net.cpp:467] Collecting Learning Rate and Weight Decay.
I1118 03:13:27.632380  8554 net.cpp:219] Network initialization done.
I1118 03:13:27.632385  8554 net.cpp:220] Memory required for data: 921616692
I1118 03:13:27.632483  8554 solver.cpp:41] Solver scaffolding done.
I1118 03:13:27.632488  8554 caffe.cpp:115] Finetuning from oxford/small.weights
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:505] Reading dangerously large protocol message.  If the message turns out to be larger than 1073741824 bytes, parsing will be halted for security reasons.  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:78] The total number of bytes read was 553432081
I1118 03:13:28.415824  8554 solver.cpp:160] Solving small
I1118 03:13:28.415849  8554 solver.cpp:161] Learning Rate Policy: fixed
I1118 03:13:28.415896  8554 solver.cpp:264] Iteration 0, Testing net (#0)
I1118 03:13:49.261961  8554 solver.cpp:305] Test loss: 0.784506
I1118 03:13:49.262002  8554 solver.cpp:318] mean_score = test_score[0] { = 432} / test_score[1] { = 868 }
I1118 03:13:49.262011  8554 solver.cpp:319]            = 0.497696
I1118 03:13:49.262022  8554 solver.cpp:328]     Test net output #0: accuracy = 0.497696
I1118 03:13:49.262027  8554 solver.cpp:318] mean_score = test_score[2] { = 55} / test_score[3] { = 100 }
I1118 03:13:49.262032  8554 solver.cpp:319]            = 0.55
I1118 03:13:49.262035  8554 solver.cpp:328]     Test net output #1: accuracy = 0.55
I1118 03:13:49.262045  8554 solver.cpp:332]     Test net output #2: accuracy = 0.503099
I1118 03:13:49.262050  8554 solver.cpp:334]     Test net output #3: accuracy = 0.523848
I1118 03:13:49.889580  8554 solver.cpp:209] Iteration 0, loss = 1.05786
I1118 03:13:49.889669  8554 solver.cpp:464] Iteration 0, lr = 1e-05
I1118 03:13:51.974366  8554 solver.cpp:209] Iteration 1, loss = 0.933594
I1118 03:13:51.974406  8554 solver.cpp:464] Iteration 1, lr = 1e-05
I1118 03:13:54.042727  8554 solver.cpp:209] Iteration 2, loss = 0.705828
I1118 03:13:54.042768  8554 solver.cpp:464] Iteration 2, lr = 1e-05
I1118 03:13:56.115283  8554 solver.cpp:209] Iteration 3, loss = 0.745306
I1118 03:13:56.115309  8554 solver.cpp:464] Iteration 3, lr = 1e-05
I1118 03:13:58.180665  8554 solver.cpp:209] Iteration 4, loss = 0.99869
I1118 03:13:58.180693  8554 solver.cpp:464] Iteration 4, lr = 1e-05
I1118 03:14:00.246012  8554 solver.cpp:209] Iteration 5, loss = 0.894645
I1118 03:14:00.246052  8554 solver.cpp:464] Iteration 5, lr = 1e-05
I1118 03:14:02.313925  8554 solver.cpp:209] Iteration 6, loss = 0.919775
I1118 03:14:02.313966  8554 solver.cpp:464] Iteration 6, lr = 1e-05
I1118 03:14:04.389895  8554 solver.cpp:209] Iteration 7, loss = 0.822229
I1118 03:14:04.389924  8554 solver.cpp:464] Iteration 7, lr = 1e-05
I1118 03:14:06.463841  8554 solver.cpp:209] Iteration 8, loss = 1.01485
I1118 03:14:06.463865  8554 solver.cpp:464] Iteration 8, lr = 1e-05
I1118 03:14:08.540153  8554 solver.cpp:209] Iteration 9, loss = 1.04588
I1118 03:14:08.540191  8554 solver.cpp:464] Iteration 9, lr = 1e-05
I1118 03:14:10.614322  8554 solver.cpp:209] Iteration 10, loss = 0.814088
I1118 03:14:10.614366  8554 solver.cpp:464] Iteration 10, lr = 1e-05
I1118 03:14:12.688254  8554 solver.cpp:209] Iteration 11, loss = 0.801238
I1118 03:14:12.688283  8554 solver.cpp:464] Iteration 11, lr = 1e-05
I1118 03:14:14.767432  8554 solver.cpp:209] Iteration 12, loss = 0.790502
I1118 03:14:14.767484  8554 solver.cpp:464] Iteration 12, lr = 1e-05
I1118 03:14:16.845361  8554 solver.cpp:209] Iteration 13, loss = 0.962212
I1118 03:14:16.845391  8554 solver.cpp:464] Iteration 13, lr = 1e-05
I1118 03:14:18.918675  8554 solver.cpp:209] Iteration 14, loss = 0.875686
I1118 03:14:18.918704  8554 solver.cpp:464] Iteration 14, lr = 1e-05
I1118 03:14:20.982264  8554 solver.cpp:209] Iteration 15, loss = 0.901001
I1118 03:14:20.982343  8554 solver.cpp:464] Iteration 15, lr = 1e-05
I1118 03:14:23.061578  8554 solver.cpp:209] Iteration 16, loss = 0.699954
I1118 03:14:23.061605  8554 solver.cpp:464] Iteration 16, lr = 1e-05
I1118 03:14:25.143237  8554 solver.cpp:209] Iteration 17, loss = 0.904482
I1118 03:14:25.143278  8554 solver.cpp:464] Iteration 17, lr = 1e-05
I1118 03:14:27.221647  8554 solver.cpp:209] Iteration 18, loss = 0.954409
I1118 03:14:27.221688  8554 solver.cpp:464] Iteration 18, lr = 1e-05
I1118 03:14:29.295120  8554 solver.cpp:209] Iteration 19, loss = 0.792855
I1118 03:14:29.295159  8554 solver.cpp:464] Iteration 19, lr = 1e-05
I1118 03:14:31.375740  8554 solver.cpp:209] Iteration 20, loss = 1.09459
I1118 03:14:31.375768  8554 solver.cpp:464] Iteration 20, lr = 1e-05
I1118 03:14:33.461112  8554 solver.cpp:209] Iteration 21, loss = 1.04321
I1118 03:14:33.461139  8554 solver.cpp:464] Iteration 21, lr = 1e-05
I1118 03:14:35.543267  8554 solver.cpp:209] Iteration 22, loss = 0.939652
I1118 03:14:35.543308  8554 solver.cpp:464] Iteration 22, lr = 1e-05
I1118 03:14:37.617512  8554 solver.cpp:209] Iteration 23, loss = 1.18722
I1118 03:14:37.617540  8554 solver.cpp:464] Iteration 23, lr = 1e-05
I1118 03:14:39.692872  8554 solver.cpp:209] Iteration 24, loss = 0.630164
I1118 03:14:39.692899  8554 solver.cpp:464] Iteration 24, lr = 1e-05
I1118 03:14:41.767917  8554 solver.cpp:209] Iteration 25, loss = 0.872757
I1118 03:14:41.767946  8554 solver.cpp:464] Iteration 25, lr = 1e-05
I1118 03:14:43.848342  8554 solver.cpp:209] Iteration 26, loss = 0.79071
I1118 03:14:43.848371  8554 solver.cpp:464] Iteration 26, lr = 1e-05
I1118 03:14:45.930902  8554 solver.cpp:209] Iteration 27, loss = 0.701634
I1118 03:14:45.930943  8554 solver.cpp:464] Iteration 27, lr = 1e-05
I1118 03:14:48.017375  8554 solver.cpp:209] Iteration 28, loss = 0.975593
I1118 03:14:48.017403  8554 solver.cpp:464] Iteration 28, lr = 1e-05
I1118 03:14:50.093736  8554 solver.cpp:209] Iteration 29, loss = 0.754587
I1118 03:14:50.093765  8554 solver.cpp:464] Iteration 29, lr = 1e-05
I1118 03:14:52.170064  8554 solver.cpp:209] Iteration 30, loss = 1.09163
I1118 03:14:52.170147  8554 solver.cpp:464] Iteration 30, lr = 1e-05
I1118 03:14:54.255978  8554 solver.cpp:209] Iteration 31, loss = 1.15168
I1118 03:14:54.256009  8554 solver.cpp:464] Iteration 31, lr = 1e-05
I1118 03:14:56.346546  8554 solver.cpp:209] Iteration 32, loss = 1.0741
I1118 03:14:56.346573  8554 solver.cpp:464] Iteration 32, lr = 1e-05
I1118 03:14:58.426170  8554 solver.cpp:209] Iteration 33, loss = 0.971495
I1118 03:14:58.426213  8554 solver.cpp:464] Iteration 33, lr = 1e-05
I1118 03:15:00.501250  8554 solver.cpp:209] Iteration 34, loss = 0.850366
I1118 03:15:00.501291  8554 solver.cpp:464] Iteration 34, lr = 1e-05
I1118 03:15:02.584801  8554 solver.cpp:209] Iteration 35, loss = 0.733172
I1118 03:15:02.584843  8554 solver.cpp:464] Iteration 35, lr = 1e-05
I1118 03:15:04.677001  8554 solver.cpp:209] Iteration 36, loss = 0.931921
I1118 03:15:04.677028  8554 solver.cpp:464] Iteration 36, lr = 1e-05
I1118 03:15:06.766048  8554 solver.cpp:209] Iteration 37, loss = 0.820835
I1118 03:15:06.766077  8554 solver.cpp:464] Iteration 37, lr = 1e-05
I1118 03:15:08.848285  8554 solver.cpp:209] Iteration 38, loss = 0.670756
I1118 03:15:08.848326  8554 solver.cpp:464] Iteration 38, lr = 1e-05
I1118 03:15:10.930688  8554 solver.cpp:209] Iteration 39, loss = 0.900575
I1118 03:15:10.930716  8554 solver.cpp:464] Iteration 39, lr = 1e-05
I1118 03:15:13.014264  8554 solver.cpp:209] Iteration 40, loss = 0.771692
I1118 03:15:13.014307  8554 solver.cpp:464] Iteration 40, lr = 1e-05
I1118 03:15:15.143275  8554 solver.cpp:209] Iteration 41, loss = 0.792243
I1118 03:15:15.143302  8554 solver.cpp:464] Iteration 41, lr = 1e-05
I1118 03:15:17.356405  8554 solver.cpp:209] Iteration 42, loss = 0.776203
I1118 03:15:17.356434  8554 solver.cpp:464] Iteration 42, lr = 1e-05
I1118 03:15:19.566253  8554 solver.cpp:209] Iteration 43, loss = 0.742587
I1118 03:15:19.566279  8554 solver.cpp:464] Iteration 43, lr = 1e-05
I1118 03:15:21.776996  8554 solver.cpp:209] Iteration 44, loss = 0.908322
I1118 03:15:21.777022  8554 solver.cpp:464] Iteration 44, lr = 1e-05
I1118 03:15:23.987263  8554 solver.cpp:209] Iteration 45, loss = 0.695307
I1118 03:15:23.987344  8554 solver.cpp:464] Iteration 45, lr = 1e-05
I1118 03:15:26.188254  8554 solver.cpp:209] Iteration 46, loss = 0.801413
I1118 03:15:26.188289  8554 solver.cpp:464] Iteration 46, lr = 1e-05
I1118 03:15:28.398833  8554 solver.cpp:209] Iteration 47, loss = 0.694081
I1118 03:15:28.398867  8554 solver.cpp:464] Iteration 47, lr = 1e-05
I1118 03:15:30.604517  8554 solver.cpp:209] Iteration 48, loss = 0.699291
I1118 03:15:30.604548  8554 solver.cpp:464] Iteration 48, lr = 1e-05
I1118 03:15:32.821558  8554 solver.cpp:209] Iteration 49, loss = 0.634379
I1118 03:15:32.821593  8554 solver.cpp:464] Iteration 49, lr = 1e-05
I1118 03:15:32.822165  8554 solver.cpp:264] Iteration 50, Testing net (#0)
I1118 03:15:54.881671  8554 solver.cpp:305] Test loss: 0.569014
I1118 03:15:54.881747  8554 solver.cpp:318] mean_score = test_score[0] { = 670} / test_score[1] { = 868 }
I1118 03:15:54.881755  8554 solver.cpp:319]            = 0.771889
I1118 03:15:54.881759  8554 solver.cpp:328]     Test net output #0: accuracy = 0.771889
I1118 03:15:54.881764  8554 solver.cpp:318] mean_score = test_score[2] { = 33} / test_score[3] { = 100 }
I1118 03:15:54.881768  8554 solver.cpp:319]            = 0.33
I1118 03:15:54.881772  8554 solver.cpp:328]     Test net output #1: accuracy = 0.33
I1118 03:15:54.881778  8554 solver.cpp:332]     Test net output #2: accuracy = 0.72624
I1118 03:15:54.881782  8554 solver.cpp:334]     Test net output #3: accuracy = 0.550945
I1118 03:15:55.524196  8554 solver.cpp:209] Iteration 50, loss = 0.911609
I1118 03:15:55.524236  8554 solver.cpp:464] Iteration 50, lr = 1e-05
I1118 03:15:57.748291  8554 solver.cpp:209] Iteration 51, loss = 0.722145
I1118 03:15:57.748318  8554 solver.cpp:464] Iteration 51, lr = 1e-05
I1118 03:15:59.986093  8554 solver.cpp:209] Iteration 52, loss = 0.675159
I1118 03:15:59.986119  8554 solver.cpp:464] Iteration 52, lr = 1e-05
I1118 03:16:02.214058  8554 solver.cpp:209] Iteration 53, loss = 0.70489
I1118 03:16:02.214085  8554 solver.cpp:464] Iteration 53, lr = 1e-05
I1118 03:16:04.439173  8554 solver.cpp:209] Iteration 54, loss = 0.575406
I1118 03:16:04.439200  8554 solver.cpp:464] Iteration 54, lr = 1e-05
I1118 03:16:06.668948  8554 solver.cpp:209] Iteration 55, loss = 0.911264
I1118 03:16:06.668977  8554 solver.cpp:464] Iteration 55, lr = 1e-05
I1118 03:16:08.901334  8554 solver.cpp:209] Iteration 56, loss = 0.866414
I1118 03:16:08.901374  8554 solver.cpp:464] Iteration 56, lr = 1e-05
I1118 03:16:11.142050  8554 solver.cpp:209] Iteration 57, loss = 0.702394
I1118 03:16:11.142079  8554 solver.cpp:464] Iteration 57, lr = 1e-05
I1118 03:16:13.372344  8554 solver.cpp:209] Iteration 58, loss = 0.71829
I1118 03:16:13.372371  8554 solver.cpp:464] Iteration 58, lr = 1e-05
I1118 03:16:15.600462  8554 solver.cpp:209] Iteration 59, loss = 0.895885
I1118 03:16:15.600503  8554 solver.cpp:464] Iteration 59, lr = 1e-05
I1118 03:16:17.830202  8554 solver.cpp:209] Iteration 60, loss = 0.639043
I1118 03:16:17.830230  8554 solver.cpp:464] Iteration 60, lr = 1e-05
I1118 03:16:20.058514  8554 solver.cpp:209] Iteration 61, loss = 0.704349
I1118 03:16:20.058542  8554 solver.cpp:464] Iteration 61, lr = 1e-05
I1118 03:16:22.291829  8554 solver.cpp:209] Iteration 62, loss = 0.528218
I1118 03:16:22.291867  8554 solver.cpp:464] Iteration 62, lr = 1e-05
I1118 03:16:24.522680  8554 solver.cpp:209] Iteration 63, loss = 0.961201
I1118 03:16:24.522708  8554 solver.cpp:464] Iteration 63, lr = 1e-05
I1118 03:16:26.754559  8554 solver.cpp:209] Iteration 64, loss = 0.70448
I1118 03:16:26.754634  8554 solver.cpp:464] Iteration 64, lr = 1e-05
I1118 03:16:28.991526  8554 solver.cpp:209] Iteration 65, loss = 0.597672
I1118 03:16:28.991554  8554 solver.cpp:464] Iteration 65, lr = 1e-05
I1118 03:16:31.225110  8554 solver.cpp:209] Iteration 66, loss = 0.76222
I1118 03:16:31.225137  8554 solver.cpp:464] Iteration 66, lr = 1e-05
I1118 03:16:33.457635  8554 solver.cpp:209] Iteration 67, loss = 0.844343
I1118 03:16:33.457676  8554 solver.cpp:464] Iteration 67, lr = 1e-05
I1118 03:16:35.686091  8554 solver.cpp:209] Iteration 68, loss = 0.70109
I1118 03:16:35.686132  8554 solver.cpp:464] Iteration 68, lr = 1e-05
I1118 03:16:37.912035  8554 solver.cpp:209] Iteration 69, loss = 0.890849
I1118 03:16:37.912063  8554 solver.cpp:464] Iteration 69, lr = 1e-05
I1118 03:16:40.147359  8554 solver.cpp:209] Iteration 70, loss = 0.947484
I1118 03:16:40.147389  8554 solver.cpp:464] Iteration 70, lr = 1e-05
I1118 03:16:42.378924  8554 solver.cpp:209] Iteration 71, loss = 0.996773
I1118 03:16:42.378953  8554 solver.cpp:464] Iteration 71, lr = 1e-05
I1118 03:16:44.617203  8554 solver.cpp:209] Iteration 72, loss = 0.721476
I1118 03:16:44.617230  8554 solver.cpp:464] Iteration 72, lr = 1e-05
I1118 03:16:46.852416  8554 solver.cpp:209] Iteration 73, loss = 0.768763
I1118 03:16:46.852443  8554 solver.cpp:464] Iteration 73, lr = 1e-05
I1118 03:16:49.081576  8554 solver.cpp:209] Iteration 74, loss = 0.764976
I1118 03:16:49.081606  8554 solver.cpp:464] Iteration 74, lr = 1e-05
I1118 03:16:51.310258  8554 solver.cpp:209] Iteration 75, loss = 1.02997
I1118 03:16:51.310286  8554 solver.cpp:464] Iteration 75, lr = 1e-05
I1118 03:16:53.531389  8554 solver.cpp:209] Iteration 76, loss = 0.665897
I1118 03:16:53.531419  8554 solver.cpp:464] Iteration 76, lr = 1e-05
I1118 03:16:55.764721  8554 solver.cpp:209] Iteration 77, loss = 0.924493
I1118 03:16:55.764749  8554 solver.cpp:464] Iteration 77, lr = 1e-05
I1118 03:16:58.001361  8554 solver.cpp:209] Iteration 78, loss = 0.620058
I1118 03:16:58.001441  8554 solver.cpp:464] Iteration 78, lr = 1e-05
I1118 03:17:00.240350  8554 solver.cpp:209] Iteration 79, loss = 0.877821
I1118 03:17:00.240378  8554 solver.cpp:464] Iteration 79, lr = 1e-05
I1118 03:17:02.475832  8554 solver.cpp:209] Iteration 80, loss = 0.797835
I1118 03:17:02.475858  8554 solver.cpp:464] Iteration 80, lr = 1e-05
I1118 03:17:04.699569  8554 solver.cpp:209] Iteration 81, loss = 0.795522
I1118 03:17:04.699597  8554 solver.cpp:464] Iteration 81, lr = 1e-05
I1118 03:17:06.932651  8554 solver.cpp:209] Iteration 82, loss = 0.660996
I1118 03:17:06.932679  8554 solver.cpp:464] Iteration 82, lr = 1e-05
I1118 03:17:09.156482  8554 solver.cpp:209] Iteration 83, loss = 0.763736
I1118 03:17:09.156522  8554 solver.cpp:464] Iteration 83, lr = 1e-05
I1118 03:17:11.396306  8554 solver.cpp:209] Iteration 84, loss = 0.719734
I1118 03:17:11.396334  8554 solver.cpp:464] Iteration 84, lr = 1e-05
I1118 03:17:13.661880  8554 solver.cpp:209] Iteration 85, loss = 0.654278
I1118 03:17:13.661907  8554 solver.cpp:464] Iteration 85, lr = 1e-05
I1118 03:17:15.916601  8554 solver.cpp:209] Iteration 86, loss = 0.797786
I1118 03:17:15.916630  8554 solver.cpp:464] Iteration 86, lr = 1e-05
I1118 03:17:18.180308  8554 solver.cpp:209] Iteration 87, loss = 0.86975
I1118 03:17:18.180349  8554 solver.cpp:464] Iteration 87, lr = 1e-05
I1118 03:17:20.436015  8554 solver.cpp:209] Iteration 88, loss = 0.718416
I1118 03:17:20.436043  8554 solver.cpp:464] Iteration 88, lr = 1e-05
I1118 03:17:22.688091  8554 solver.cpp:209] Iteration 89, loss = 0.739079
I1118 03:17:22.688118  8554 solver.cpp:464] Iteration 89, lr = 1e-05
I1118 03:17:24.941501  8554 solver.cpp:209] Iteration 90, loss = 0.856285
I1118 03:17:24.941541  8554 solver.cpp:464] Iteration 90, lr = 1e-05
I1118 03:17:27.194025  8554 solver.cpp:209] Iteration 91, loss = 0.767059
I1118 03:17:27.194053  8554 solver.cpp:464] Iteration 91, lr = 1e-05
I1118 03:17:29.455869  8554 solver.cpp:209] Iteration 92, loss = 0.716521
I1118 03:17:29.455952  8554 solver.cpp:464] Iteration 92, lr = 1e-05
I1118 03:17:31.725136  8554 solver.cpp:209] Iteration 93, loss = 0.669104
I1118 03:17:31.725177  8554 solver.cpp:464] Iteration 93, lr = 1e-05
I1118 03:17:33.980304  8554 solver.cpp:209] Iteration 94, loss = 0.770597
I1118 03:17:33.980332  8554 solver.cpp:464] Iteration 94, lr = 1e-05
I1118 03:17:36.243556  8554 solver.cpp:209] Iteration 95, loss = 0.53257
I1118 03:17:36.243582  8554 solver.cpp:464] Iteration 95, lr = 1e-05
I1118 03:17:38.497334  8554 solver.cpp:209] Iteration 96, loss = 0.625805
I1118 03:17:38.497364  8554 solver.cpp:464] Iteration 96, lr = 1e-05
I1118 03:17:40.754215  8554 solver.cpp:209] Iteration 97, loss = 0.579746
I1118 03:17:40.754243  8554 solver.cpp:464] Iteration 97, lr = 1e-05
I1118 03:17:43.009152  8554 solver.cpp:209] Iteration 98, loss = 0.836553
I1118 03:17:43.009193  8554 solver.cpp:464] Iteration 98, lr = 1e-05
I1118 03:17:45.260555  8554 solver.cpp:209] Iteration 99, loss = 0.656699
I1118 03:17:45.260596  8554 solver.cpp:464] Iteration 99, lr = 1e-05
I1118 03:17:45.261169  8554 solver.cpp:264] Iteration 100, Testing net (#0)
I1118 03:18:07.844602  8554 solver.cpp:305] Test loss: 0.642622
I1118 03:18:07.844674  8554 solver.cpp:318] mean_score = test_score[0] { = 547} / test_score[1] { = 868 }
I1118 03:18:07.844683  8554 solver.cpp:319]            = 0.630184
I1118 03:18:07.844687  8554 solver.cpp:328]     Test net output #0: accuracy = 0.630184
I1118 03:18:07.844692  8554 solver.cpp:318] mean_score = test_score[2] { = 55} / test_score[3] { = 100 }
I1118 03:18:07.844696  8554 solver.cpp:319]            = 0.55
I1118 03:18:07.844701  8554 solver.cpp:328]     Test net output #1: accuracy = 0.55
I1118 03:18:07.844704  8554 solver.cpp:332]     Test net output #2: accuracy = 0.621901
I1118 03:18:07.844708  8554 solver.cpp:334]     Test net output #3: accuracy = 0.590092
I1118 03:18:08.508550  8554 solver.cpp:209] Iteration 100, loss = 0.654138
I1118 03:18:08.508590  8554 solver.cpp:464] Iteration 100, lr = 1e-05
I1118 03:18:10.806257  8554 solver.cpp:209] Iteration 101, loss = 0.655966
I1118 03:18:10.806285  8554 solver.cpp:464] Iteration 101, lr = 1e-05
I1118 03:18:14.130631  8554 solver.cpp:209] Iteration 102, loss = 0.756436
I1118 03:18:14.130674  8554 solver.cpp:464] Iteration 102, lr = 1e-05
I1118 03:18:16.219349  8554 solver.cpp:209] Iteration 103, loss = 0.804817
I1118 03:18:16.219377  8554 solver.cpp:464] Iteration 103, lr = 1e-05
I1118 03:18:18.470824  8554 solver.cpp:209] Iteration 104, loss = 0.649166
I1118 03:18:18.470865  8554 solver.cpp:464] Iteration 104, lr = 1e-05
I1118 03:18:20.733196  8554 solver.cpp:209] Iteration 105, loss = 0.867795
I1118 03:18:20.733237  8554 solver.cpp:464] Iteration 105, lr = 1e-05
I1118 03:18:22.998656  8554 solver.cpp:209] Iteration 106, loss = 0.766875
I1118 03:18:22.998697  8554 solver.cpp:464] Iteration 106, lr = 1e-05
I1118 03:18:25.253669  8554 solver.cpp:209] Iteration 107, loss = 0.652839
I1118 03:18:25.253707  8554 solver.cpp:464] Iteration 107, lr = 1e-05
I1118 03:18:27.502718  8554 solver.cpp:209] Iteration 108, loss = 0.530789
I1118 03:18:27.502748  8554 solver.cpp:464] Iteration 108, lr = 1e-05
I1118 03:18:29.757138  8554 solver.cpp:209] Iteration 109, loss = 0.835878
I1118 03:18:29.757166  8554 solver.cpp:464] Iteration 109, lr = 1e-05
I1118 03:18:32.026741  8554 solver.cpp:209] Iteration 110, loss = 0.738566
I1118 03:18:32.026768  8554 solver.cpp:464] Iteration 110, lr = 1e-05
I1118 03:18:34.288635  8554 solver.cpp:209] Iteration 111, loss = 0.778385
I1118 03:18:34.288663  8554 solver.cpp:464] Iteration 111, lr = 1e-05
I1118 03:18:36.547308  8554 solver.cpp:209] Iteration 112, loss = 0.667055
I1118 03:18:36.547336  8554 solver.cpp:464] Iteration 112, lr = 1e-05
I1118 03:18:38.804860  8554 solver.cpp:209] Iteration 113, loss = 0.747582
I1118 03:18:38.804946  8554 solver.cpp:464] Iteration 113, lr = 1e-05
I1118 03:18:41.061166  8554 solver.cpp:209] Iteration 114, loss = 0.628964
I1118 03:18:41.061193  8554 solver.cpp:464] Iteration 114, lr = 1e-05
I1118 03:18:43.317472  8554 solver.cpp:209] Iteration 115, loss = 0.671595
I1118 03:18:43.317500  8554 solver.cpp:464] Iteration 115, lr = 1e-05
I1118 03:18:45.571740  8554 solver.cpp:209] Iteration 116, loss = 0.503787
I1118 03:18:45.571768  8554 solver.cpp:464] Iteration 116, lr = 1e-05
I1118 03:18:47.830519  8554 solver.cpp:209] Iteration 117, loss = 0.816632
I1118 03:18:47.830559  8554 solver.cpp:464] Iteration 117, lr = 1e-05
I1118 03:18:50.086272  8554 solver.cpp:209] Iteration 118, loss = 0.742056
I1118 03:18:50.086300  8554 solver.cpp:464] Iteration 118, lr = 1e-05
I1118 03:18:52.343703  8554 solver.cpp:209] Iteration 119, loss = 0.793184
I1118 03:18:52.343730  8554 solver.cpp:464] Iteration 119, lr = 1e-05
I1118 03:18:54.610556  8554 solver.cpp:209] Iteration 120, loss = 0.867761
I1118 03:18:54.610597  8554 solver.cpp:464] Iteration 120, lr = 1e-05
I1118 03:18:56.873965  8554 solver.cpp:209] Iteration 121, loss = 0.893273
I1118 03:18:56.873993  8554 solver.cpp:464] Iteration 121, lr = 1e-05
I1118 03:18:59.136096  8554 solver.cpp:209] Iteration 122, loss = 0.610882
I1118 03:18:59.136124  8554 solver.cpp:464] Iteration 122, lr = 1e-05
I1118 03:19:01.412298  8554 solver.cpp:209] Iteration 123, loss = 0.635149
I1118 03:19:01.412325  8554 solver.cpp:464] Iteration 123, lr = 1e-05
I1118 03:19:03.711174  8554 solver.cpp:209] Iteration 124, loss = 0.756771
I1118 03:19:03.711215  8554 solver.cpp:464] Iteration 124, lr = 1e-05
I1118 03:19:06.033298  8554 solver.cpp:209] Iteration 125, loss = 0.895214
I1118 03:19:06.033325  8554 solver.cpp:464] Iteration 125, lr = 1e-05
I1118 03:19:08.351429  8554 solver.cpp:209] Iteration 126, loss = 0.7077
I1118 03:19:08.351457  8554 solver.cpp:464] Iteration 126, lr = 1e-05
I1118 03:19:10.669080  8554 solver.cpp:209] Iteration 127, loss = 0.702503
I1118 03:19:10.669162  8554 solver.cpp:464] Iteration 127, lr = 1e-05
I1118 03:19:12.981462  8554 solver.cpp:209] Iteration 128, loss = 0.664262
I1118 03:19:12.981492  8554 solver.cpp:464] Iteration 128, lr = 1e-05
I1118 03:19:15.294697  8554 solver.cpp:209] Iteration 129, loss = 0.756935
I1118 03:19:15.294738  8554 solver.cpp:464] Iteration 129, lr = 1e-05
I1118 03:19:17.618132  8554 solver.cpp:209] Iteration 130, loss = 0.806154
I1118 03:19:17.618160  8554 solver.cpp:464] Iteration 130, lr = 1e-05
I1118 03:19:19.951264  8554 solver.cpp:209] Iteration 131, loss = 0.729658
I1118 03:19:19.951292  8554 solver.cpp:464] Iteration 131, lr = 1e-05
I1118 03:19:22.273377  8554 solver.cpp:209] Iteration 132, loss = 0.58594
I1118 03:19:22.273404  8554 solver.cpp:464] Iteration 132, lr = 1e-05
I1118 03:19:24.561971  8554 solver.cpp:209] Iteration 133, loss = 0.618749
I1118 03:19:24.562000  8554 solver.cpp:464] Iteration 133, lr = 1e-05
I1118 03:19:26.878569  8554 solver.cpp:209] Iteration 134, loss = 0.736374
I1118 03:19:26.878598  8554 solver.cpp:464] Iteration 134, lr = 1e-05
I1118 03:19:29.173123  8554 solver.cpp:209] Iteration 135, loss = 0.751614
I1118 03:19:29.173163  8554 solver.cpp:464] Iteration 135, lr = 1e-05
I1118 03:19:31.457021  8554 solver.cpp:209] Iteration 136, loss = 0.785264
I1118 03:19:31.457051  8554 solver.cpp:464] Iteration 136, lr = 1e-05
I1118 03:19:33.742477  8554 solver.cpp:209] Iteration 137, loss = 0.812577
I1118 03:19:33.742506  8554 solver.cpp:464] Iteration 137, lr = 1e-05
I1118 03:19:36.022807  8554 solver.cpp:209] Iteration 138, loss = 0.52137
I1118 03:19:36.022846  8554 solver.cpp:464] Iteration 138, lr = 1e-05
I1118 03:19:38.309465  8554 solver.cpp:209] Iteration 139, loss = 0.617668
I1118 03:19:38.309494  8554 solver.cpp:464] Iteration 139, lr = 1e-05
I1118 03:19:40.592339  8554 solver.cpp:209] Iteration 140, loss = 0.920835
I1118 03:19:40.592367  8554 solver.cpp:464] Iteration 140, lr = 1e-05
I1118 03:19:42.873558  8554 solver.cpp:209] Iteration 141, loss = 0.830436
I1118 03:19:42.873641  8554 solver.cpp:464] Iteration 141, lr = 1e-05
I1118 03:19:45.173010  8554 solver.cpp:209] Iteration 142, loss = 0.734629
I1118 03:19:45.173050  8554 solver.cpp:464] Iteration 142, lr = 1e-05
I1118 03:19:47.462939  8554 solver.cpp:209] Iteration 143, loss = 0.620999
I1118 03:19:47.462978  8554 solver.cpp:464] Iteration 143, lr = 1e-05
I1118 03:19:49.753435  8554 solver.cpp:209] Iteration 144, loss = 0.483088
I1118 03:19:49.753475  8554 solver.cpp:464] Iteration 144, lr = 1e-05
I1118 03:19:52.070940  8554 solver.cpp:209] Iteration 145, loss = 0.811223
I1118 03:19:52.070977  8554 solver.cpp:464] Iteration 145, lr = 1e-05
I1118 03:19:54.357403  8554 solver.cpp:209] Iteration 146, loss = 0.529903
I1118 03:19:54.357431  8554 solver.cpp:464] Iteration 146, lr = 1e-05
I1118 03:19:56.638911  8554 solver.cpp:209] Iteration 147, loss = 0.476889
I1118 03:19:56.638941  8554 solver.cpp:464] Iteration 147, lr = 1e-05
I1118 03:19:58.924437  8554 solver.cpp:209] Iteration 148, loss = 0.762821
I1118 03:19:58.924464  8554 solver.cpp:464] Iteration 148, lr = 1e-05
I1118 03:20:01.211900  8554 solver.cpp:209] Iteration 149, loss = 0.561881
I1118 03:20:01.211927  8554 solver.cpp:464] Iteration 149, lr = 1e-05
I1118 03:20:01.212519  8554 solver.cpp:264] Iteration 150, Testing net (#0)
I1118 03:20:24.057602  8554 solver.cpp:305] Test loss: 0.461746
I1118 03:20:24.057646  8554 solver.cpp:318] mean_score = test_score[0] { = 798} / test_score[1] { = 868 }
I1118 03:20:24.057665  8554 solver.cpp:319]            = 0.919355
I1118 03:20:24.057670  8554 solver.cpp:328]     Test net output #0: accuracy = 0.919355
I1118 03:20:24.057674  8554 solver.cpp:318] mean_score = test_score[2] { = 21} / test_score[3] { = 100 }
I1118 03:20:24.057679  8554 solver.cpp:319]            = 0.21
I1118 03:20:24.057683  8554 solver.cpp:328]     Test net output #1: accuracy = 0.21
I1118 03:20:24.057687  8554 solver.cpp:332]     Test net output #2: accuracy = 0.846074
I1118 03:20:24.057692  8554 solver.cpp:334]     Test net output #3: accuracy = 0.564677
I1118 03:20:24.733868  8554 solver.cpp:209] Iteration 150, loss = 0.591334
I1118 03:20:24.733909  8554 solver.cpp:464] Iteration 150, lr = 1e-05
I1118 03:20:27.057245  8554 solver.cpp:209] Iteration 151, loss = 0.588606
I1118 03:20:27.057274  8554 solver.cpp:464] Iteration 151, lr = 1e-05
I1118 03:20:29.380673  8554 solver.cpp:209] Iteration 152, loss = 0.725153
I1118 03:20:29.380714  8554 solver.cpp:464] Iteration 152, lr = 1e-05
I1118 03:20:31.712200  8554 solver.cpp:209] Iteration 153, loss = 0.65471
I1118 03:20:31.712229  8554 solver.cpp:464] Iteration 153, lr = 1e-05
I1118 03:20:34.030731  8554 solver.cpp:209] Iteration 154, loss = 0.534371
I1118 03:20:34.030772  8554 solver.cpp:464] Iteration 154, lr = 1e-05
I1118 03:20:36.349465  8554 solver.cpp:209] Iteration 155, loss = 0.614301
I1118 03:20:36.349493  8554 solver.cpp:464] Iteration 155, lr = 1e-05
I1118 03:20:38.666374  8554 solver.cpp:209] Iteration 156, loss = 0.948205
I1118 03:20:38.666414  8554 solver.cpp:464] Iteration 156, lr = 1e-05
I1118 03:20:40.982141  8554 solver.cpp:209] Iteration 157, loss = 0.65042
I1118 03:20:40.982169  8554 solver.cpp:464] Iteration 157, lr = 1e-05
I1118 03:20:43.296720  8554 solver.cpp:209] Iteration 158, loss = 1.0956
I1118 03:20:43.296761  8554 solver.cpp:464] Iteration 158, lr = 1e-05
I1118 03:20:45.607637  8554 solver.cpp:209] Iteration 159, loss = 0.80547
I1118 03:20:45.607666  8554 solver.cpp:464] Iteration 159, lr = 1e-05
I1118 03:20:47.926909  8554 solver.cpp:209] Iteration 160, loss = 0.609171
I1118 03:20:47.926939  8554 solver.cpp:464] Iteration 160, lr = 1e-05
I1118 03:20:50.222971  8554 solver.cpp:209] Iteration 161, loss = 0.648169
I1118 03:20:50.223002  8554 solver.cpp:464] Iteration 161, lr = 1e-05
I1118 03:20:52.488224  8554 solver.cpp:209] Iteration 162, loss = 0.710511
I1118 03:20:52.488255  8554 solver.cpp:464] Iteration 162, lr = 1e-05
I1118 03:20:54.749212  8554 solver.cpp:209] Iteration 163, loss = 0.455228
I1118 03:20:54.749300  8554 solver.cpp:464] Iteration 163, lr = 1e-05
I1118 03:20:57.001191  8554 solver.cpp:209] Iteration 164, loss = 0.853469
I1118 03:20:57.001222  8554 solver.cpp:464] Iteration 164, lr = 1e-05
I1118 03:20:59.272384  8554 solver.cpp:209] Iteration 165, loss = 0.735801
I1118 03:20:59.272426  8554 solver.cpp:464] Iteration 165, lr = 1e-05
I1118 03:21:01.554983  8554 solver.cpp:209] Iteration 166, loss = 0.888681
I1118 03:21:01.555013  8554 solver.cpp:464] Iteration 166, lr = 1e-05
I1118 03:21:03.807235  8554 solver.cpp:209] Iteration 167, loss = 0.741706
I1118 03:21:03.807265  8554 solver.cpp:464] Iteration 167, lr = 1e-05
I1118 03:21:06.061702  8554 solver.cpp:209] Iteration 168, loss = 0.667486
I1118 03:21:06.061729  8554 solver.cpp:464] Iteration 168, lr = 1e-05
I1118 03:21:08.313557  8554 solver.cpp:209] Iteration 169, loss = 0.660575
I1118 03:21:08.313599  8554 solver.cpp:464] Iteration 169, lr = 1e-05
I1118 03:21:10.570863  8554 solver.cpp:209] Iteration 170, loss = 0.779877
I1118 03:21:10.570893  8554 solver.cpp:464] Iteration 170, lr = 1e-05
I1118 03:21:12.833150  8554 solver.cpp:209] Iteration 171, loss = 0.570266
I1118 03:21:12.833179  8554 solver.cpp:464] Iteration 171, lr = 1e-05
I1118 03:21:15.093281  8554 solver.cpp:209] Iteration 172, loss = 0.618721
I1118 03:21:15.093322  8554 solver.cpp:464] Iteration 172, lr = 1e-05
I1118 03:21:17.353135  8554 solver.cpp:209] Iteration 173, loss = 0.705211
I1118 03:21:17.353165  8554 solver.cpp:464] Iteration 173, lr = 1e-05
I1118 03:21:19.606170  8554 solver.cpp:209] Iteration 174, loss = 0.507996
I1118 03:21:19.606199  8554 solver.cpp:464] Iteration 174, lr = 1e-05
I1118 03:21:21.856633  8554 solver.cpp:209] Iteration 175, loss = 0.700372
I1118 03:21:21.856662  8554 solver.cpp:464] Iteration 175, lr = 1e-05
I1118 03:21:24.108729  8554 solver.cpp:209] Iteration 176, loss = 0.723614
I1118 03:21:24.108772  8554 solver.cpp:464] Iteration 176, lr = 1e-05
I1118 03:21:26.361826  8554 solver.cpp:209] Iteration 177, loss = 0.704573
I1118 03:21:26.361899  8554 solver.cpp:464] Iteration 177, lr = 1e-05
I1118 03:21:28.618295  8554 solver.cpp:209] Iteration 178, loss = 0.738586
I1118 03:21:28.618325  8554 solver.cpp:464] Iteration 178, lr = 1e-05
I1118 03:21:30.879942  8554 solver.cpp:209] Iteration 179, loss = 0.728249
I1118 03:21:30.879984  8554 solver.cpp:464] Iteration 179, lr = 1e-05
I1118 03:21:33.147917  8554 solver.cpp:209] Iteration 180, loss = 0.701338
I1118 03:21:33.147946  8554 solver.cpp:464] Iteration 180, lr = 1e-05
I1118 03:21:35.409749  8554 solver.cpp:209] Iteration 181, loss = 0.878683
I1118 03:21:35.409788  8554 solver.cpp:464] Iteration 181, lr = 1e-05
I1118 03:21:37.668438  8554 solver.cpp:209] Iteration 182, loss = 0.60968
I1118 03:21:37.668468  8554 solver.cpp:464] Iteration 182, lr = 1e-05
I1118 03:21:39.925408  8554 solver.cpp:209] Iteration 183, loss = 0.551092
I1118 03:21:39.925437  8554 solver.cpp:464] Iteration 183, lr = 1e-05
I1118 03:21:42.177279  8554 solver.cpp:209] Iteration 184, loss = 0.937912
I1118 03:21:42.177306  8554 solver.cpp:464] Iteration 184, lr = 1e-05
I1118 03:21:44.431025  8554 solver.cpp:209] Iteration 185, loss = 0.623768
I1118 03:21:44.431053  8554 solver.cpp:464] Iteration 185, lr = 1e-05
I1118 03:21:46.690773  8554 solver.cpp:209] Iteration 186, loss = 0.652738
I1118 03:21:46.690801  8554 solver.cpp:464] Iteration 186, lr = 1e-05
I1118 03:21:48.957815  8554 solver.cpp:209] Iteration 187, loss = 0.577045
I1118 03:21:48.957855  8554 solver.cpp:464] Iteration 187, lr = 1e-05
I1118 03:21:51.218216  8554 solver.cpp:209] Iteration 188, loss = 0.559842
I1118 03:21:51.218243  8554 solver.cpp:464] Iteration 188, lr = 1e-05
I1118 03:21:53.475106  8554 solver.cpp:209] Iteration 189, loss = 0.790314
I1118 03:21:53.475134  8554 solver.cpp:464] Iteration 189, lr = 1e-05
I1118 03:21:55.734127  8554 solver.cpp:209] Iteration 190, loss = 0.633433
I1118 03:21:55.734166  8554 solver.cpp:464] Iteration 190, lr = 1e-05
I1118 03:21:57.992485  8554 solver.cpp:209] Iteration 191, loss = 0.631247
I1118 03:21:57.992583  8554 solver.cpp:464] Iteration 191, lr = 1e-05
I1118 03:22:00.249656  8554 solver.cpp:209] Iteration 192, loss = 0.532088
I1118 03:22:00.249685  8554 solver.cpp:464] Iteration 192, lr = 1e-05
I1118 03:22:02.503245  8554 solver.cpp:209] Iteration 193, loss = 0.516772
I1118 03:22:02.503273  8554 solver.cpp:464] Iteration 193, lr = 1e-05
I1118 03:22:04.757609  8554 solver.cpp:209] Iteration 194, loss = 0.686255
I1118 03:22:04.757650  8554 solver.cpp:464] Iteration 194, lr = 1e-05
I1118 03:22:07.017408  8554 solver.cpp:209] Iteration 195, loss = 0.807104
I1118 03:22:07.017448  8554 solver.cpp:464] Iteration 195, lr = 1e-05
I1118 03:22:09.281932  8554 solver.cpp:209] Iteration 196, loss = 0.413007
I1118 03:22:09.281973  8554 solver.cpp:464] Iteration 196, lr = 1e-05
I1118 03:22:11.544884  8554 solver.cpp:209] Iteration 197, loss = 0.802579
I1118 03:22:11.544925  8554 solver.cpp:464] Iteration 197, lr = 1e-05
I1118 03:22:13.806306  8554 solver.cpp:209] Iteration 198, loss = 0.565615
I1118 03:22:13.806347  8554 solver.cpp:464] Iteration 198, lr = 1e-05
I1118 03:22:16.058434  8554 solver.cpp:209] Iteration 199, loss = 0.513977
I1118 03:22:16.058460  8554 solver.cpp:464] Iteration 199, lr = 1e-05
I1118 03:22:16.059028  8554 solver.cpp:264] Iteration 200, Testing net (#0)
I1118 03:22:38.613054  8554 solver.cpp:305] Test loss: 0.461233
I1118 03:22:38.613111  8554 solver.cpp:318] mean_score = test_score[0] { = 780} / test_score[1] { = 868 }
I1118 03:22:38.613131  8554 solver.cpp:319]            = 0.898618
I1118 03:22:38.613137  8554 solver.cpp:328]     Test net output #0: accuracy = 0.898618
I1118 03:22:38.613142  8554 solver.cpp:318] mean_score = test_score[2] { = 29} / test_score[3] { = 100 }
I1118 03:22:38.613145  8554 solver.cpp:319]            = 0.29
I1118 03:22:38.613149  8554 solver.cpp:328]     Test net output #1: accuracy = 0.29
I1118 03:22:38.613153  8554 solver.cpp:332]     Test net output #2: accuracy = 0.835744
I1118 03:22:38.613157  8554 solver.cpp:334]     Test net output #3: accuracy = 0.594309
I1118 03:22:39.268391  8554 solver.cpp:209] Iteration 200, loss = 0.589274
I1118 03:22:39.268419  8554 solver.cpp:464] Iteration 200, lr = 1e-05
I1118 03:22:41.529971  8554 solver.cpp:209] Iteration 201, loss = 0.746691
I1118 03:22:41.530011  8554 solver.cpp:464] Iteration 201, lr = 1e-05
I1118 03:22:43.788590  8554 solver.cpp:209] Iteration 202, loss = 0.681807
I1118 03:22:43.788619  8554 solver.cpp:464] Iteration 202, lr = 1e-05
I1118 03:22:46.041815  8554 solver.cpp:209] Iteration 203, loss = 0.572606
I1118 03:22:46.041844  8554 solver.cpp:464] Iteration 203, lr = 1e-05
I1118 03:22:48.300640  8554 solver.cpp:209] Iteration 204, loss = 0.731783
I1118 03:22:48.300680  8554 solver.cpp:464] Iteration 204, lr = 1e-05
I1118 03:22:50.552736  8554 solver.cpp:209] Iteration 205, loss = 0.6095
I1118 03:22:50.552765  8554 solver.cpp:464] Iteration 205, lr = 1e-05
I1118 03:22:52.804911  8554 solver.cpp:209] Iteration 206, loss = 0.765633
I1118 03:22:52.804940  8554 solver.cpp:464] Iteration 206, lr = 1e-05
I1118 03:22:55.067996  8554 solver.cpp:209] Iteration 207, loss = 0.682131
I1118 03:22:55.068024  8554 solver.cpp:464] Iteration 207, lr = 1e-05
I1118 03:22:57.324586  8554 solver.cpp:209] Iteration 208, loss = 0.874067
I1118 03:22:57.324615  8554 solver.cpp:464] Iteration 208, lr = 1e-05
I1118 03:22:59.588528  8554 solver.cpp:209] Iteration 209, loss = 0.527574
I1118 03:22:59.588557  8554 solver.cpp:464] Iteration 209, lr = 1e-05
I1118 03:23:01.848263  8554 solver.cpp:209] Iteration 210, loss = 0.584248
I1118 03:23:01.848291  8554 solver.cpp:464] Iteration 210, lr = 1e-05
I1118 03:23:04.102463  8554 solver.cpp:209] Iteration 211, loss = 0.559992
I1118 03:23:04.102491  8554 solver.cpp:464] Iteration 211, lr = 1e-05
I1118 03:23:06.356534  8554 solver.cpp:209] Iteration 212, loss = 0.801214
I1118 03:23:06.356561  8554 solver.cpp:464] Iteration 212, lr = 1e-05
I1118 03:23:08.605412  8554 solver.cpp:209] Iteration 213, loss = 0.608286
I1118 03:23:08.605439  8554 solver.cpp:464] Iteration 213, lr = 1e-05
I1118 03:23:10.861887  8554 solver.cpp:209] Iteration 214, loss = 0.888953
I1118 03:23:10.861974  8554 solver.cpp:464] Iteration 214, lr = 1e-05
I1118 03:23:13.119192  8554 solver.cpp:209] Iteration 215, loss = 0.695942
I1118 03:23:13.119221  8554 solver.cpp:464] Iteration 215, lr = 1e-05
I1118 03:23:15.378949  8554 solver.cpp:209] Iteration 216, loss = 0.763928
I1118 03:23:15.378978  8554 solver.cpp:464] Iteration 216, lr = 1e-05
I1118 03:23:17.638499  8554 solver.cpp:209] Iteration 217, loss = 0.584292
I1118 03:23:17.638527  8554 solver.cpp:464] Iteration 217, lr = 1e-05
I1118 03:23:19.895011  8554 solver.cpp:209] Iteration 218, loss = 0.742425
I1118 03:23:19.895040  8554 solver.cpp:464] Iteration 218, lr = 1e-05
I1118 03:23:22.148862  8554 solver.cpp:209] Iteration 219, loss = 0.909775
I1118 03:23:22.148900  8554 solver.cpp:464] Iteration 219, lr = 1e-05
I1118 03:23:24.374944  8554 solver.cpp:209] Iteration 220, loss = 0.694024
I1118 03:23:24.374984  8554 solver.cpp:464] Iteration 220, lr = 1e-05
I1118 03:23:26.602108  8554 solver.cpp:209] Iteration 221, loss = 0.82295
I1118 03:23:26.602147  8554 solver.cpp:464] Iteration 221, lr = 1e-05
I1118 03:23:28.830340  8554 solver.cpp:209] Iteration 222, loss = 0.80881
I1118 03:23:28.830380  8554 solver.cpp:464] Iteration 222, lr = 1e-05
I1118 03:23:31.066439  8554 solver.cpp:209] Iteration 223, loss = 0.957608
I1118 03:23:31.066467  8554 solver.cpp:464] Iteration 223, lr = 1e-05
I1118 03:23:33.301942  8554 solver.cpp:209] Iteration 224, loss = 0.761966
I1118 03:23:33.301972  8554 solver.cpp:464] Iteration 224, lr = 1e-05
I1118 03:23:35.533752  8554 solver.cpp:209] Iteration 225, loss = 0.687877
I1118 03:23:35.533782  8554 solver.cpp:464] Iteration 225, lr = 1e-05
I1118 03:23:37.759387  8554 solver.cpp:209] Iteration 226, loss = 0.689607
I1118 03:23:37.759418  8554 solver.cpp:464] Iteration 226, lr = 1e-05
I1118 03:23:39.989073  8554 solver.cpp:209] Iteration 227, loss = 0.575878
I1118 03:23:39.989102  8554 solver.cpp:464] Iteration 227, lr = 1e-05
I1118 03:23:42.214551  8554 solver.cpp:209] Iteration 228, loss = 0.706075
I1118 03:23:42.214633  8554 solver.cpp:464] Iteration 228, lr = 1e-05
I1118 03:23:44.448447  8554 solver.cpp:209] Iteration 229, loss = 0.530834
I1118 03:23:44.448477  8554 solver.cpp:464] Iteration 229, lr = 1e-05
I1118 03:23:46.687333  8554 solver.cpp:209] Iteration 230, loss = 0.723767
I1118 03:23:46.687362  8554 solver.cpp:464] Iteration 230, lr = 1e-05
I1118 03:23:48.919992  8554 solver.cpp:209] Iteration 231, loss = 0.641829
I1118 03:23:48.920032  8554 solver.cpp:464] Iteration 231, lr = 1e-05
I1118 03:23:51.155364  8554 solver.cpp:209] Iteration 232, loss = 0.534306
I1118 03:23:51.155402  8554 solver.cpp:464] Iteration 232, lr = 1e-05
I1118 03:23:53.374897  8554 solver.cpp:209] Iteration 233, loss = 0.582777
I1118 03:23:53.374924  8554 solver.cpp:464] Iteration 233, lr = 1e-05
I1118 03:23:55.608407  8554 solver.cpp:209] Iteration 234, loss = 0.59449
I1118 03:23:55.608435  8554 solver.cpp:464] Iteration 234, lr = 1e-05
I1118 03:23:57.835808  8554 solver.cpp:209] Iteration 235, loss = 0.662564
I1118 03:23:57.835836  8554 solver.cpp:464] Iteration 235, lr = 1e-05
I1118 03:24:00.064841  8554 solver.cpp:209] Iteration 236, loss = 0.619201
I1118 03:24:00.064869  8554 solver.cpp:464] Iteration 236, lr = 1e-05
I1118 03:24:02.300568  8554 solver.cpp:209] Iteration 237, loss = 0.580076
I1118 03:24:02.300595  8554 solver.cpp:464] Iteration 237, lr = 1e-05
I1118 03:24:04.525035  8554 solver.cpp:209] Iteration 238, loss = 0.884881
I1118 03:24:04.525076  8554 solver.cpp:464] Iteration 238, lr = 1e-05
I1118 03:24:06.759536  8554 solver.cpp:209] Iteration 239, loss = 0.648914
I1118 03:24:06.759567  8554 solver.cpp:464] Iteration 239, lr = 1e-05
I1118 03:24:08.983434  8554 solver.cpp:209] Iteration 240, loss = 0.724048
I1118 03:24:08.983485  8554 solver.cpp:464] Iteration 240, lr = 1e-05
I1118 03:24:11.210552  8554 solver.cpp:209] Iteration 241, loss = 0.798418
I1118 03:24:11.210582  8554 solver.cpp:464] Iteration 241, lr = 1e-05
I1118 03:24:13.436784  8554 solver.cpp:209] Iteration 242, loss = 0.683154
I1118 03:24:13.436872  8554 solver.cpp:464] Iteration 242, lr = 1e-05
I1118 03:24:15.658848  8554 solver.cpp:209] Iteration 243, loss = 0.576914
I1118 03:24:15.658887  8554 solver.cpp:464] Iteration 243, lr = 1e-05
I1118 03:24:17.896636  8554 solver.cpp:209] Iteration 244, loss = 0.684326
I1118 03:24:17.896666  8554 solver.cpp:464] Iteration 244, lr = 1e-05
I1118 03:24:20.128695  8554 solver.cpp:209] Iteration 245, loss = 0.733456
I1118 03:24:20.128737  8554 solver.cpp:464] Iteration 245, lr = 1e-05
I1118 03:24:22.376667  8554 solver.cpp:209] Iteration 246, loss = 0.666945
I1118 03:24:22.376705  8554 solver.cpp:464] Iteration 246, lr = 1e-05
I1118 03:24:24.604176  8554 solver.cpp:209] Iteration 247, loss = 0.589648
I1118 03:24:24.604217  8554 solver.cpp:464] Iteration 247, lr = 1e-05
I1118 03:24:26.829357  8554 solver.cpp:209] Iteration 248, loss = 0.869257
I1118 03:24:26.829386  8554 solver.cpp:464] Iteration 248, lr = 1e-05
I1118 03:24:29.056059  8554 solver.cpp:209] Iteration 249, loss = 0.69247
I1118 03:24:29.056099  8554 solver.cpp:464] Iteration 249, lr = 1e-05
I1118 03:24:29.056674  8554 solver.cpp:264] Iteration 250, Testing net (#0)
I1118 03:24:51.430784  8554 solver.cpp:305] Test loss: 0.495841
I1118 03:24:51.430830  8554 solver.cpp:318] mean_score = test_score[0] { = 734} / test_score[1] { = 868 }
I1118 03:24:51.430848  8554 solver.cpp:319]            = 0.845622
I1118 03:24:51.430853  8554 solver.cpp:328]     Test net output #0: accuracy = 0.845622
I1118 03:24:51.430857  8554 solver.cpp:318] mean_score = test_score[2] { = 45} / test_score[3] { = 100 }
I1118 03:24:51.430862  8554 solver.cpp:319]            = 0.45
I1118 03:24:51.430866  8554 solver.cpp:328]     Test net output #1: accuracy = 0.45
I1118 03:24:51.430871  8554 solver.cpp:332]     Test net output #2: accuracy = 0.804752
I1118 03:24:51.430874  8554 solver.cpp:334]     Test net output #3: accuracy = 0.647811
I1118 03:24:52.085731  8554 solver.cpp:209] Iteration 250, loss = 0.614425
I1118 03:24:52.085759  8554 solver.cpp:464] Iteration 250, lr = 1e-05
I1118 03:24:54.334506  8554 solver.cpp:209] Iteration 251, loss = 0.81824
I1118 03:24:54.334544  8554 solver.cpp:464] Iteration 251, lr = 1e-05
I1118 03:24:56.589046  8554 solver.cpp:209] Iteration 252, loss = 0.64529
I1118 03:24:56.589085  8554 solver.cpp:464] Iteration 252, lr = 1e-05
I1118 03:24:58.858202  8554 solver.cpp:209] Iteration 253, loss = 0.638192
I1118 03:24:58.858242  8554 solver.cpp:464] Iteration 253, lr = 1e-05
I1118 03:25:01.116665  8554 solver.cpp:209] Iteration 254, loss = 0.623412
I1118 03:25:01.116693  8554 solver.cpp:464] Iteration 254, lr = 1e-05
I1118 03:25:03.375874  8554 solver.cpp:209] Iteration 255, loss = 0.687052
I1118 03:25:03.375915  8554 solver.cpp:464] Iteration 255, lr = 1e-05
I1118 03:25:05.639895  8554 solver.cpp:209] Iteration 256, loss = 0.541535
I1118 03:25:05.639937  8554 solver.cpp:464] Iteration 256, lr = 1e-05
I1118 03:25:07.893801  8554 solver.cpp:209] Iteration 257, loss = 0.781727
I1118 03:25:07.893831  8554 solver.cpp:464] Iteration 257, lr = 1e-05
I1118 03:25:10.148849  8554 solver.cpp:209] Iteration 258, loss = 0.79139
I1118 03:25:10.148876  8554 solver.cpp:464] Iteration 258, lr = 1e-05
I1118 03:25:12.399879  8554 solver.cpp:209] Iteration 259, loss = 0.617822
I1118 03:25:12.399906  8554 solver.cpp:464] Iteration 259, lr = 1e-05
I1118 03:25:14.625637  8554 solver.cpp:209] Iteration 260, loss = 0.674969
I1118 03:25:14.625676  8554 solver.cpp:464] Iteration 260, lr = 1e-05
I1118 03:25:16.865142  8554 solver.cpp:209] Iteration 261, loss = 0.929184
I1118 03:25:16.865181  8554 solver.cpp:464] Iteration 261, lr = 1e-05
I1118 03:25:19.449482  8554 solver.cpp:209] Iteration 262, loss = 0.564412
I1118 03:25:19.449512  8554 solver.cpp:464] Iteration 262, lr = 1e-05
I1118 03:25:21.550411  8554 solver.cpp:209] Iteration 263, loss = 0.819493
I1118 03:25:21.550498  8554 solver.cpp:464] Iteration 263, lr = 1e-05
I1118 03:25:23.680016  8554 solver.cpp:209] Iteration 264, loss = 0.559574
I1118 03:25:23.680055  8554 solver.cpp:464] Iteration 264, lr = 1e-05
I1118 03:25:25.852301  8554 solver.cpp:209] Iteration 265, loss = 0.7291
I1118 03:25:25.852341  8554 solver.cpp:464] Iteration 265, lr = 1e-05
I1118 03:25:28.070020  8554 solver.cpp:209] Iteration 266, loss = 0.894059
I1118 03:25:28.070061  8554 solver.cpp:464] Iteration 266, lr = 1e-05
I1118 03:25:30.296507  8554 solver.cpp:209] Iteration 267, loss = 0.627043
I1118 03:25:30.296548  8554 solver.cpp:464] Iteration 267, lr = 1e-05
I1118 03:25:32.527541  8554 solver.cpp:209] Iteration 268, loss = 0.576935
I1118 03:25:32.527571  8554 solver.cpp:464] Iteration 268, lr = 1e-05
I1118 03:25:34.756477  8554 solver.cpp:209] Iteration 269, loss = 0.733322
I1118 03:25:34.756505  8554 solver.cpp:464] Iteration 269, lr = 1e-05
I1118 03:25:36.988551  8554 solver.cpp:209] Iteration 270, loss = 0.484172
I1118 03:25:36.988581  8554 solver.cpp:464] Iteration 270, lr = 1e-05
I1118 03:25:39.216114  8554 solver.cpp:209] Iteration 271, loss = 1.08635
I1118 03:25:39.216141  8554 solver.cpp:464] Iteration 271, lr = 1e-05
I1118 03:25:41.442085  8554 solver.cpp:209] Iteration 272, loss = 0.624185
I1118 03:25:41.442113  8554 solver.cpp:464] Iteration 272, lr = 1e-05
I1118 03:25:43.669622  8554 solver.cpp:209] Iteration 273, loss = 1.06571
I1118 03:25:43.669651  8554 solver.cpp:464] Iteration 273, lr = 1e-05
I1118 03:25:45.893110  8554 solver.cpp:209] Iteration 274, loss = 0.542878
I1118 03:25:45.893139  8554 solver.cpp:464] Iteration 274, lr = 1e-05
I1118 03:25:48.125202  8554 solver.cpp:209] Iteration 275, loss = 0.639129
I1118 03:25:48.125231  8554 solver.cpp:464] Iteration 275, lr = 1e-05
I1118 03:25:50.360932  8554 solver.cpp:209] Iteration 276, loss = 0.556874
I1118 03:25:50.360960  8554 solver.cpp:464] Iteration 276, lr = 1e-05
I1118 03:25:52.591799  8554 solver.cpp:209] Iteration 277, loss = 0.511861
I1118 03:25:52.591883  8554 solver.cpp:464] Iteration 277, lr = 1e-05
I1118 03:25:54.822015  8554 solver.cpp:209] Iteration 278, loss = 0.47402
I1118 03:25:54.822043  8554 solver.cpp:464] Iteration 278, lr = 1e-05
I1118 03:25:57.042538  8554 solver.cpp:209] Iteration 279, loss = 0.825415
I1118 03:25:57.042579  8554 solver.cpp:464] Iteration 279, lr = 1e-05
I1118 03:25:59.271024  8554 solver.cpp:209] Iteration 280, loss = 0.700624
I1118 03:25:59.271052  8554 solver.cpp:464] Iteration 280, lr = 1e-05
I1118 03:26:01.503134  8554 solver.cpp:209] Iteration 281, loss = 0.626509
I1118 03:26:01.503165  8554 solver.cpp:464] Iteration 281, lr = 1e-05
I1118 03:26:03.740391  8554 solver.cpp:209] Iteration 282, loss = 0.665349
I1118 03:26:03.740422  8554 solver.cpp:464] Iteration 282, lr = 1e-05
I1118 03:26:05.976414  8554 solver.cpp:209] Iteration 283, loss = 0.711566
I1118 03:26:05.976444  8554 solver.cpp:464] Iteration 283, lr = 1e-05
I1118 03:26:08.205818  8554 solver.cpp:209] Iteration 284, loss = 0.597195
I1118 03:26:08.205860  8554 solver.cpp:464] Iteration 284, lr = 1e-05
I1118 03:26:10.429011  8554 solver.cpp:209] Iteration 285, loss = 0.471664
I1118 03:26:10.429040  8554 solver.cpp:464] Iteration 285, lr = 1e-05
I1118 03:26:12.656780  8554 solver.cpp:209] Iteration 286, loss = 0.583667
I1118 03:26:12.656810  8554 solver.cpp:464] Iteration 286, lr = 1e-05
I1118 03:26:14.882318  8554 solver.cpp:209] Iteration 287, loss = 0.765059
I1118 03:26:14.882347  8554 solver.cpp:464] Iteration 287, lr = 1e-05
I1118 03:26:17.110337  8554 solver.cpp:209] Iteration 288, loss = 0.565552
I1118 03:26:17.110379  8554 solver.cpp:464] Iteration 288, lr = 1e-05
I1118 03:26:19.316112  8554 solver.cpp:209] Iteration 289, loss = 0.598076
I1118 03:26:19.316140  8554 solver.cpp:464] Iteration 289, lr = 1e-05
I1118 03:26:21.519418  8554 solver.cpp:209] Iteration 290, loss = 0.571559
I1118 03:26:21.519448  8554 solver.cpp:464] Iteration 290, lr = 1e-05
I1118 03:26:23.724849  8554 solver.cpp:209] Iteration 291, loss = 0.800812
I1118 03:26:23.724932  8554 solver.cpp:464] Iteration 291, lr = 1e-05
I1118 03:26:25.926895  8554 solver.cpp:209] Iteration 292, loss = 0.726174
I1118 03:26:25.926936  8554 solver.cpp:464] Iteration 292, lr = 1e-05
I1118 03:26:28.127702  8554 solver.cpp:209] Iteration 293, loss = 0.46198
I1118 03:26:28.127730  8554 solver.cpp:464] Iteration 293, lr = 1e-05
I1118 03:26:30.323065  8554 solver.cpp:209] Iteration 294, loss = 0.841278
I1118 03:26:30.323106  8554 solver.cpp:464] Iteration 294, lr = 1e-05
I1118 03:26:32.527102  8554 solver.cpp:209] Iteration 295, loss = 0.492882
I1118 03:26:32.527132  8554 solver.cpp:464] Iteration 295, lr = 1e-05
I1118 03:26:34.731580  8554 solver.cpp:209] Iteration 296, loss = 0.57967
I1118 03:26:34.731608  8554 solver.cpp:464] Iteration 296, lr = 1e-05
I1118 03:26:36.938709  8554 solver.cpp:209] Iteration 297, loss = 0.460163
I1118 03:26:36.938737  8554 solver.cpp:464] Iteration 297, lr = 1e-05
I1118 03:26:39.143206  8554 solver.cpp:209] Iteration 298, loss = 0.907462
I1118 03:26:39.143236  8554 solver.cpp:464] Iteration 298, lr = 1e-05
I1118 03:26:41.337852  8554 solver.cpp:209] Iteration 299, loss = 0.788113
I1118 03:26:41.337879  8554 solver.cpp:464] Iteration 299, lr = 1e-05
I1118 03:26:41.338495  8554 solver.cpp:264] Iteration 300, Testing net (#0)
I1118 03:27:03.799412  8554 solver.cpp:305] Test loss: 0.424099
I1118 03:27:03.799499  8554 solver.cpp:318] mean_score = test_score[0] { = 805} / test_score[1] { = 868 }
I1118 03:27:03.799507  8554 solver.cpp:319]            = 0.927419
I1118 03:27:03.799512  8554 solver.cpp:328]     Test net output #0: accuracy = 0.927419
I1118 03:27:03.799516  8554 solver.cpp:318] mean_score = test_score[2] { = 19} / test_score[3] { = 100 }
I1118 03:27:03.799521  8554 solver.cpp:319]            = 0.19
I1118 03:27:03.799525  8554 solver.cpp:328]     Test net output #1: accuracy = 0.19
I1118 03:27:03.799530  8554 solver.cpp:332]     Test net output #2: accuracy = 0.85124
I1118 03:27:03.799533  8554 solver.cpp:334]     Test net output #3: accuracy = 0.55871
I1118 03:27:04.462705  8554 solver.cpp:209] Iteration 300, loss = 0.704249
I1118 03:27:04.462735  8554 solver.cpp:464] Iteration 300, lr = 1e-05
I1118 03:27:06.744366  8554 solver.cpp:209] Iteration 301, loss = 0.636188
I1118 03:27:06.744395  8554 solver.cpp:464] Iteration 301, lr = 1e-05
I1118 03:27:09.028128  8554 solver.cpp:209] Iteration 302, loss = 0.678433
I1118 03:27:09.028156  8554 solver.cpp:464] Iteration 302, lr = 1e-05
I1118 03:27:11.308779  8554 solver.cpp:209] Iteration 303, loss = 0.80235
I1118 03:27:11.308807  8554 solver.cpp:464] Iteration 303, lr = 1e-05
I1118 03:27:13.590461  8554 solver.cpp:209] Iteration 304, loss = 0.728543
I1118 03:27:13.590490  8554 solver.cpp:464] Iteration 304, lr = 1e-05
I1118 03:27:15.837529  8554 solver.cpp:209] Iteration 305, loss = 0.838715
I1118 03:27:15.837558  8554 solver.cpp:464] Iteration 305, lr = 1e-05
I1118 03:27:18.067865  8554 solver.cpp:209] Iteration 306, loss = 0.681227
I1118 03:27:18.067893  8554 solver.cpp:464] Iteration 306, lr = 1e-05
I1118 03:27:20.272294  8554 solver.cpp:209] Iteration 307, loss = 0.85163
I1118 03:27:20.272322  8554 solver.cpp:464] Iteration 307, lr = 1e-05
I1118 03:27:22.470247  8554 solver.cpp:209] Iteration 308, loss = 0.586323
I1118 03:27:22.470274  8554 solver.cpp:464] Iteration 308, lr = 1e-05
I1118 03:27:24.676126  8554 solver.cpp:209] Iteration 309, loss = 0.498327
I1118 03:27:24.676153  8554 solver.cpp:464] Iteration 309, lr = 1e-05
I1118 03:27:26.871799  8554 solver.cpp:209] Iteration 310, loss = 0.762001
I1118 03:27:26.871827  8554 solver.cpp:464] Iteration 310, lr = 1e-05
I1118 03:27:29.081377  8554 solver.cpp:209] Iteration 311, loss = 0.6432
I1118 03:27:29.081404  8554 solver.cpp:464] Iteration 311, lr = 1e-05
I1118 03:27:31.289208  8554 solver.cpp:209] Iteration 312, loss = 0.594527
I1118 03:27:31.289237  8554 solver.cpp:464] Iteration 312, lr = 1e-05
I1118 03:27:33.490700  8554 solver.cpp:209] Iteration 313, loss = 0.524269
I1118 03:27:33.490730  8554 solver.cpp:464] Iteration 313, lr = 1e-05
I1118 03:27:35.695139  8554 solver.cpp:209] Iteration 314, loss = 0.609887
I1118 03:27:35.695221  8554 solver.cpp:464] Iteration 314, lr = 1e-05
I1118 03:27:37.890821  8554 solver.cpp:209] Iteration 315, loss = 0.498928
I1118 03:27:37.890861  8554 solver.cpp:464] Iteration 315, lr = 1e-05
I1118 03:27:40.095640  8554 solver.cpp:209] Iteration 316, loss = 0.699011
I1118 03:27:40.095669  8554 solver.cpp:464] Iteration 316, lr = 1e-05
I1118 03:27:42.299437  8554 solver.cpp:209] Iteration 317, loss = 0.562919
I1118 03:27:42.299486  8554 solver.cpp:464] Iteration 317, lr = 1e-05
I1118 03:27:44.503196  8554 solver.cpp:209] Iteration 318, loss = 0.786216
I1118 03:27:44.503235  8554 solver.cpp:464] Iteration 318, lr = 1e-05
I1118 03:27:46.712630  8554 solver.cpp:209] Iteration 319, loss = 0.767346
I1118 03:27:46.712658  8554 solver.cpp:464] Iteration 319, lr = 1e-05
I1118 03:27:48.915827  8554 solver.cpp:209] Iteration 320, loss = 0.624011
I1118 03:27:48.915868  8554 solver.cpp:464] Iteration 320, lr = 1e-05
I1118 03:27:51.145043  8554 solver.cpp:209] Iteration 321, loss = 0.531443
I1118 03:27:51.145071  8554 solver.cpp:464] Iteration 321, lr = 1e-05
I1118 03:27:53.369320  8554 solver.cpp:209] Iteration 322, loss = 0.637898
I1118 03:27:53.369350  8554 solver.cpp:464] Iteration 322, lr = 1e-05
I1118 03:27:55.597038  8554 solver.cpp:209] Iteration 323, loss = 0.520061
I1118 03:27:55.597077  8554 solver.cpp:464] Iteration 323, lr = 1e-05
I1118 03:27:57.835387  8554 solver.cpp:209] Iteration 324, loss = 0.57661
I1118 03:27:57.835418  8554 solver.cpp:464] Iteration 324, lr = 1e-05
I1118 03:28:00.060760  8554 solver.cpp:209] Iteration 325, loss = 0.644927
I1118 03:28:00.060801  8554 solver.cpp:464] Iteration 325, lr = 1e-05
I1118 03:28:02.313182  8554 solver.cpp:209] Iteration 326, loss = 0.730958
I1118 03:28:02.313222  8554 solver.cpp:464] Iteration 326, lr = 1e-05
I1118 03:28:04.541674  8554 solver.cpp:209] Iteration 327, loss = 0.648524
I1118 03:28:04.541715  8554 solver.cpp:464] Iteration 327, lr = 1e-05
I1118 03:28:06.765233  8554 solver.cpp:209] Iteration 328, loss = 0.844593
I1118 03:28:06.765318  8554 solver.cpp:464] Iteration 328, lr = 1e-05
I1118 03:28:08.992254  8554 solver.cpp:209] Iteration 329, loss = 0.566582
I1118 03:28:08.992281  8554 solver.cpp:464] Iteration 329, lr = 1e-05
I1118 03:28:11.217984  8554 solver.cpp:209] Iteration 330, loss = 0.595417
I1118 03:28:11.218022  8554 solver.cpp:464] Iteration 330, lr = 1e-05
I1118 03:28:13.454813  8554 solver.cpp:209] Iteration 331, loss = 0.754397
I1118 03:28:13.454841  8554 solver.cpp:464] Iteration 331, lr = 1e-05
I1118 03:28:15.687080  8554 solver.cpp:209] Iteration 332, loss = 0.533241
I1118 03:28:15.687121  8554 solver.cpp:464] Iteration 332, lr = 1e-05
I1118 03:28:17.919826  8554 solver.cpp:209] Iteration 333, loss = 0.444913
I1118 03:28:17.919855  8554 solver.cpp:464] Iteration 333, lr = 1e-05
I1118 03:28:20.146420  8554 solver.cpp:209] Iteration 334, loss = 0.556693
I1118 03:28:20.146461  8554 solver.cpp:464] Iteration 334, lr = 1e-05
I1118 03:28:22.369297  8554 solver.cpp:209] Iteration 335, loss = 0.538775
I1118 03:28:22.369324  8554 solver.cpp:464] Iteration 335, lr = 1e-05
I1118 03:28:24.595710  8554 solver.cpp:209] Iteration 336, loss = 0.697205
I1118 03:28:24.595739  8554 solver.cpp:464] Iteration 336, lr = 1e-05
I1118 03:28:26.822710  8554 solver.cpp:209] Iteration 337, loss = 0.608471
I1118 03:28:26.822739  8554 solver.cpp:464] Iteration 337, lr = 1e-05
I1118 03:28:29.025957  8554 solver.cpp:209] Iteration 338, loss = 0.651681
I1118 03:28:29.025986  8554 solver.cpp:464] Iteration 338, lr = 1e-05
I1118 03:28:31.239064  8554 solver.cpp:209] Iteration 339, loss = 0.677192
I1118 03:28:31.239092  8554 solver.cpp:464] Iteration 339, lr = 1e-05
I1118 03:28:33.433714  8554 solver.cpp:209] Iteration 340, loss = 0.707829
I1118 03:28:33.433743  8554 solver.cpp:464] Iteration 340, lr = 1e-05
I1118 03:28:35.636276  8554 solver.cpp:209] Iteration 341, loss = 0.600776
I1118 03:28:35.636304  8554 solver.cpp:464] Iteration 341, lr = 1e-05
I1118 03:28:37.832605  8554 solver.cpp:209] Iteration 342, loss = 0.62256
I1118 03:28:37.832691  8554 solver.cpp:464] Iteration 342, lr = 1e-05
I1118 03:28:40.033182  8554 solver.cpp:209] Iteration 343, loss = 0.819419
I1118 03:28:40.033210  8554 solver.cpp:464] Iteration 343, lr = 1e-05
I1118 03:28:42.239408  8554 solver.cpp:209] Iteration 344, loss = 0.606887
I1118 03:28:42.239436  8554 solver.cpp:464] Iteration 344, lr = 1e-05
I1118 03:28:44.441061  8554 solver.cpp:209] Iteration 345, loss = 0.397556
I1118 03:28:44.441092  8554 solver.cpp:464] Iteration 345, lr = 1e-05
I1118 03:28:46.642720  8554 solver.cpp:209] Iteration 346, loss = 0.537832
I1118 03:28:46.642750  8554 solver.cpp:464] Iteration 346, lr = 1e-05
I1118 03:28:48.844537  8554 solver.cpp:209] Iteration 347, loss = 0.64198
I1118 03:28:48.844566  8554 solver.cpp:464] Iteration 347, lr = 1e-05
I1118 03:28:51.039248  8554 solver.cpp:209] Iteration 348, loss = 0.744205
I1118 03:28:51.039276  8554 solver.cpp:464] Iteration 348, lr = 1e-05
I1118 03:28:53.248652  8554 solver.cpp:209] Iteration 349, loss = 0.661693
I1118 03:28:53.248692  8554 solver.cpp:464] Iteration 349, lr = 1e-05
I1118 03:28:53.249274  8554 solver.cpp:264] Iteration 350, Testing net (#0)
I1118 03:29:15.450839  8554 solver.cpp:305] Test loss: 0.465466
I1118 03:29:15.450898  8554 solver.cpp:318] mean_score = test_score[0] { = 763} / test_score[1] { = 868 }
I1118 03:29:15.450906  8554 solver.cpp:319]            = 0.879032
I1118 03:29:15.450911  8554 solver.cpp:328]     Test net output #0: accuracy = 0.879032
I1118 03:29:15.450916  8554 solver.cpp:318] mean_score = test_score[2] { = 28} / test_score[3] { = 100 }
I1118 03:29:15.450920  8554 solver.cpp:319]            = 0.28
I1118 03:29:15.450923  8554 solver.cpp:328]     Test net output #1: accuracy = 0.28
I1118 03:29:15.450928  8554 solver.cpp:332]     Test net output #2: accuracy = 0.817149
I1118 03:29:15.450932  8554 solver.cpp:334]     Test net output #3: accuracy = 0.579516
I1118 03:29:16.100733  8554 solver.cpp:209] Iteration 350, loss = 0.526978
I1118 03:29:16.100759  8554 solver.cpp:464] Iteration 350, lr = 1e-05
I1118 03:29:18.365751  8554 solver.cpp:209] Iteration 351, loss = 0.82347
I1118 03:29:18.365790  8554 solver.cpp:464] Iteration 351, lr = 1e-05
I1118 03:29:20.619099  8554 solver.cpp:209] Iteration 352, loss = 0.589695
I1118 03:29:20.619127  8554 solver.cpp:464] Iteration 352, lr = 1e-05
I1118 03:29:22.881145  8554 solver.cpp:209] Iteration 353, loss = 0.581322
I1118 03:29:22.881183  8554 solver.cpp:464] Iteration 353, lr = 1e-05
I1118 03:29:25.139626  8554 solver.cpp:209] Iteration 354, loss = 0.586009
I1118 03:29:25.139653  8554 solver.cpp:464] Iteration 354, lr = 1e-05
I1118 03:29:27.398247  8554 solver.cpp:209] Iteration 355, loss = 0.796381
I1118 03:29:27.398286  8554 solver.cpp:464] Iteration 355, lr = 1e-05
I1118 03:29:29.653743  8554 solver.cpp:209] Iteration 356, loss = 0.628915
I1118 03:29:29.653770  8554 solver.cpp:464] Iteration 356, lr = 1e-05
I1118 03:29:31.912514  8554 solver.cpp:209] Iteration 357, loss = 0.612444
I1118 03:29:31.912542  8554 solver.cpp:464] Iteration 357, lr = 1e-05
I1118 03:29:34.139384  8554 solver.cpp:209] Iteration 358, loss = 0.992649
I1118 03:29:34.139415  8554 solver.cpp:464] Iteration 358, lr = 1e-05
I1118 03:29:36.348460  8554 solver.cpp:209] Iteration 359, loss = 0.68645
I1118 03:29:36.348486  8554 solver.cpp:464] Iteration 359, lr = 1e-05
I1118 03:29:38.557615  8554 solver.cpp:209] Iteration 360, loss = 0.69007
I1118 03:29:38.557642  8554 solver.cpp:464] Iteration 360, lr = 1e-05
I1118 03:29:40.761543  8554 solver.cpp:209] Iteration 361, loss = 0.858404
I1118 03:29:40.761571  8554 solver.cpp:464] Iteration 361, lr = 1e-05
I1118 03:29:42.958003  8554 solver.cpp:209] Iteration 362, loss = 0.541103
I1118 03:29:42.958044  8554 solver.cpp:464] Iteration 362, lr = 1e-05
I1118 03:29:45.160889  8554 solver.cpp:209] Iteration 363, loss = 0.736376
I1118 03:29:45.160918  8554 solver.cpp:464] Iteration 363, lr = 1e-05
I1118 03:29:47.388352  8554 solver.cpp:209] Iteration 364, loss = 0.575088
I1118 03:29:47.388434  8554 solver.cpp:464] Iteration 364, lr = 1e-05
I1118 03:29:49.602656  8554 solver.cpp:209] Iteration 365, loss = 0.710157
I1118 03:29:49.602697  8554 solver.cpp:464] Iteration 365, lr = 1e-05
I1118 03:29:51.812331  8554 solver.cpp:209] Iteration 366, loss = 0.526758
I1118 03:29:51.812362  8554 solver.cpp:464] Iteration 366, lr = 1e-05
I1118 03:29:54.012064  8554 solver.cpp:209] Iteration 367, loss = 0.744223
I1118 03:29:54.012104  8554 solver.cpp:464] Iteration 367, lr = 1e-05
I1118 03:29:56.212982  8554 solver.cpp:209] Iteration 368, loss = 0.489539
I1118 03:29:56.213009  8554 solver.cpp:464] Iteration 368, lr = 1e-05
I1118 03:29:58.421975  8554 solver.cpp:209] Iteration 369, loss = 0.72873
I1118 03:29:58.422015  8554 solver.cpp:464] Iteration 369, lr = 1e-05
I1118 03:30:00.630336  8554 solver.cpp:209] Iteration 370, loss = 0.643879
I1118 03:30:00.630377  8554 solver.cpp:464] Iteration 370, lr = 1e-05
I1118 03:30:02.841595  8554 solver.cpp:209] Iteration 371, loss = 0.7387
I1118 03:30:02.841637  8554 solver.cpp:464] Iteration 371, lr = 1e-05
I1118 03:30:05.043570  8554 solver.cpp:209] Iteration 372, loss = 0.568345
I1118 03:30:05.043598  8554 solver.cpp:464] Iteration 372, lr = 1e-05
I1118 03:30:07.251441  8554 solver.cpp:209] Iteration 373, loss = 0.512062
I1118 03:30:07.251489  8554 solver.cpp:464] Iteration 373, lr = 1e-05
I1118 03:30:09.452949  8554 solver.cpp:209] Iteration 374, loss = 0.927315
I1118 03:30:09.452990  8554 solver.cpp:464] Iteration 374, lr = 1e-05
I1118 03:30:11.654425  8554 solver.cpp:209] Iteration 375, loss = 0.677912
I1118 03:30:11.654453  8554 solver.cpp:464] Iteration 375, lr = 1e-05
I1118 03:30:13.854534  8554 solver.cpp:209] Iteration 376, loss = 0.41587
I1118 03:30:13.854574  8554 solver.cpp:464] Iteration 376, lr = 1e-05
I1118 03:30:16.055310  8554 solver.cpp:209] Iteration 377, loss = 0.598701
I1118 03:30:16.055335  8554 solver.cpp:464] Iteration 377, lr = 1e-05
I1118 03:30:18.263324  8554 solver.cpp:209] Iteration 378, loss = 0.792494
I1118 03:30:18.263404  8554 solver.cpp:464] Iteration 378, lr = 1e-05
I1118 03:30:20.468729  8554 solver.cpp:209] Iteration 379, loss = 0.670401
I1118 03:30:20.468755  8554 solver.cpp:464] Iteration 379, lr = 1e-05
I1118 03:30:22.671744  8554 solver.cpp:209] Iteration 380, loss = 0.618253
I1118 03:30:22.671784  8554 solver.cpp:464] Iteration 380, lr = 1e-05
I1118 03:30:24.873610  8554 solver.cpp:209] Iteration 381, loss = 0.730925
I1118 03:30:24.873649  8554 solver.cpp:464] Iteration 381, lr = 1e-05
I1118 03:30:27.073490  8554 solver.cpp:209] Iteration 382, loss = 0.678902
I1118 03:30:27.073519  8554 solver.cpp:464] Iteration 382, lr = 1e-05
I1118 03:30:29.290676  8554 solver.cpp:209] Iteration 383, loss = 0.525166
I1118 03:30:29.290715  8554 solver.cpp:464] Iteration 383, lr = 1e-05
I1118 03:30:31.499105  8554 solver.cpp:209] Iteration 384, loss = 0.831321
I1118 03:30:31.499135  8554 solver.cpp:464] Iteration 384, lr = 1e-05
I1118 03:30:33.713865  8554 solver.cpp:209] Iteration 385, loss = 0.493687
I1118 03:30:33.713907  8554 solver.cpp:464] Iteration 385, lr = 1e-05
I1118 03:30:35.944169  8554 solver.cpp:209] Iteration 386, loss = 0.545659
I1118 03:30:35.944200  8554 solver.cpp:464] Iteration 386, lr = 1e-05
I1118 03:30:38.164723  8554 solver.cpp:209] Iteration 387, loss = 0.772695
I1118 03:30:38.164764  8554 solver.cpp:464] Iteration 387, lr = 1e-05
I1118 03:30:40.398061  8554 solver.cpp:209] Iteration 388, loss = 0.885698
I1118 03:30:40.398102  8554 solver.cpp:464] Iteration 388, lr = 1e-05
I1118 03:30:42.630422  8554 solver.cpp:209] Iteration 389, loss = 0.937773
I1118 03:30:42.630452  8554 solver.cpp:464] Iteration 389, lr = 1e-05
I1118 03:30:44.863605  8554 solver.cpp:209] Iteration 390, loss = 0.61999
I1118 03:30:44.863636  8554 solver.cpp:464] Iteration 390, lr = 1e-05
I1118 03:30:47.096747  8554 solver.cpp:209] Iteration 391, loss = 0.94942
I1118 03:30:47.096776  8554 solver.cpp:464] Iteration 391, lr = 1e-05
I1118 03:30:49.326141  8554 solver.cpp:209] Iteration 392, loss = 0.602389
I1118 03:30:49.326228  8554 solver.cpp:464] Iteration 392, lr = 1e-05
I1118 03:30:51.557169  8554 solver.cpp:209] Iteration 393, loss = 0.680083
I1118 03:30:51.557209  8554 solver.cpp:464] Iteration 393, lr = 1e-05
I1118 03:30:53.780789  8554 solver.cpp:209] Iteration 394, loss = 0.629785
I1118 03:30:53.780817  8554 solver.cpp:464] Iteration 394, lr = 1e-05
I1118 03:30:56.009747  8554 solver.cpp:209] Iteration 395, loss = 0.474044
I1118 03:30:56.009788  8554 solver.cpp:464] Iteration 395, lr = 1e-05
I1118 03:30:58.244169  8554 solver.cpp:209] Iteration 396, loss = 0.667664
I1118 03:30:58.244197  8554 solver.cpp:464] Iteration 396, lr = 1e-05
I1118 03:31:00.474839  8554 solver.cpp:209] Iteration 397, loss = 0.741673
I1118 03:31:00.474867  8554 solver.cpp:464] Iteration 397, lr = 1e-05
I1118 03:31:02.710450  8554 solver.cpp:209] Iteration 398, loss = 0.567115
I1118 03:31:02.710479  8554 solver.cpp:464] Iteration 398, lr = 1e-05
I1118 03:31:04.944264  8554 solver.cpp:209] Iteration 399, loss = 0.534841
I1118 03:31:04.944305  8554 solver.cpp:464] Iteration 399, lr = 1e-05
I1118 03:31:04.944880  8554 solver.cpp:264] Iteration 400, Testing net (#0)
I1118 03:31:26.995223  8554 solver.cpp:305] Test loss: 0.428758
I1118 03:31:26.995270  8554 solver.cpp:318] mean_score = test_score[0] { = 809} / test_score[1] { = 868 }
I1118 03:31:26.995278  8554 solver.cpp:319]            = 0.932028
I1118 03:31:26.995282  8554 solver.cpp:328]     Test net output #0: accuracy = 0.932028
I1118 03:31:26.995286  8554 solver.cpp:318] mean_score = test_score[2] { = 29} / test_score[3] { = 100 }
I1118 03:31:26.995291  8554 solver.cpp:319]            = 0.29
I1118 03:31:26.995296  8554 solver.cpp:328]     Test net output #1: accuracy = 0.29
I1118 03:31:26.995300  8554 solver.cpp:332]     Test net output #2: accuracy = 0.865702
I1118 03:31:26.995303  8554 solver.cpp:334]     Test net output #3: accuracy = 0.611014
I1118 03:31:27.636718  8554 solver.cpp:209] Iteration 400, loss = 0.673744
I1118 03:31:27.636744  8554 solver.cpp:464] Iteration 400, lr = 1e-05
I1118 03:31:29.895069  8554 solver.cpp:209] Iteration 401, loss = 0.860284
I1118 03:31:29.895109  8554 solver.cpp:464] Iteration 401, lr = 1e-05
I1118 03:31:32.154260  8554 solver.cpp:209] Iteration 402, loss = 0.59792
I1118 03:31:32.154299  8554 solver.cpp:464] Iteration 402, lr = 1e-05
I1118 03:31:34.411175  8554 solver.cpp:209] Iteration 403, loss = 0.711338
I1118 03:31:34.411202  8554 solver.cpp:464] Iteration 403, lr = 1e-05
I1118 03:31:36.671666  8554 solver.cpp:209] Iteration 404, loss = 0.657271
I1118 03:31:36.671696  8554 solver.cpp:464] Iteration 404, lr = 1e-05
I1118 03:31:38.933646  8554 solver.cpp:209] Iteration 405, loss = 0.583997
I1118 03:31:38.933673  8554 solver.cpp:464] Iteration 405, lr = 1e-05
I1118 03:31:41.198735  8554 solver.cpp:209] Iteration 406, loss = 0.542669
I1118 03:31:41.198763  8554 solver.cpp:464] Iteration 406, lr = 1e-05
I1118 03:31:43.486414  8554 solver.cpp:209] Iteration 407, loss = 0.557893
I1118 03:31:43.486443  8554 solver.cpp:464] Iteration 407, lr = 1e-05
I1118 03:31:45.722218  8554 solver.cpp:209] Iteration 408, loss = 0.533754
I1118 03:31:45.722257  8554 solver.cpp:464] Iteration 408, lr = 1e-05
I1118 03:31:47.948791  8554 solver.cpp:209] Iteration 409, loss = 0.634398
I1118 03:31:47.948819  8554 solver.cpp:464] Iteration 409, lr = 1e-05
I1118 03:31:50.167742  8554 solver.cpp:209] Iteration 410, loss = 0.717752
I1118 03:31:50.167784  8554 solver.cpp:464] Iteration 410, lr = 1e-05
I1118 03:31:52.356148  8554 solver.cpp:209] Iteration 411, loss = 0.729104
I1118 03:31:52.356175  8554 solver.cpp:464] Iteration 411, lr = 1e-05
I1118 03:31:54.539345  8554 solver.cpp:209] Iteration 412, loss = 0.532602
I1118 03:31:54.539374  8554 solver.cpp:464] Iteration 412, lr = 1e-05
I1118 03:31:56.742846  8554 solver.cpp:209] Iteration 413, loss = 0.7168
I1118 03:31:56.742887  8554 solver.cpp:464] Iteration 413, lr = 1e-05
I1118 03:31:58.918582  8554 solver.cpp:209] Iteration 414, loss = 0.573593
I1118 03:31:58.918670  8554 solver.cpp:464] Iteration 414, lr = 1e-05
I1118 03:32:01.089921  8554 solver.cpp:209] Iteration 415, loss = 0.683027
I1118 03:32:01.089949  8554 solver.cpp:464] Iteration 415, lr = 1e-05
I1118 03:32:03.273780  8554 solver.cpp:209] Iteration 416, loss = 0.60356
I1118 03:32:03.273809  8554 solver.cpp:464] Iteration 416, lr = 1e-05
I1118 03:32:05.476930  8554 solver.cpp:209] Iteration 417, loss = 0.409048
I1118 03:32:05.476971  8554 solver.cpp:464] Iteration 417, lr = 1e-05
I1118 03:32:07.660971  8554 solver.cpp:209] Iteration 418, loss = 0.477325
I1118 03:32:07.661011  8554 solver.cpp:464] Iteration 418, lr = 1e-05
I1118 03:32:09.864684  8554 solver.cpp:209] Iteration 419, loss = 0.679398
I1118 03:32:09.864725  8554 solver.cpp:464] Iteration 419, lr = 1e-05
I1118 03:32:12.063088  8554 solver.cpp:209] Iteration 420, loss = 0.649493
I1118 03:32:12.063117  8554 solver.cpp:464] Iteration 420, lr = 1e-05
I1118 03:32:14.269089  8554 solver.cpp:209] Iteration 421, loss = 0.686713
I1118 03:32:14.269129  8554 solver.cpp:464] Iteration 421, lr = 1e-05
I1118 03:32:16.473176  8554 solver.cpp:209] Iteration 422, loss = 0.373782
I1118 03:32:16.473204  8554 solver.cpp:464] Iteration 422, lr = 1e-05
I1118 03:32:18.704999  8554 solver.cpp:209] Iteration 423, loss = 0.421903
I1118 03:32:18.705029  8554 solver.cpp:464] Iteration 423, lr = 1e-05
I1118 03:32:20.945929  8554 solver.cpp:209] Iteration 424, loss = 0.590456
I1118 03:32:20.945968  8554 solver.cpp:464] Iteration 424, lr = 1e-05
I1118 03:32:23.175014  8554 solver.cpp:209] Iteration 425, loss = 0.585255
I1118 03:32:23.175043  8554 solver.cpp:464] Iteration 425, lr = 1e-05
I1118 03:32:25.407703  8554 solver.cpp:209] Iteration 426, loss = 0.706077
I1118 03:32:25.407733  8554 solver.cpp:464] Iteration 426, lr = 1e-05
I1118 03:32:27.634295  8554 solver.cpp:209] Iteration 427, loss = 0.922712
I1118 03:32:27.634325  8554 solver.cpp:464] Iteration 427, lr = 1e-05
I1118 03:32:29.860527  8554 solver.cpp:209] Iteration 428, loss = 0.515684
I1118 03:32:29.860611  8554 solver.cpp:464] Iteration 428, lr = 1e-05
I1118 03:32:32.090452  8554 solver.cpp:209] Iteration 429, loss = 0.567304
I1118 03:32:32.090483  8554 solver.cpp:464] Iteration 429, lr = 1e-05
I1118 03:32:34.326074  8554 solver.cpp:209] Iteration 430, loss = 0.499486
I1118 03:32:34.326102  8554 solver.cpp:464] Iteration 430, lr = 1e-05
I1118 03:32:36.563899  8554 solver.cpp:209] Iteration 431, loss = 0.844745
I1118 03:32:36.563926  8554 solver.cpp:464] Iteration 431, lr = 1e-05
I1118 03:32:38.803805  8554 solver.cpp:209] Iteration 432, loss = 0.432385
I1118 03:32:38.803845  8554 solver.cpp:464] Iteration 432, lr = 1e-05
I1118 03:32:41.029005  8554 solver.cpp:209] Iteration 433, loss = 0.410461
I1118 03:32:41.029043  8554 solver.cpp:464] Iteration 433, lr = 1e-05
I1118 03:32:43.253053  8554 solver.cpp:209] Iteration 434, loss = 0.446124
I1118 03:32:43.253093  8554 solver.cpp:464] Iteration 434, lr = 1e-05
I1118 03:32:45.472488  8554 solver.cpp:209] Iteration 435, loss = 0.499473
I1118 03:32:45.472529  8554 solver.cpp:464] Iteration 435, lr = 1e-05
I1118 03:32:47.704891  8554 solver.cpp:209] Iteration 436, loss = 0.598346
I1118 03:32:47.704921  8554 solver.cpp:464] Iteration 436, lr = 1e-05
I1118 03:32:49.949229  8554 solver.cpp:209] Iteration 437, loss = 0.499993
I1118 03:32:49.949257  8554 solver.cpp:464] Iteration 437, lr = 1e-05
I1118 03:32:52.152837  8554 solver.cpp:209] Iteration 438, loss = 0.832589
I1118 03:32:52.152865  8554 solver.cpp:464] Iteration 438, lr = 1e-05
I1118 03:32:54.358727  8554 solver.cpp:209] Iteration 439, loss = 0.926799
I1118 03:32:54.358755  8554 solver.cpp:464] Iteration 439, lr = 1e-05
I1118 03:32:56.533004  8554 solver.cpp:209] Iteration 440, loss = 0.487871
I1118 03:32:56.533043  8554 solver.cpp:464] Iteration 440, lr = 1e-05
I1118 03:32:58.710115  8554 solver.cpp:209] Iteration 441, loss = 0.501482
I1118 03:32:58.710144  8554 solver.cpp:464] Iteration 441, lr = 1e-05
I1118 03:33:00.894063  8554 solver.cpp:209] Iteration 442, loss = 0.476414
I1118 03:33:00.894152  8554 solver.cpp:464] Iteration 442, lr = 1e-05
I1118 03:33:03.072763  8554 solver.cpp:209] Iteration 443, loss = 0.890558
I1118 03:33:03.072793  8554 solver.cpp:464] Iteration 443, lr = 1e-05
I1118 03:33:05.255718  8554 solver.cpp:209] Iteration 444, loss = 0.64871
I1118 03:33:05.255749  8554 solver.cpp:464] Iteration 444, lr = 1e-05
I1118 03:33:07.428623  8554 solver.cpp:209] Iteration 445, loss = 0.890255
I1118 03:33:07.428653  8554 solver.cpp:464] Iteration 445, lr = 1e-05
I1118 03:33:09.607890  8554 solver.cpp:209] Iteration 446, loss = 0.77556
I1118 03:33:09.607931  8554 solver.cpp:464] Iteration 446, lr = 1e-05
I1118 03:33:11.790607  8554 solver.cpp:209] Iteration 447, loss = 0.588102
I1118 03:33:11.790637  8554 solver.cpp:464] Iteration 447, lr = 1e-05
I1118 03:33:13.997426  8554 solver.cpp:209] Iteration 448, loss = 0.504261
I1118 03:33:13.997468  8554 solver.cpp:464] Iteration 448, lr = 1e-05
I1118 03:33:16.219419  8554 solver.cpp:209] Iteration 449, loss = 0.418905
I1118 03:33:16.219445  8554 solver.cpp:464] Iteration 449, lr = 1e-05
I1118 03:33:16.220047  8554 solver.cpp:264] Iteration 450, Testing net (#0)
