nohup: ignoring input
I1118 02:16:01.471371 22484 caffe.cpp:99] Use GPU with device ID 0
I1118 02:16:02.456262 22484 caffe.cpp:107] Starting Optimization
I1118 02:16:02.456362 22484 solver.cpp:32] Initializing solver from parameters: 
test_iter: 72
test_interval: 50
base_lr: 1e-05
display: 1
max_iter: 20000
lr_policy: "fixed"
momentum: 0.9
weight_decay: 0.0005
snapshot: 2000
snapshot_prefix: "task/water_high_o/none/"
solver_mode: GPU
test_compute_loss: true
net: "task/water_high_o/train_val.prototxt"
I1118 02:16:02.456385 22484 solver.cpp:67] Creating training net from net file: task/water_high_o/train_val.prototxt
I1118 02:16:02.457147 22484 net.cpp:275] The NetState phase (0) differed from the phase (1) specified by a rule in layer data
I1118 02:16:02.457176 22484 net.cpp:275] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy
I1118 02:16:02.457363 22484 net.cpp:39] Initializing net from parameters: 
name: "small"
layers {
  top: "data"
  top: "label"
  name: "data"
  type: IMAGE_DATA
  image_data_param {
    source: "/data/ad6813/devCaffe/caffe/data/water_high_o/train.txt"
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
I1118 02:16:02.457479 22484 layer_factory.hpp:78] Creating layer data
I1118 02:16:02.457500 22484 net.cpp:67] Creating Layer data
I1118 02:16:02.457507 22484 net.cpp:356] data -> data
I1118 02:16:02.457526 22484 net.cpp:356] data -> label
I1118 02:16:02.457538 22484 net.cpp:96] Setting up data
I1118 02:16:02.457545 22484 image_data_layer.cpp:34] Opening file /data/ad6813/devCaffe/caffe/data/water_high_o/train.txt
I1118 02:16:02.458724 22484 image_data_layer.cpp:49] A total of 3289 images.
I1118 02:16:02.467198 22484 image_data_layer.cpp:78] output data size: 32,3,224,224
I1118 02:16:02.470052 22484 net.cpp:103] Top shape: 32 3 224 224 (4816896)
I1118 02:16:02.470077 22484 net.cpp:103] Top shape: 32 1 1 1 (32)
I1118 02:16:02.470082 22484 layer_factory.hpp:78] Creating layer conv1_1
I1118 02:16:02.470095 22484 net.cpp:67] Creating Layer conv1_1
I1118 02:16:02.470100 22484 net.cpp:394] conv1_1 <- data
I1118 02:16:02.470114 22484 net.cpp:356] conv1_1 -> conv1_1
I1118 02:16:02.470124 22484 net.cpp:96] Setting up conv1_1
I1118 02:16:02.491116 22484 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 02:16:02.491157 22484 layer_factory.hpp:78] Creating layer relu1_1
I1118 02:16:02.491171 22484 net.cpp:67] Creating Layer relu1_1
I1118 02:16:02.491178 22484 net.cpp:394] relu1_1 <- conv1_1
I1118 02:16:02.491185 22484 net.cpp:345] relu1_1 -> conv1_1 (in-place)
I1118 02:16:02.491195 22484 net.cpp:96] Setting up relu1_1
I1118 02:16:02.491211 22484 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 02:16:02.491216 22484 layer_factory.hpp:78] Creating layer conv1_2
I1118 02:16:02.491224 22484 net.cpp:67] Creating Layer conv1_2
I1118 02:16:02.491227 22484 net.cpp:394] conv1_2 <- conv1_1
I1118 02:16:02.491235 22484 net.cpp:356] conv1_2 -> conv1_2
I1118 02:16:02.491242 22484 net.cpp:96] Setting up conv1_2
I1118 02:16:02.492596 22484 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 02:16:02.492612 22484 layer_factory.hpp:78] Creating layer relu1_2
I1118 02:16:02.492620 22484 net.cpp:67] Creating Layer relu1_2
I1118 02:16:02.492624 22484 net.cpp:394] relu1_2 <- conv1_2
I1118 02:16:02.492630 22484 net.cpp:345] relu1_2 -> conv1_2 (in-place)
I1118 02:16:02.492636 22484 net.cpp:96] Setting up relu1_2
I1118 02:16:02.492642 22484 net.cpp:103] Top shape: 32 64 224 224 (102760448)
I1118 02:16:02.492647 22484 layer_factory.hpp:78] Creating layer pool1
I1118 02:16:02.492657 22484 net.cpp:67] Creating Layer pool1
I1118 02:16:02.492661 22484 net.cpp:394] pool1 <- conv1_2
I1118 02:16:02.492667 22484 net.cpp:356] pool1 -> pool1
I1118 02:16:02.492674 22484 net.cpp:96] Setting up pool1
I1118 02:16:02.492694 22484 net.cpp:103] Top shape: 32 64 112 112 (25690112)
I1118 02:16:02.492702 22484 layer_factory.hpp:78] Creating layer conv2_1
I1118 02:16:02.492710 22484 net.cpp:67] Creating Layer conv2_1
I1118 02:16:02.492714 22484 net.cpp:394] conv2_1 <- pool1
I1118 02:16:02.492720 22484 net.cpp:356] conv2_1 -> conv2_1
I1118 02:16:02.492727 22484 net.cpp:96] Setting up conv2_1
I1118 02:16:02.495231 22484 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 02:16:02.495249 22484 layer_factory.hpp:78] Creating layer relu2_1
I1118 02:16:02.495256 22484 net.cpp:67] Creating Layer relu2_1
I1118 02:16:02.495260 22484 net.cpp:394] relu2_1 <- conv2_1
I1118 02:16:02.495266 22484 net.cpp:345] relu2_1 -> conv2_1 (in-place)
I1118 02:16:02.495272 22484 net.cpp:96] Setting up relu2_1
I1118 02:16:02.495278 22484 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 02:16:02.495283 22484 layer_factory.hpp:78] Creating layer conv2_2
I1118 02:16:02.495292 22484 net.cpp:67] Creating Layer conv2_2
I1118 02:16:02.495296 22484 net.cpp:394] conv2_2 <- conv2_1
I1118 02:16:02.495302 22484 net.cpp:356] conv2_2 -> conv2_2
I1118 02:16:02.495309 22484 net.cpp:96] Setting up conv2_2
I1118 02:16:02.499851 22484 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 02:16:02.499863 22484 layer_factory.hpp:78] Creating layer relu2_2
I1118 02:16:02.499868 22484 net.cpp:67] Creating Layer relu2_2
I1118 02:16:02.499871 22484 net.cpp:394] relu2_2 <- conv2_2
I1118 02:16:02.499876 22484 net.cpp:345] relu2_2 -> conv2_2 (in-place)
I1118 02:16:02.499881 22484 net.cpp:96] Setting up relu2_2
I1118 02:16:02.499886 22484 net.cpp:103] Top shape: 32 128 112 112 (51380224)
I1118 02:16:02.499897 22484 layer_factory.hpp:78] Creating layer pool2
I1118 02:16:02.499904 22484 net.cpp:67] Creating Layer pool2
I1118 02:16:02.499907 22484 net.cpp:394] pool2 <- conv2_2
I1118 02:16:02.499912 22484 net.cpp:356] pool2 -> pool2
I1118 02:16:02.499917 22484 net.cpp:96] Setting up pool2
I1118 02:16:02.499922 22484 net.cpp:103] Top shape: 32 128 56 56 (12845056)
I1118 02:16:02.499925 22484 layer_factory.hpp:78] Creating layer conv3_1
I1118 02:16:02.499933 22484 net.cpp:67] Creating Layer conv3_1
I1118 02:16:02.499935 22484 net.cpp:394] conv3_1 <- pool2
I1118 02:16:02.499939 22484 net.cpp:356] conv3_1 -> conv3_1
I1118 02:16:02.499945 22484 net.cpp:96] Setting up conv3_1
I1118 02:16:02.507282 22484 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 02:16:02.507297 22484 layer_factory.hpp:78] Creating layer relu3_1
I1118 02:16:02.507302 22484 net.cpp:67] Creating Layer relu3_1
I1118 02:16:02.507304 22484 net.cpp:394] relu3_1 <- conv3_1
I1118 02:16:02.507308 22484 net.cpp:345] relu3_1 -> conv3_1 (in-place)
I1118 02:16:02.507313 22484 net.cpp:96] Setting up relu3_1
I1118 02:16:02.507318 22484 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 02:16:02.507321 22484 layer_factory.hpp:78] Creating layer conv3_2
I1118 02:16:02.507328 22484 net.cpp:67] Creating Layer conv3_2
I1118 02:16:02.507330 22484 net.cpp:394] conv3_2 <- conv3_1
I1118 02:16:02.507334 22484 net.cpp:356] conv3_2 -> conv3_2
I1118 02:16:02.507339 22484 net.cpp:96] Setting up conv3_2
I1118 02:16:02.522128 22484 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 02:16:02.522151 22484 layer_factory.hpp:78] Creating layer relu3_2
I1118 02:16:02.522161 22484 net.cpp:67] Creating Layer relu3_2
I1118 02:16:02.522166 22484 net.cpp:394] relu3_2 <- conv3_2
I1118 02:16:02.522171 22484 net.cpp:345] relu3_2 -> conv3_2 (in-place)
I1118 02:16:02.522177 22484 net.cpp:96] Setting up relu3_2
I1118 02:16:02.522183 22484 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 02:16:02.522186 22484 layer_factory.hpp:78] Creating layer conv3_3
I1118 02:16:02.522193 22484 net.cpp:67] Creating Layer conv3_3
I1118 02:16:02.522197 22484 net.cpp:394] conv3_3 <- conv3_2
I1118 02:16:02.522202 22484 net.cpp:356] conv3_3 -> conv3_3
I1118 02:16:02.522207 22484 net.cpp:96] Setting up conv3_3
I1118 02:16:02.536882 22484 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 02:16:02.536902 22484 layer_factory.hpp:78] Creating layer relu3_3
I1118 02:16:02.536914 22484 net.cpp:67] Creating Layer relu3_3
I1118 02:16:02.536918 22484 net.cpp:394] relu3_3 <- conv3_3
I1118 02:16:02.536924 22484 net.cpp:345] relu3_3 -> conv3_3 (in-place)
I1118 02:16:02.536931 22484 net.cpp:96] Setting up relu3_3
I1118 02:16:02.536936 22484 net.cpp:103] Top shape: 32 256 56 56 (25690112)
I1118 02:16:02.536939 22484 layer_factory.hpp:78] Creating layer pool3
I1118 02:16:02.536945 22484 net.cpp:67] Creating Layer pool3
I1118 02:16:02.536948 22484 net.cpp:394] pool3 <- conv3_3
I1118 02:16:02.536952 22484 net.cpp:356] pool3 -> pool3
I1118 02:16:02.536957 22484 net.cpp:96] Setting up pool3
I1118 02:16:02.536964 22484 net.cpp:103] Top shape: 32 256 28 28 (6422528)
I1118 02:16:02.536967 22484 layer_factory.hpp:78] Creating layer conv4_1
I1118 02:16:02.536974 22484 net.cpp:67] Creating Layer conv4_1
I1118 02:16:02.536978 22484 net.cpp:394] conv4_1 <- pool3
I1118 02:16:02.536981 22484 net.cpp:356] conv4_1 -> conv4_1
I1118 02:16:02.536988 22484 net.cpp:96] Setting up conv4_1
I1118 02:16:02.565942 22484 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 02:16:02.565965 22484 layer_factory.hpp:78] Creating layer relu4_1
I1118 02:16:02.565974 22484 net.cpp:67] Creating Layer relu4_1
I1118 02:16:02.565979 22484 net.cpp:394] relu4_1 <- conv4_1
I1118 02:16:02.565986 22484 net.cpp:345] relu4_1 -> conv4_1 (in-place)
I1118 02:16:02.565994 22484 net.cpp:96] Setting up relu4_1
I1118 02:16:02.565999 22484 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 02:16:02.566002 22484 layer_factory.hpp:78] Creating layer conv4_2
I1118 02:16:02.566009 22484 net.cpp:67] Creating Layer conv4_2
I1118 02:16:02.566011 22484 net.cpp:394] conv4_2 <- conv4_1
I1118 02:16:02.566028 22484 net.cpp:356] conv4_2 -> conv4_2
I1118 02:16:02.566035 22484 net.cpp:96] Setting up conv4_2
I1118 02:16:02.623450 22484 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 02:16:02.623477 22484 layer_factory.hpp:78] Creating layer relu4_2
I1118 02:16:02.623486 22484 net.cpp:67] Creating Layer relu4_2
I1118 02:16:02.623492 22484 net.cpp:394] relu4_2 <- conv4_2
I1118 02:16:02.623499 22484 net.cpp:345] relu4_2 -> conv4_2 (in-place)
I1118 02:16:02.623507 22484 net.cpp:96] Setting up relu4_2
I1118 02:16:02.623512 22484 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 02:16:02.623514 22484 layer_factory.hpp:78] Creating layer conv4_3
I1118 02:16:02.623520 22484 net.cpp:67] Creating Layer conv4_3
I1118 02:16:02.623523 22484 net.cpp:394] conv4_3 <- conv4_2
I1118 02:16:02.623528 22484 net.cpp:356] conv4_3 -> conv4_3
I1118 02:16:02.623533 22484 net.cpp:96] Setting up conv4_3
I1118 02:16:02.681185 22484 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 02:16:02.681208 22484 layer_factory.hpp:78] Creating layer relu4_3
I1118 02:16:02.681217 22484 net.cpp:67] Creating Layer relu4_3
I1118 02:16:02.681222 22484 net.cpp:394] relu4_3 <- conv4_3
I1118 02:16:02.681229 22484 net.cpp:345] relu4_3 -> conv4_3 (in-place)
I1118 02:16:02.681236 22484 net.cpp:96] Setting up relu4_3
I1118 02:16:02.681241 22484 net.cpp:103] Top shape: 32 512 28 28 (12845056)
I1118 02:16:02.681244 22484 layer_factory.hpp:78] Creating layer pool4
I1118 02:16:02.681249 22484 net.cpp:67] Creating Layer pool4
I1118 02:16:02.681252 22484 net.cpp:394] pool4 <- conv4_3
I1118 02:16:02.681257 22484 net.cpp:356] pool4 -> pool4
I1118 02:16:02.681262 22484 net.cpp:96] Setting up pool4
I1118 02:16:02.681270 22484 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 02:16:02.681274 22484 layer_factory.hpp:78] Creating layer conv5_1
I1118 02:16:02.681280 22484 net.cpp:67] Creating Layer conv5_1
I1118 02:16:02.681283 22484 net.cpp:394] conv5_1 <- pool4
I1118 02:16:02.681290 22484 net.cpp:356] conv5_1 -> conv5_1
I1118 02:16:02.681298 22484 net.cpp:96] Setting up conv5_1
I1118 02:16:02.738911 22484 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 02:16:02.738935 22484 layer_factory.hpp:78] Creating layer relu5_1
I1118 02:16:02.738944 22484 net.cpp:67] Creating Layer relu5_1
I1118 02:16:02.738947 22484 net.cpp:394] relu5_1 <- conv5_1
I1118 02:16:02.738960 22484 net.cpp:345] relu5_1 -> conv5_1 (in-place)
I1118 02:16:02.738967 22484 net.cpp:96] Setting up relu5_1
I1118 02:16:02.738972 22484 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 02:16:02.738976 22484 layer_factory.hpp:78] Creating layer conv5_2
I1118 02:16:02.738981 22484 net.cpp:67] Creating Layer conv5_2
I1118 02:16:02.738986 22484 net.cpp:394] conv5_2 <- conv5_1
I1118 02:16:02.738991 22484 net.cpp:356] conv5_2 -> conv5_2
I1118 02:16:02.738996 22484 net.cpp:96] Setting up conv5_2
I1118 02:16:02.796268 22484 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 02:16:02.796298 22484 layer_factory.hpp:78] Creating layer relu5_2
I1118 02:16:02.796306 22484 net.cpp:67] Creating Layer relu5_2
I1118 02:16:02.796311 22484 net.cpp:394] relu5_2 <- conv5_2
I1118 02:16:02.796319 22484 net.cpp:345] relu5_2 -> conv5_2 (in-place)
I1118 02:16:02.796325 22484 net.cpp:96] Setting up relu5_2
I1118 02:16:02.796330 22484 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 02:16:02.796334 22484 layer_factory.hpp:78] Creating layer conv5_3
I1118 02:16:02.796339 22484 net.cpp:67] Creating Layer conv5_3
I1118 02:16:02.796342 22484 net.cpp:394] conv5_3 <- conv5_2
I1118 02:16:02.796349 22484 net.cpp:356] conv5_3 -> conv5_3
I1118 02:16:02.796355 22484 net.cpp:96] Setting up conv5_3
I1118 02:16:02.853868 22484 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 02:16:02.853894 22484 layer_factory.hpp:78] Creating layer relu5_3
I1118 02:16:02.853901 22484 net.cpp:67] Creating Layer relu5_3
I1118 02:16:02.853906 22484 net.cpp:394] relu5_3 <- conv5_3
I1118 02:16:02.853915 22484 net.cpp:345] relu5_3 -> conv5_3 (in-place)
I1118 02:16:02.853921 22484 net.cpp:96] Setting up relu5_3
I1118 02:16:02.853927 22484 net.cpp:103] Top shape: 32 512 14 14 (3211264)
I1118 02:16:02.853940 22484 layer_factory.hpp:78] Creating layer pool5
I1118 02:16:02.853945 22484 net.cpp:67] Creating Layer pool5
I1118 02:16:02.853948 22484 net.cpp:394] pool5 <- conv5_3
I1118 02:16:02.853955 22484 net.cpp:356] pool5 -> pool5
I1118 02:16:02.853960 22484 net.cpp:96] Setting up pool5
I1118 02:16:02.853967 22484 net.cpp:103] Top shape: 32 512 7 7 (802816)
I1118 02:16:02.853971 22484 layer_factory.hpp:78] Creating layer fc6
I1118 02:16:02.853984 22484 net.cpp:67] Creating Layer fc6
I1118 02:16:02.853987 22484 net.cpp:394] fc6 <- pool5
I1118 02:16:02.853992 22484 net.cpp:356] fc6 -> fc6
I1118 02:16:02.853998 22484 net.cpp:96] Setting up fc6
I1118 02:16:05.235056 22484 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 02:16:05.235100 22484 layer_factory.hpp:78] Creating layer relu6
I1118 02:16:05.235108 22484 net.cpp:67] Creating Layer relu6
I1118 02:16:05.235113 22484 net.cpp:394] relu6 <- fc6
I1118 02:16:05.235119 22484 net.cpp:345] relu6 -> fc6 (in-place)
I1118 02:16:05.235126 22484 net.cpp:96] Setting up relu6
I1118 02:16:05.235141 22484 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 02:16:05.235143 22484 layer_factory.hpp:78] Creating layer drop6
I1118 02:16:05.235152 22484 net.cpp:67] Creating Layer drop6
I1118 02:16:05.235154 22484 net.cpp:394] drop6 <- fc6
I1118 02:16:05.235159 22484 net.cpp:345] drop6 -> fc6 (in-place)
I1118 02:16:05.235164 22484 net.cpp:96] Setting up drop6
I1118 02:16:05.235170 22484 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 02:16:05.235173 22484 layer_factory.hpp:78] Creating layer fc7
I1118 02:16:05.235178 22484 net.cpp:67] Creating Layer fc7
I1118 02:16:05.235182 22484 net.cpp:394] fc7 <- fc6
I1118 02:16:05.235187 22484 net.cpp:356] fc7 -> fc7
I1118 02:16:05.235193 22484 net.cpp:96] Setting up fc7
I1118 02:16:05.622151 22484 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 02:16:05.622194 22484 layer_factory.hpp:78] Creating layer relu7
I1118 02:16:05.622201 22484 net.cpp:67] Creating Layer relu7
I1118 02:16:05.622206 22484 net.cpp:394] relu7 <- fc7
I1118 02:16:05.622213 22484 net.cpp:345] relu7 -> fc7 (in-place)
I1118 02:16:05.622220 22484 net.cpp:96] Setting up relu7
I1118 02:16:05.622234 22484 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 02:16:05.622237 22484 layer_factory.hpp:78] Creating layer drop7
I1118 02:16:05.622244 22484 net.cpp:67] Creating Layer drop7
I1118 02:16:05.622247 22484 net.cpp:394] drop7 <- fc7
I1118 02:16:05.622251 22484 net.cpp:345] drop7 -> fc7 (in-place)
I1118 02:16:05.622256 22484 net.cpp:96] Setting up drop7
I1118 02:16:05.622259 22484 net.cpp:103] Top shape: 32 4096 1 1 (131072)
I1118 02:16:05.622262 22484 layer_factory.hpp:78] Creating layer fc8_2
I1118 02:16:05.622268 22484 net.cpp:67] Creating Layer fc8_2
I1118 02:16:05.622272 22484 net.cpp:394] fc8_2 <- fc7
I1118 02:16:05.622277 22484 net.cpp:356] fc8_2 -> fc8_2
I1118 02:16:05.622282 22484 net.cpp:96] Setting up fc8_2
I1118 02:16:05.622493 22484 net.cpp:103] Top shape: 32 2 1 1 (64)
I1118 02:16:05.622503 22484 layer_factory.hpp:78] Creating layer loss
I1118 02:16:05.622511 22484 net.cpp:67] Creating Layer loss
I1118 02:16:05.622516 22484 net.cpp:394] loss <- fc8_2
I1118 02:16:05.622520 22484 net.cpp:394] loss <- label
I1118 02:16:05.622527 22484 net.cpp:356] loss -> (automatic)
I1118 02:16:05.622534 22484 net.cpp:96] Setting up loss
I1118 02:16:05.622545 22484 net.cpp:103] Top shape: 1 1 1 1 (1)
I1118 02:16:05.622548 22484 net.cpp:109]     with loss weight 1
I1118 02:16:05.622581 22484 net.cpp:170] loss needs backward computation.
I1118 02:16:05.622586 22484 net.cpp:170] fc8_2 needs backward computation.
I1118 02:16:05.622589 22484 net.cpp:170] drop7 needs backward computation.
I1118 02:16:05.622591 22484 net.cpp:170] relu7 needs backward computation.
I1118 02:16:05.622594 22484 net.cpp:170] fc7 needs backward computation.
I1118 02:16:05.622596 22484 net.cpp:170] drop6 needs backward computation.
I1118 02:16:05.622599 22484 net.cpp:170] relu6 needs backward computation.
I1118 02:16:05.622602 22484 net.cpp:170] fc6 needs backward computation.
I1118 02:16:05.622613 22484 net.cpp:170] pool5 needs backward computation.
I1118 02:16:05.622617 22484 net.cpp:170] relu5_3 needs backward computation.
I1118 02:16:05.622619 22484 net.cpp:170] conv5_3 needs backward computation.
I1118 02:16:05.622622 22484 net.cpp:170] relu5_2 needs backward computation.
I1118 02:16:05.622627 22484 net.cpp:170] conv5_2 needs backward computation.
I1118 02:16:05.622628 22484 net.cpp:170] relu5_1 needs backward computation.
I1118 02:16:05.622632 22484 net.cpp:170] conv5_1 needs backward computation.
I1118 02:16:05.622634 22484 net.cpp:170] pool4 needs backward computation.
I1118 02:16:05.622637 22484 net.cpp:170] relu4_3 needs backward computation.
I1118 02:16:05.622640 22484 net.cpp:170] conv4_3 needs backward computation.
I1118 02:16:05.622643 22484 net.cpp:170] relu4_2 needs backward computation.
I1118 02:16:05.622645 22484 net.cpp:170] conv4_2 needs backward computation.
I1118 02:16:05.622648 22484 net.cpp:170] relu4_1 needs backward computation.
I1118 02:16:05.622652 22484 net.cpp:170] conv4_1 needs backward computation.
I1118 02:16:05.622654 22484 net.cpp:170] pool3 needs backward computation.
I1118 02:16:05.622658 22484 net.cpp:170] relu3_3 needs backward computation.
I1118 02:16:05.622659 22484 net.cpp:170] conv3_3 needs backward computation.
I1118 02:16:05.622663 22484 net.cpp:170] relu3_2 needs backward computation.
I1118 02:16:05.622665 22484 net.cpp:170] conv3_2 needs backward computation.
I1118 02:16:05.622668 22484 net.cpp:170] relu3_1 needs backward computation.
I1118 02:16:05.622670 22484 net.cpp:170] conv3_1 needs backward computation.
I1118 02:16:05.622673 22484 net.cpp:170] pool2 needs backward computation.
I1118 02:16:05.622676 22484 net.cpp:170] relu2_2 needs backward computation.
I1118 02:16:05.622678 22484 net.cpp:170] conv2_2 needs backward computation.
I1118 02:16:05.622681 22484 net.cpp:170] relu2_1 needs backward computation.
I1118 02:16:05.622684 22484 net.cpp:170] conv2_1 needs backward computation.
I1118 02:16:05.622688 22484 net.cpp:170] pool1 needs backward computation.
I1118 02:16:05.622689 22484 net.cpp:170] relu1_2 needs backward computation.
I1118 02:16:05.622692 22484 net.cpp:170] conv1_2 needs backward computation.
I1118 02:16:05.622695 22484 net.cpp:170] relu1_1 needs backward computation.
I1118 02:16:05.622697 22484 net.cpp:170] conv1_1 needs backward computation.
I1118 02:16:05.622700 22484 net.cpp:172] data does not need backward computation.
I1118 02:16:05.622719 22484 net.cpp:467] Collecting Learning Rate and Weight Decay.
I1118 02:16:05.622725 22484 net.cpp:219] Network initialization done.
I1118 02:16:05.622730 22484 net.cpp:220] Memory required for data: 3686465924
I1118 02:16:05.623504 22484 solver.cpp:151] Creating test net (#0) specified by net file: task/water_high_o/train_val.prototxt
I1118 02:16:05.623550 22484 net.cpp:275] The NetState phase (1) differed from the phase (0) specified by a rule in layer data
I1118 02:16:05.623754 22484 net.cpp:39] Initializing net from parameters: 
name: "small"
layers {
  top: "data"
  top: "label"
  name: "data"
  type: IMAGE_DATA
  image_data_param {
    source: "/data/ad6813/devCaffe/caffe/data/water_high_o/val.txt"
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
I1118 02:16:05.623884 22484 layer_factory.hpp:78] Creating layer data
I1118 02:16:05.623893 22484 net.cpp:67] Creating Layer data
I1118 02:16:05.623898 22484 net.cpp:356] data -> data
I1118 02:16:05.623905 22484 net.cpp:356] data -> label
I1118 02:16:05.623910 22484 net.cpp:96] Setting up data
I1118 02:16:05.623914 22484 image_data_layer.cpp:34] Opening file /data/ad6813/devCaffe/caffe/data/water_high_o/val.txt
I1118 02:16:05.624141 22484 image_data_layer.cpp:49] A total of 576 images.
I1118 02:16:05.631034 22484 image_data_layer.cpp:78] output data size: 8,3,224,224
I1118 02:16:05.631896 22484 net.cpp:103] Top shape: 8 3 224 224 (1204224)
I1118 02:16:05.631906 22484 net.cpp:103] Top shape: 8 1 1 1 (8)
I1118 02:16:05.631911 22484 layer_factory.hpp:78] Creating layer label_data_1_split
I1118 02:16:05.631921 22484 net.cpp:67] Creating Layer label_data_1_split
I1118 02:16:05.631924 22484 net.cpp:394] label_data_1_split <- label
I1118 02:16:05.631932 22484 net.cpp:356] label_data_1_split -> label_data_1_split_0
I1118 02:16:05.631939 22484 net.cpp:356] label_data_1_split -> label_data_1_split_1
I1118 02:16:05.631945 22484 net.cpp:96] Setting up label_data_1_split
I1118 02:16:05.631950 22484 net.cpp:103] Top shape: 8 1 1 1 (8)
I1118 02:16:05.631953 22484 net.cpp:103] Top shape: 8 1 1 1 (8)
I1118 02:16:05.631957 22484 layer_factory.hpp:78] Creating layer conv1_1
I1118 02:16:05.631963 22484 net.cpp:67] Creating Layer conv1_1
I1118 02:16:05.631966 22484 net.cpp:394] conv1_1 <- data
I1118 02:16:05.631971 22484 net.cpp:356] conv1_1 -> conv1_1
I1118 02:16:05.631978 22484 net.cpp:96] Setting up conv1_1
I1118 02:16:05.632133 22484 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 02:16:05.632146 22484 layer_factory.hpp:78] Creating layer relu1_1
I1118 02:16:05.632154 22484 net.cpp:67] Creating Layer relu1_1
I1118 02:16:05.632158 22484 net.cpp:394] relu1_1 <- conv1_1
I1118 02:16:05.632163 22484 net.cpp:345] relu1_1 -> conv1_1 (in-place)
I1118 02:16:05.632174 22484 net.cpp:96] Setting up relu1_1
I1118 02:16:05.632179 22484 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 02:16:05.632182 22484 layer_factory.hpp:78] Creating layer conv1_2
I1118 02:16:05.632187 22484 net.cpp:67] Creating Layer conv1_2
I1118 02:16:05.632190 22484 net.cpp:394] conv1_2 <- conv1_1
I1118 02:16:05.632195 22484 net.cpp:356] conv1_2 -> conv1_2
I1118 02:16:05.632200 22484 net.cpp:96] Setting up conv1_2
I1118 02:16:05.633165 22484 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 02:16:05.633177 22484 layer_factory.hpp:78] Creating layer relu1_2
I1118 02:16:05.633183 22484 net.cpp:67] Creating Layer relu1_2
I1118 02:16:05.633185 22484 net.cpp:394] relu1_2 <- conv1_2
I1118 02:16:05.633189 22484 net.cpp:345] relu1_2 -> conv1_2 (in-place)
I1118 02:16:05.633194 22484 net.cpp:96] Setting up relu1_2
I1118 02:16:05.633199 22484 net.cpp:103] Top shape: 8 64 224 224 (25690112)
I1118 02:16:05.633203 22484 layer_factory.hpp:78] Creating layer pool1
I1118 02:16:05.633208 22484 net.cpp:67] Creating Layer pool1
I1118 02:16:05.633210 22484 net.cpp:394] pool1 <- conv1_2
I1118 02:16:05.633214 22484 net.cpp:356] pool1 -> pool1
I1118 02:16:05.633219 22484 net.cpp:96] Setting up pool1
I1118 02:16:05.633226 22484 net.cpp:103] Top shape: 8 64 112 112 (6422528)
I1118 02:16:05.633229 22484 layer_factory.hpp:78] Creating layer conv2_1
I1118 02:16:05.633234 22484 net.cpp:67] Creating Layer conv2_1
I1118 02:16:05.633236 22484 net.cpp:394] conv2_1 <- pool1
I1118 02:16:05.633241 22484 net.cpp:356] conv2_1 -> conv2_1
I1118 02:16:05.633246 22484 net.cpp:96] Setting up conv2_1
I1118 02:16:05.635227 22484 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 02:16:05.635241 22484 layer_factory.hpp:78] Creating layer relu2_1
I1118 02:16:05.635246 22484 net.cpp:67] Creating Layer relu2_1
I1118 02:16:05.635251 22484 net.cpp:394] relu2_1 <- conv2_1
I1118 02:16:05.635254 22484 net.cpp:345] relu2_1 -> conv2_1 (in-place)
I1118 02:16:05.635259 22484 net.cpp:96] Setting up relu2_1
I1118 02:16:05.635264 22484 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 02:16:05.635267 22484 layer_factory.hpp:78] Creating layer conv2_2
I1118 02:16:05.635272 22484 net.cpp:67] Creating Layer conv2_2
I1118 02:16:05.635275 22484 net.cpp:394] conv2_2 <- conv2_1
I1118 02:16:05.635279 22484 net.cpp:356] conv2_2 -> conv2_2
I1118 02:16:05.635285 22484 net.cpp:96] Setting up conv2_2
I1118 02:16:05.638902 22484 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 02:16:05.638912 22484 layer_factory.hpp:78] Creating layer relu2_2
I1118 02:16:05.638917 22484 net.cpp:67] Creating Layer relu2_2
I1118 02:16:05.638921 22484 net.cpp:394] relu2_2 <- conv2_2
I1118 02:16:05.638924 22484 net.cpp:345] relu2_2 -> conv2_2 (in-place)
I1118 02:16:05.638929 22484 net.cpp:96] Setting up relu2_2
I1118 02:16:05.638933 22484 net.cpp:103] Top shape: 8 128 112 112 (12845056)
I1118 02:16:05.638937 22484 layer_factory.hpp:78] Creating layer pool2
I1118 02:16:05.638942 22484 net.cpp:67] Creating Layer pool2
I1118 02:16:05.638944 22484 net.cpp:394] pool2 <- conv2_2
I1118 02:16:05.638948 22484 net.cpp:356] pool2 -> pool2
I1118 02:16:05.638952 22484 net.cpp:96] Setting up pool2
I1118 02:16:05.638958 22484 net.cpp:103] Top shape: 8 128 56 56 (3211264)
I1118 02:16:05.638962 22484 layer_factory.hpp:78] Creating layer conv3_1
I1118 02:16:05.638967 22484 net.cpp:67] Creating Layer conv3_1
I1118 02:16:05.638969 22484 net.cpp:394] conv3_1 <- pool2
I1118 02:16:05.638973 22484 net.cpp:356] conv3_1 -> conv3_1
I1118 02:16:05.638978 22484 net.cpp:96] Setting up conv3_1
I1118 02:16:05.646155 22484 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 02:16:05.646168 22484 layer_factory.hpp:78] Creating layer relu3_1
I1118 02:16:05.646173 22484 net.cpp:67] Creating Layer relu3_1
I1118 02:16:05.646177 22484 net.cpp:394] relu3_1 <- conv3_1
I1118 02:16:05.646181 22484 net.cpp:345] relu3_1 -> conv3_1 (in-place)
I1118 02:16:05.646185 22484 net.cpp:96] Setting up relu3_1
I1118 02:16:05.646190 22484 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 02:16:05.646193 22484 layer_factory.hpp:78] Creating layer conv3_2
I1118 02:16:05.646204 22484 net.cpp:67] Creating Layer conv3_2
I1118 02:16:05.646208 22484 net.cpp:394] conv3_2 <- conv3_1
I1118 02:16:05.646213 22484 net.cpp:356] conv3_2 -> conv3_2
I1118 02:16:05.646217 22484 net.cpp:96] Setting up conv3_2
I1118 02:16:05.660751 22484 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 02:16:05.660773 22484 layer_factory.hpp:78] Creating layer relu3_2
I1118 02:16:05.660781 22484 net.cpp:67] Creating Layer relu3_2
I1118 02:16:05.660785 22484 net.cpp:394] relu3_2 <- conv3_2
I1118 02:16:05.660792 22484 net.cpp:345] relu3_2 -> conv3_2 (in-place)
I1118 02:16:05.660799 22484 net.cpp:96] Setting up relu3_2
I1118 02:16:05.660804 22484 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 02:16:05.660807 22484 layer_factory.hpp:78] Creating layer conv3_3
I1118 02:16:05.660815 22484 net.cpp:67] Creating Layer conv3_3
I1118 02:16:05.660819 22484 net.cpp:394] conv3_3 <- conv3_2
I1118 02:16:05.660825 22484 net.cpp:356] conv3_3 -> conv3_3
I1118 02:16:05.660830 22484 net.cpp:96] Setting up conv3_3
I1118 02:16:05.675462 22484 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 02:16:05.675478 22484 layer_factory.hpp:78] Creating layer relu3_3
I1118 02:16:05.675485 22484 net.cpp:67] Creating Layer relu3_3
I1118 02:16:05.675489 22484 net.cpp:394] relu3_3 <- conv3_3
I1118 02:16:05.675494 22484 net.cpp:345] relu3_3 -> conv3_3 (in-place)
I1118 02:16:05.675500 22484 net.cpp:96] Setting up relu3_3
I1118 02:16:05.675504 22484 net.cpp:103] Top shape: 8 256 56 56 (6422528)
I1118 02:16:05.675508 22484 layer_factory.hpp:78] Creating layer pool3
I1118 02:16:05.675513 22484 net.cpp:67] Creating Layer pool3
I1118 02:16:05.675516 22484 net.cpp:394] pool3 <- conv3_3
I1118 02:16:05.675523 22484 net.cpp:356] pool3 -> pool3
I1118 02:16:05.675528 22484 net.cpp:96] Setting up pool3
I1118 02:16:05.675534 22484 net.cpp:103] Top shape: 8 256 28 28 (1605632)
I1118 02:16:05.675537 22484 layer_factory.hpp:78] Creating layer conv4_1
I1118 02:16:05.675544 22484 net.cpp:67] Creating Layer conv4_1
I1118 02:16:05.675545 22484 net.cpp:394] conv4_1 <- pool3
I1118 02:16:05.675551 22484 net.cpp:356] conv4_1 -> conv4_1
I1118 02:16:05.675556 22484 net.cpp:96] Setting up conv4_1
I1118 02:16:05.703696 22484 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 02:16:05.703732 22484 layer_factory.hpp:78] Creating layer relu4_1
I1118 02:16:05.703740 22484 net.cpp:67] Creating Layer relu4_1
I1118 02:16:05.703745 22484 net.cpp:394] relu4_1 <- conv4_1
I1118 02:16:05.703752 22484 net.cpp:345] relu4_1 -> conv4_1 (in-place)
I1118 02:16:05.703758 22484 net.cpp:96] Setting up relu4_1
I1118 02:16:05.703764 22484 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 02:16:05.703768 22484 layer_factory.hpp:78] Creating layer conv4_2
I1118 02:16:05.703774 22484 net.cpp:67] Creating Layer conv4_2
I1118 02:16:05.703778 22484 net.cpp:394] conv4_2 <- conv4_1
I1118 02:16:05.703783 22484 net.cpp:356] conv4_2 -> conv4_2
I1118 02:16:05.703788 22484 net.cpp:96] Setting up conv4_2
I1118 02:16:05.758550 22484 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 02:16:05.758594 22484 layer_factory.hpp:78] Creating layer relu4_2
I1118 02:16:05.758601 22484 net.cpp:67] Creating Layer relu4_2
I1118 02:16:05.758605 22484 net.cpp:394] relu4_2 <- conv4_2
I1118 02:16:05.758612 22484 net.cpp:345] relu4_2 -> conv4_2 (in-place)
I1118 02:16:05.758620 22484 net.cpp:96] Setting up relu4_2
I1118 02:16:05.758625 22484 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 02:16:05.758627 22484 layer_factory.hpp:78] Creating layer conv4_3
I1118 02:16:05.758633 22484 net.cpp:67] Creating Layer conv4_3
I1118 02:16:05.758636 22484 net.cpp:394] conv4_3 <- conv4_2
I1118 02:16:05.758642 22484 net.cpp:356] conv4_3 -> conv4_3
I1118 02:16:05.758649 22484 net.cpp:96] Setting up conv4_3
I1118 02:16:05.813283 22484 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 02:16:05.813323 22484 layer_factory.hpp:78] Creating layer relu4_3
I1118 02:16:05.813331 22484 net.cpp:67] Creating Layer relu4_3
I1118 02:16:05.813336 22484 net.cpp:394] relu4_3 <- conv4_3
I1118 02:16:05.813352 22484 net.cpp:345] relu4_3 -> conv4_3 (in-place)
I1118 02:16:05.813359 22484 net.cpp:96] Setting up relu4_3
I1118 02:16:05.813364 22484 net.cpp:103] Top shape: 8 512 28 28 (3211264)
I1118 02:16:05.813369 22484 layer_factory.hpp:78] Creating layer pool4
I1118 02:16:05.813374 22484 net.cpp:67] Creating Layer pool4
I1118 02:16:05.813376 22484 net.cpp:394] pool4 <- conv4_3
I1118 02:16:05.813381 22484 net.cpp:356] pool4 -> pool4
I1118 02:16:05.813386 22484 net.cpp:96] Setting up pool4
I1118 02:16:05.813393 22484 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 02:16:05.813397 22484 layer_factory.hpp:78] Creating layer conv5_1
I1118 02:16:05.813403 22484 net.cpp:67] Creating Layer conv5_1
I1118 02:16:05.813406 22484 net.cpp:394] conv5_1 <- pool4
I1118 02:16:05.813411 22484 net.cpp:356] conv5_1 -> conv5_1
I1118 02:16:05.813416 22484 net.cpp:96] Setting up conv5_1
I1118 02:16:05.868199 22484 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 02:16:05.868234 22484 layer_factory.hpp:78] Creating layer relu5_1
I1118 02:16:05.868243 22484 net.cpp:67] Creating Layer relu5_1
I1118 02:16:05.868248 22484 net.cpp:394] relu5_1 <- conv5_1
I1118 02:16:05.868254 22484 net.cpp:345] relu5_1 -> conv5_1 (in-place)
I1118 02:16:05.868260 22484 net.cpp:96] Setting up relu5_1
I1118 02:16:05.868265 22484 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 02:16:05.868269 22484 layer_factory.hpp:78] Creating layer conv5_2
I1118 02:16:05.868274 22484 net.cpp:67] Creating Layer conv5_2
I1118 02:16:05.868278 22484 net.cpp:394] conv5_2 <- conv5_1
I1118 02:16:05.868283 22484 net.cpp:356] conv5_2 -> conv5_2
I1118 02:16:05.868289 22484 net.cpp:96] Setting up conv5_2
I1118 02:16:05.922863 22484 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 02:16:05.922901 22484 layer_factory.hpp:78] Creating layer relu5_2
I1118 02:16:05.922909 22484 net.cpp:67] Creating Layer relu5_2
I1118 02:16:05.922914 22484 net.cpp:394] relu5_2 <- conv5_2
I1118 02:16:05.922921 22484 net.cpp:345] relu5_2 -> conv5_2 (in-place)
I1118 02:16:05.922929 22484 net.cpp:96] Setting up relu5_2
I1118 02:16:05.922933 22484 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 02:16:05.922936 22484 layer_factory.hpp:78] Creating layer conv5_3
I1118 02:16:05.922943 22484 net.cpp:67] Creating Layer conv5_3
I1118 02:16:05.922945 22484 net.cpp:394] conv5_3 <- conv5_2
I1118 02:16:05.922951 22484 net.cpp:356] conv5_3 -> conv5_3
I1118 02:16:05.922957 22484 net.cpp:96] Setting up conv5_3
I1118 02:16:05.977911 22484 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 02:16:05.977949 22484 layer_factory.hpp:78] Creating layer relu5_3
I1118 02:16:05.977957 22484 net.cpp:67] Creating Layer relu5_3
I1118 02:16:05.977962 22484 net.cpp:394] relu5_3 <- conv5_3
I1118 02:16:05.977970 22484 net.cpp:345] relu5_3 -> conv5_3 (in-place)
I1118 02:16:05.977977 22484 net.cpp:96] Setting up relu5_3
I1118 02:16:05.977982 22484 net.cpp:103] Top shape: 8 512 14 14 (802816)
I1118 02:16:05.977985 22484 layer_factory.hpp:78] Creating layer pool5
I1118 02:16:05.977998 22484 net.cpp:67] Creating Layer pool5
I1118 02:16:05.978000 22484 net.cpp:394] pool5 <- conv5_3
I1118 02:16:05.978005 22484 net.cpp:356] pool5 -> pool5
I1118 02:16:05.978011 22484 net.cpp:96] Setting up pool5
I1118 02:16:05.978018 22484 net.cpp:103] Top shape: 8 512 7 7 (200704)
I1118 02:16:05.978021 22484 layer_factory.hpp:78] Creating layer fc6
I1118 02:16:05.978027 22484 net.cpp:67] Creating Layer fc6
I1118 02:16:05.978030 22484 net.cpp:394] fc6 <- pool5
I1118 02:16:05.978035 22484 net.cpp:356] fc6 -> fc6
I1118 02:16:05.978041 22484 net.cpp:96] Setting up fc6
I1118 02:16:08.342810 22484 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 02:16:08.342850 22484 layer_factory.hpp:78] Creating layer relu6
I1118 02:16:08.342859 22484 net.cpp:67] Creating Layer relu6
I1118 02:16:08.342864 22484 net.cpp:394] relu6 <- fc6
I1118 02:16:08.342872 22484 net.cpp:345] relu6 -> fc6 (in-place)
I1118 02:16:08.342880 22484 net.cpp:96] Setting up relu6
I1118 02:16:08.342893 22484 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 02:16:08.342897 22484 layer_factory.hpp:78] Creating layer drop6
I1118 02:16:08.342911 22484 net.cpp:67] Creating Layer drop6
I1118 02:16:08.342914 22484 net.cpp:394] drop6 <- fc6
I1118 02:16:08.342918 22484 net.cpp:345] drop6 -> fc6 (in-place)
I1118 02:16:08.342922 22484 net.cpp:96] Setting up drop6
I1118 02:16:08.342926 22484 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 02:16:08.342929 22484 layer_factory.hpp:78] Creating layer fc7
I1118 02:16:08.342936 22484 net.cpp:67] Creating Layer fc7
I1118 02:16:08.342938 22484 net.cpp:394] fc7 <- fc6
I1118 02:16:08.342943 22484 net.cpp:356] fc7 -> fc7
I1118 02:16:08.342949 22484 net.cpp:96] Setting up fc7
I1118 02:16:08.729841 22484 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 02:16:08.729886 22484 layer_factory.hpp:78] Creating layer relu7
I1118 02:16:08.729894 22484 net.cpp:67] Creating Layer relu7
I1118 02:16:08.729898 22484 net.cpp:394] relu7 <- fc7
I1118 02:16:08.729905 22484 net.cpp:345] relu7 -> fc7 (in-place)
I1118 02:16:08.729912 22484 net.cpp:96] Setting up relu7
I1118 02:16:08.729926 22484 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 02:16:08.729929 22484 layer_factory.hpp:78] Creating layer drop7
I1118 02:16:08.729934 22484 net.cpp:67] Creating Layer drop7
I1118 02:16:08.729938 22484 net.cpp:394] drop7 <- fc7
I1118 02:16:08.729943 22484 net.cpp:345] drop7 -> fc7 (in-place)
I1118 02:16:08.729948 22484 net.cpp:96] Setting up drop7
I1118 02:16:08.729951 22484 net.cpp:103] Top shape: 8 4096 1 1 (32768)
I1118 02:16:08.729954 22484 layer_factory.hpp:78] Creating layer fc8_2
I1118 02:16:08.729959 22484 net.cpp:67] Creating Layer fc8_2
I1118 02:16:08.729962 22484 net.cpp:394] fc8_2 <- fc7
I1118 02:16:08.729969 22484 net.cpp:356] fc8_2 -> fc8_2
I1118 02:16:08.729974 22484 net.cpp:96] Setting up fc8_2
I1118 02:16:08.730186 22484 net.cpp:103] Top shape: 8 2 1 1 (16)
I1118 02:16:08.730195 22484 layer_factory.hpp:78] Creating layer fc8_2_fc8_2_0_split
I1118 02:16:08.730200 22484 net.cpp:67] Creating Layer fc8_2_fc8_2_0_split
I1118 02:16:08.730202 22484 net.cpp:394] fc8_2_fc8_2_0_split <- fc8_2
I1118 02:16:08.730206 22484 net.cpp:356] fc8_2_fc8_2_0_split -> fc8_2_fc8_2_0_split_0
I1118 02:16:08.730213 22484 net.cpp:356] fc8_2_fc8_2_0_split -> fc8_2_fc8_2_0_split_1
I1118 02:16:08.730219 22484 net.cpp:96] Setting up fc8_2_fc8_2_0_split
I1118 02:16:08.730226 22484 net.cpp:103] Top shape: 8 2 1 1 (16)
I1118 02:16:08.730229 22484 net.cpp:103] Top shape: 8 2 1 1 (16)
I1118 02:16:08.730232 22484 layer_factory.hpp:78] Creating layer loss
I1118 02:16:08.730237 22484 net.cpp:67] Creating Layer loss
I1118 02:16:08.730242 22484 net.cpp:394] loss <- fc8_2_fc8_2_0_split_0
I1118 02:16:08.730245 22484 net.cpp:394] loss <- label_data_1_split_0
I1118 02:16:08.730250 22484 net.cpp:356] loss -> (automatic)
I1118 02:16:08.730254 22484 net.cpp:96] Setting up loss
I1118 02:16:08.730260 22484 net.cpp:103] Top shape: 1 1 1 1 (1)
I1118 02:16:08.730263 22484 net.cpp:109]     with loss weight 1
I1118 02:16:08.730276 22484 layer_factory.hpp:78] Creating layer accuracy
I1118 02:16:08.730283 22484 net.cpp:67] Creating Layer accuracy
I1118 02:16:08.730285 22484 net.cpp:394] accuracy <- fc8_2_fc8_2_0_split_1
I1118 02:16:08.730288 22484 net.cpp:394] accuracy <- label_data_1_split_1
I1118 02:16:08.730295 22484 net.cpp:356] accuracy -> accuracy
I1118 02:16:08.730300 22484 net.cpp:96] Setting up accuracy
I1118 02:16:08.730309 22484 net.cpp:103] Top shape: 1 1 1 4 (4)
I1118 02:16:08.730311 22484 net.cpp:172] accuracy does not need backward computation.
I1118 02:16:08.730314 22484 net.cpp:170] loss needs backward computation.
I1118 02:16:08.730317 22484 net.cpp:170] fc8_2_fc8_2_0_split needs backward computation.
I1118 02:16:08.730320 22484 net.cpp:170] fc8_2 needs backward computation.
I1118 02:16:08.730324 22484 net.cpp:170] drop7 needs backward computation.
I1118 02:16:08.730325 22484 net.cpp:170] relu7 needs backward computation.
I1118 02:16:08.730329 22484 net.cpp:170] fc7 needs backward computation.
I1118 02:16:08.730331 22484 net.cpp:170] drop6 needs backward computation.
I1118 02:16:08.730334 22484 net.cpp:170] relu6 needs backward computation.
I1118 02:16:08.730343 22484 net.cpp:170] fc6 needs backward computation.
I1118 02:16:08.730347 22484 net.cpp:170] pool5 needs backward computation.
I1118 02:16:08.730350 22484 net.cpp:170] relu5_3 needs backward computation.
I1118 02:16:08.730352 22484 net.cpp:170] conv5_3 needs backward computation.
I1118 02:16:08.730355 22484 net.cpp:170] relu5_2 needs backward computation.
I1118 02:16:08.730358 22484 net.cpp:170] conv5_2 needs backward computation.
I1118 02:16:08.730361 22484 net.cpp:170] relu5_1 needs backward computation.
I1118 02:16:08.730365 22484 net.cpp:170] conv5_1 needs backward computation.
I1118 02:16:08.730367 22484 net.cpp:170] pool4 needs backward computation.
I1118 02:16:08.730370 22484 net.cpp:170] relu4_3 needs backward computation.
I1118 02:16:08.730373 22484 net.cpp:170] conv4_3 needs backward computation.
I1118 02:16:08.730376 22484 net.cpp:170] relu4_2 needs backward computation.
I1118 02:16:08.730378 22484 net.cpp:170] conv4_2 needs backward computation.
I1118 02:16:08.730381 22484 net.cpp:170] relu4_1 needs backward computation.
I1118 02:16:08.730384 22484 net.cpp:170] conv4_1 needs backward computation.
I1118 02:16:08.730387 22484 net.cpp:170] pool3 needs backward computation.
I1118 02:16:08.730391 22484 net.cpp:170] relu3_3 needs backward computation.
I1118 02:16:08.730393 22484 net.cpp:170] conv3_3 needs backward computation.
I1118 02:16:08.730396 22484 net.cpp:170] relu3_2 needs backward computation.
I1118 02:16:08.730398 22484 net.cpp:170] conv3_2 needs backward computation.
I1118 02:16:08.730401 22484 net.cpp:170] relu3_1 needs backward computation.
I1118 02:16:08.730404 22484 net.cpp:170] conv3_1 needs backward computation.
I1118 02:16:08.730407 22484 net.cpp:170] pool2 needs backward computation.
I1118 02:16:08.730409 22484 net.cpp:170] relu2_2 needs backward computation.
I1118 02:16:08.730412 22484 net.cpp:170] conv2_2 needs backward computation.
I1118 02:16:08.730415 22484 net.cpp:170] relu2_1 needs backward computation.
I1118 02:16:08.730418 22484 net.cpp:170] conv2_1 needs backward computation.
I1118 02:16:08.730422 22484 net.cpp:170] pool1 needs backward computation.
I1118 02:16:08.730424 22484 net.cpp:170] relu1_2 needs backward computation.
I1118 02:16:08.730427 22484 net.cpp:170] conv1_2 needs backward computation.
I1118 02:16:08.730429 22484 net.cpp:170] relu1_1 needs backward computation.
I1118 02:16:08.730432 22484 net.cpp:170] conv1_1 needs backward computation.
I1118 02:16:08.730435 22484 net.cpp:172] label_data_1_split does not need backward computation.
I1118 02:16:08.730438 22484 net.cpp:172] data does not need backward computation.
I1118 02:16:08.730440 22484 net.cpp:208] This network produces output accuracy
I1118 02:16:08.730461 22484 net.cpp:467] Collecting Learning Rate and Weight Decay.
I1118 02:16:08.730469 22484 net.cpp:219] Network initialization done.
I1118 02:16:08.730473 22484 net.cpp:220] Memory required for data: 921616692
I1118 02:16:08.730574 22484 solver.cpp:41] Solver scaffolding done.
I1118 02:16:08.730584 22484 caffe.cpp:115] Finetuning from oxford/small.weights
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:505] Reading dangerously large protocol message.  If the message turns out to be larger than 1073741824 bytes, parsing will be halted for security reasons.  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:78] The total number of bytes read was 553432081
I1118 02:16:09.482918 22484 solver.cpp:160] Solving small
I1118 02:16:09.482955 22484 solver.cpp:161] Learning Rate Policy: fixed
I1118 02:16:09.483002 22484 solver.cpp:264] Iteration 0, Testing net (#0)
I1118 02:16:21.024520 22484 solver.cpp:305] Test loss: 0.812834
I1118 02:16:21.024560 22484 solver.cpp:318] mean_score = test_score[0] { = 237} / test_score[1] { = 517 }
I1118 02:16:21.024569 22484 solver.cpp:319]            = 0.458414
I1118 02:16:21.024581 22484 solver.cpp:328]     Test net output #0: accuracy = 0.458414
I1118 02:16:21.024586 22484 solver.cpp:318] mean_score = test_score[2] { = 18} / test_score[3] { = 59 }
I1118 02:16:21.024590 22484 solver.cpp:319]            = 0.305085
I1118 02:16:21.024595 22484 solver.cpp:328]     Test net output #1: accuracy = 0.305085
I1118 02:16:21.024605 22484 solver.cpp:332]     Test net output #2: accuracy = 0.442708
I1118 02:16:21.024610 22484 solver.cpp:334]     Test net output #3: accuracy = 0.381749
I1118 02:16:21.654938 22484 solver.cpp:209] Iteration 0, loss = 1.10121
I1118 02:16:21.654979 22484 solver.cpp:464] Iteration 0, lr = 1e-05
I1118 02:16:23.739239 22484 solver.cpp:209] Iteration 1, loss = 0.692558
I1118 02:16:23.739280 22484 solver.cpp:464] Iteration 1, lr = 1e-05
I1118 02:16:25.808925 22484 solver.cpp:209] Iteration 2, loss = 0.798618
I1118 02:16:25.808955 22484 solver.cpp:464] Iteration 2, lr = 1e-05
I1118 02:16:27.886006 22484 solver.cpp:209] Iteration 3, loss = 0.983252
I1118 02:16:27.886046 22484 solver.cpp:464] Iteration 3, lr = 1e-05
I1118 02:16:29.950505 22484 solver.cpp:209] Iteration 4, loss = 0.830857
I1118 02:16:29.950534 22484 solver.cpp:464] Iteration 4, lr = 1e-05
I1118 02:16:32.020112 22484 solver.cpp:209] Iteration 5, loss = 0.702187
I1118 02:16:32.020205 22484 solver.cpp:464] Iteration 5, lr = 1e-05
I1118 02:16:34.096967 22484 solver.cpp:209] Iteration 6, loss = 0.726826
I1118 02:16:34.097009 22484 solver.cpp:464] Iteration 6, lr = 1e-05
I1118 02:16:36.170567 22484 solver.cpp:209] Iteration 7, loss = 0.660225
I1118 02:16:36.170619 22484 solver.cpp:464] Iteration 7, lr = 1e-05
I1118 02:16:38.244113 22484 solver.cpp:209] Iteration 8, loss = 0.617379
I1118 02:16:38.244141 22484 solver.cpp:464] Iteration 8, lr = 1e-05
I1118 02:16:40.316138 22484 solver.cpp:209] Iteration 9, loss = 0.576909
I1118 02:16:40.316179 22484 solver.cpp:464] Iteration 9, lr = 1e-05
I1118 02:16:42.394973 22484 solver.cpp:209] Iteration 10, loss = 0.458556
I1118 02:16:42.395012 22484 solver.cpp:464] Iteration 10, lr = 1e-05
I1118 02:16:44.471741 22484 solver.cpp:209] Iteration 11, loss = 0.524402
I1118 02:16:44.471781 22484 solver.cpp:464] Iteration 11, lr = 1e-05
I1118 02:16:46.550365 22484 solver.cpp:209] Iteration 12, loss = 0.44048
I1118 02:16:46.550406 22484 solver.cpp:464] Iteration 12, lr = 1e-05
I1118 02:16:48.634516 22484 solver.cpp:209] Iteration 13, loss = 0.631207
I1118 02:16:48.634557 22484 solver.cpp:464] Iteration 13, lr = 1e-05
I1118 02:16:50.715209 22484 solver.cpp:209] Iteration 14, loss = 0.381777
I1118 02:16:50.715252 22484 solver.cpp:464] Iteration 14, lr = 1e-05
I1118 02:16:52.790777 22484 solver.cpp:209] Iteration 15, loss = 0.339417
I1118 02:16:52.790807 22484 solver.cpp:464] Iteration 15, lr = 1e-05
I1118 02:16:54.863973 22484 solver.cpp:209] Iteration 16, loss = 0.442317
I1118 02:16:54.864014 22484 solver.cpp:464] Iteration 16, lr = 1e-05
I1118 02:16:56.941423 22484 solver.cpp:209] Iteration 17, loss = 0.174137
I1118 02:16:56.941452 22484 solver.cpp:464] Iteration 17, lr = 1e-05
I1118 02:16:59.034499 22484 solver.cpp:209] Iteration 18, loss = 0.281395
I1118 02:16:59.034539 22484 solver.cpp:464] Iteration 18, lr = 1e-05
I1118 02:17:01.112051 22484 solver.cpp:209] Iteration 19, loss = 0.222118
I1118 02:17:01.112084 22484 solver.cpp:464] Iteration 19, lr = 1e-05
I1118 02:17:03.193588 22484 solver.cpp:209] Iteration 20, loss = 0.356889
I1118 02:17:03.193660 22484 solver.cpp:464] Iteration 20, lr = 1e-05
I1118 02:17:05.277004 22484 solver.cpp:209] Iteration 21, loss = 0.443367
I1118 02:17:05.277045 22484 solver.cpp:464] Iteration 21, lr = 1e-05
I1118 02:17:07.356032 22484 solver.cpp:209] Iteration 22, loss = 0.445794
I1118 02:17:07.356063 22484 solver.cpp:464] Iteration 22, lr = 1e-05
I1118 02:17:09.436380 22484 solver.cpp:209] Iteration 23, loss = 0.159959
I1118 02:17:09.436410 22484 solver.cpp:464] Iteration 23, lr = 1e-05
I1118 02:17:11.519073 22484 solver.cpp:209] Iteration 24, loss = 0.337418
I1118 02:17:11.519103 22484 solver.cpp:464] Iteration 24, lr = 1e-05
I1118 02:17:13.607352 22484 solver.cpp:209] Iteration 25, loss = 0.178485
I1118 02:17:13.607394 22484 solver.cpp:464] Iteration 25, lr = 1e-05
I1118 02:17:15.693733 22484 solver.cpp:209] Iteration 26, loss = 0.289879
I1118 02:17:15.693775 22484 solver.cpp:464] Iteration 26, lr = 1e-05
I1118 02:17:17.785898 22484 solver.cpp:209] Iteration 27, loss = 0.437153
I1118 02:17:17.785939 22484 solver.cpp:464] Iteration 27, lr = 1e-05
I1118 02:17:19.884533 22484 solver.cpp:209] Iteration 28, loss = 0.16237
I1118 02:17:19.884575 22484 solver.cpp:464] Iteration 28, lr = 1e-05
I1118 02:17:21.971161 22484 solver.cpp:209] Iteration 29, loss = 0.17586
I1118 02:17:21.971204 22484 solver.cpp:464] Iteration 29, lr = 1e-05
I1118 02:17:24.061641 22484 solver.cpp:209] Iteration 30, loss = 0.815144
I1118 02:17:24.061669 22484 solver.cpp:464] Iteration 30, lr = 1e-05
I1118 02:17:26.145206 22484 solver.cpp:209] Iteration 31, loss = 0.329396
I1118 02:17:26.145234 22484 solver.cpp:464] Iteration 31, lr = 1e-05
I1118 02:17:28.241634 22484 solver.cpp:209] Iteration 32, loss = 0.338364
I1118 02:17:28.241664 22484 solver.cpp:464] Iteration 32, lr = 1e-05
I1118 02:17:30.344269 22484 solver.cpp:209] Iteration 33, loss = 0.353642
I1118 02:17:30.344298 22484 solver.cpp:464] Iteration 33, lr = 1e-05
I1118 02:17:32.435001 22484 solver.cpp:209] Iteration 34, loss = 0.24069
I1118 02:17:32.435031 22484 solver.cpp:464] Iteration 34, lr = 1e-05
I1118 02:17:34.523169 22484 solver.cpp:209] Iteration 35, loss = 0.294901
I1118 02:17:34.523259 22484 solver.cpp:464] Iteration 35, lr = 1e-05
I1118 02:17:36.621441 22484 solver.cpp:209] Iteration 36, loss = 0.196204
I1118 02:17:36.621482 22484 solver.cpp:464] Iteration 36, lr = 1e-05
I1118 02:17:38.718724 22484 solver.cpp:209] Iteration 37, loss = 0.579822
I1118 02:17:38.718754 22484 solver.cpp:464] Iteration 37, lr = 1e-05
I1118 02:17:40.807714 22484 solver.cpp:209] Iteration 38, loss = 0.835927
I1118 02:17:40.807749 22484 solver.cpp:464] Iteration 38, lr = 1e-05
I1118 02:17:42.886293 22484 solver.cpp:209] Iteration 39, loss = 0.758214
I1118 02:17:42.886322 22484 solver.cpp:464] Iteration 39, lr = 1e-05
I1118 02:17:44.979395 22484 solver.cpp:209] Iteration 40, loss = 0.290596
I1118 02:17:44.979424 22484 solver.cpp:464] Iteration 40, lr = 1e-05
I1118 02:17:47.140467 22484 solver.cpp:209] Iteration 41, loss = 0.439469
I1118 02:17:47.140508 22484 solver.cpp:464] Iteration 41, lr = 1e-05
I1118 02:17:49.371975 22484 solver.cpp:209] Iteration 42, loss = 0.137997
I1118 02:17:49.372004 22484 solver.cpp:464] Iteration 42, lr = 1e-05
I1118 02:17:51.603663 22484 solver.cpp:209] Iteration 43, loss = 0.335272
I1118 02:17:51.603693 22484 solver.cpp:464] Iteration 43, lr = 1e-05
I1118 02:17:53.828807 22484 solver.cpp:209] Iteration 44, loss = 0.213918
I1118 02:17:53.828835 22484 solver.cpp:464] Iteration 44, lr = 1e-05
I1118 02:17:56.055769 22484 solver.cpp:209] Iteration 45, loss = 0.231475
I1118 02:17:56.055810 22484 solver.cpp:464] Iteration 45, lr = 1e-05
I1118 02:17:58.303449 22484 solver.cpp:209] Iteration 46, loss = 0.155705
I1118 02:17:58.303489 22484 solver.cpp:464] Iteration 46, lr = 1e-05
I1118 02:18:00.536953 22484 solver.cpp:209] Iteration 47, loss = 0.136391
I1118 02:18:00.536994 22484 solver.cpp:464] Iteration 47, lr = 1e-05
I1118 02:18:02.777029 22484 solver.cpp:209] Iteration 48, loss = 0.359111
I1118 02:18:02.777057 22484 solver.cpp:464] Iteration 48, lr = 1e-05
I1118 02:18:05.004061 22484 solver.cpp:209] Iteration 49, loss = 1.01103
I1118 02:18:05.004137 22484 solver.cpp:464] Iteration 49, lr = 1e-05
I1118 02:18:05.004744 22484 solver.cpp:264] Iteration 50, Testing net (#0)
I1118 02:18:18.897907 22484 solver.cpp:305] Test loss: 0.342376
I1118 02:18:18.897948 22484 solver.cpp:318] mean_score = test_score[0] { = 517} / test_score[1] { = 517 }
I1118 02:18:18.897955 22484 solver.cpp:319]            = 1
I1118 02:18:18.897960 22484 solver.cpp:328]     Test net output #0: accuracy = 1
I1118 02:18:18.897964 22484 solver.cpp:318] mean_score = test_score[2] { = 8.46356e-37} / test_score[3] { = 59 }
I1118 02:18:18.897970 22484 solver.cpp:319]            = 1.4345e-38
I1118 02:18:18.897974 22484 solver.cpp:328]     Test net output #1: accuracy = 1.4345e-38
I1118 02:18:18.897979 22484 solver.cpp:332]     Test net output #2: accuracy = 0.897569
I1118 02:18:18.897982 22484 solver.cpp:334]     Test net output #3: accuracy = 0.5
I1118 02:18:19.544304 22484 solver.cpp:209] Iteration 50, loss = 0.350412
I1118 02:18:19.544344 22484 solver.cpp:464] Iteration 50, lr = 1e-05
I1118 02:18:21.768105 22484 solver.cpp:209] Iteration 51, loss = 0.362158
I1118 02:18:21.768134 22484 solver.cpp:464] Iteration 51, lr = 1e-05
I1118 02:18:24.000757 22484 solver.cpp:209] Iteration 52, loss = 0.348944
I1118 02:18:24.000787 22484 solver.cpp:464] Iteration 52, lr = 1e-05
I1118 02:18:26.231140 22484 solver.cpp:209] Iteration 53, loss = 0.488989
I1118 02:18:26.231169 22484 solver.cpp:464] Iteration 53, lr = 1e-05
I1118 02:18:28.466408 22484 solver.cpp:209] Iteration 54, loss = 0.0574045
I1118 02:18:28.466436 22484 solver.cpp:464] Iteration 54, lr = 1e-05
I1118 02:18:30.702139 22484 solver.cpp:209] Iteration 55, loss = 0.415844
I1118 02:18:30.702180 22484 solver.cpp:464] Iteration 55, lr = 1e-05
I1118 02:18:32.931823 22484 solver.cpp:209] Iteration 56, loss = 0.500375
I1118 02:18:32.931864 22484 solver.cpp:464] Iteration 56, lr = 1e-05
I1118 02:18:35.167069 22484 solver.cpp:209] Iteration 57, loss = 0.292815
I1118 02:18:35.167152 22484 solver.cpp:464] Iteration 57, lr = 1e-05
I1118 02:18:37.393107 22484 solver.cpp:209] Iteration 58, loss = 0.360852
I1118 02:18:37.393136 22484 solver.cpp:464] Iteration 58, lr = 1e-05
I1118 02:18:39.621892 22484 solver.cpp:209] Iteration 59, loss = 0.311357
I1118 02:18:39.621933 22484 solver.cpp:464] Iteration 59, lr = 1e-05
I1118 02:18:41.849673 22484 solver.cpp:209] Iteration 60, loss = 0.296852
I1118 02:18:41.849714 22484 solver.cpp:464] Iteration 60, lr = 1e-05
I1118 02:18:44.079545 22484 solver.cpp:209] Iteration 61, loss = 0.419181
I1118 02:18:44.079584 22484 solver.cpp:464] Iteration 61, lr = 1e-05
I1118 02:18:46.319867 22484 solver.cpp:209] Iteration 62, loss = 0.681126
I1118 02:18:46.319908 22484 solver.cpp:464] Iteration 62, lr = 1e-05
I1118 02:18:48.552675 22484 solver.cpp:209] Iteration 63, loss = 0.568662
I1118 02:18:48.552705 22484 solver.cpp:464] Iteration 63, lr = 1e-05
I1118 02:18:50.786403 22484 solver.cpp:209] Iteration 64, loss = 0.91199
I1118 02:18:50.786429 22484 solver.cpp:464] Iteration 64, lr = 1e-05
I1118 02:18:53.016561 22484 solver.cpp:209] Iteration 65, loss = 0.219995
I1118 02:18:53.016602 22484 solver.cpp:464] Iteration 65, lr = 1e-05
I1118 02:18:55.244626 22484 solver.cpp:209] Iteration 66, loss = 0.520399
I1118 02:18:55.244667 22484 solver.cpp:464] Iteration 66, lr = 1e-05
I1118 02:18:57.476583 22484 solver.cpp:209] Iteration 67, loss = 0.331481
I1118 02:18:57.476610 22484 solver.cpp:464] Iteration 67, lr = 1e-05
I1118 02:18:59.702774 22484 solver.cpp:209] Iteration 68, loss = 0.340428
I1118 02:18:59.702803 22484 solver.cpp:464] Iteration 68, lr = 1e-05
I1118 02:19:01.939784 22484 solver.cpp:209] Iteration 69, loss = 0.599637
I1118 02:19:01.939811 22484 solver.cpp:464] Iteration 69, lr = 1e-05
I1118 02:19:04.172206 22484 solver.cpp:209] Iteration 70, loss = 0.401144
I1118 02:19:04.172247 22484 solver.cpp:464] Iteration 70, lr = 1e-05
I1118 02:19:06.414288 22484 solver.cpp:209] Iteration 71, loss = 0.584119
I1118 02:19:06.414366 22484 solver.cpp:464] Iteration 71, lr = 1e-05
I1118 02:19:08.657820 22484 solver.cpp:209] Iteration 72, loss = 0.419742
I1118 02:19:08.657860 22484 solver.cpp:464] Iteration 72, lr = 1e-05
I1118 02:19:10.896932 22484 solver.cpp:209] Iteration 73, loss = 0.368579
I1118 02:19:10.896958 22484 solver.cpp:464] Iteration 73, lr = 1e-05
I1118 02:19:13.140179 22484 solver.cpp:209] Iteration 74, loss = 0.793123
I1118 02:19:13.140220 22484 solver.cpp:464] Iteration 74, lr = 1e-05
I1118 02:19:15.384423 22484 solver.cpp:209] Iteration 75, loss = 0.17969
I1118 02:19:15.384450 22484 solver.cpp:464] Iteration 75, lr = 1e-05
I1118 02:19:17.637837 22484 solver.cpp:209] Iteration 76, loss = 0.266427
I1118 02:19:17.637866 22484 solver.cpp:464] Iteration 76, lr = 1e-05
I1118 02:19:19.890000 22484 solver.cpp:209] Iteration 77, loss = 0.179198
I1118 02:19:19.890030 22484 solver.cpp:464] Iteration 77, lr = 1e-05
I1118 02:19:22.136579 22484 solver.cpp:209] Iteration 78, loss = 0.480241
I1118 02:19:22.136607 22484 solver.cpp:464] Iteration 78, lr = 1e-05
I1118 02:19:24.381705 22484 solver.cpp:209] Iteration 79, loss = 0.356293
I1118 02:19:24.381734 22484 solver.cpp:464] Iteration 79, lr = 1e-05
I1118 02:19:26.621994 22484 solver.cpp:209] Iteration 80, loss = 0.174077
I1118 02:19:26.622022 22484 solver.cpp:464] Iteration 80, lr = 1e-05
I1118 02:19:28.868351 22484 solver.cpp:209] Iteration 81, loss = 0.290172
I1118 02:19:28.868391 22484 solver.cpp:464] Iteration 81, lr = 1e-05
I1118 02:19:31.113453 22484 solver.cpp:209] Iteration 82, loss = 0.329334
I1118 02:19:31.113481 22484 solver.cpp:464] Iteration 82, lr = 1e-05
I1118 02:19:33.355784 22484 solver.cpp:209] Iteration 83, loss = 0.456773
I1118 02:19:33.355813 22484 solver.cpp:464] Iteration 83, lr = 1e-05
I1118 02:19:35.601796 22484 solver.cpp:209] Iteration 84, loss = 0.311199
I1118 02:19:35.601837 22484 solver.cpp:464] Iteration 84, lr = 1e-05
I1118 02:19:37.844643 22484 solver.cpp:209] Iteration 85, loss = 0.795798
I1118 02:19:37.844702 22484 solver.cpp:464] Iteration 85, lr = 1e-05
I1118 02:19:40.109889 22484 solver.cpp:209] Iteration 86, loss = 0.445647
I1118 02:19:40.109918 22484 solver.cpp:464] Iteration 86, lr = 1e-05
I1118 02:19:42.351608 22484 solver.cpp:209] Iteration 87, loss = 0.463054
I1118 02:19:42.351637 22484 solver.cpp:464] Iteration 87, lr = 1e-05
I1118 02:19:44.588858 22484 solver.cpp:209] Iteration 88, loss = 0.10602
I1118 02:19:44.588887 22484 solver.cpp:464] Iteration 88, lr = 1e-05
I1118 02:19:46.831007 22484 solver.cpp:209] Iteration 89, loss = 0.110826
I1118 02:19:46.831035 22484 solver.cpp:464] Iteration 89, lr = 1e-05
I1118 02:19:49.070279 22484 solver.cpp:209] Iteration 90, loss = 0.33561
I1118 02:19:49.070320 22484 solver.cpp:464] Iteration 90, lr = 1e-05
I1118 02:19:51.328753 22484 solver.cpp:209] Iteration 91, loss = 0.566476
I1118 02:19:51.328783 22484 solver.cpp:464] Iteration 91, lr = 1e-05
I1118 02:19:53.580107 22484 solver.cpp:209] Iteration 92, loss = 0.228082
I1118 02:19:53.580135 22484 solver.cpp:464] Iteration 92, lr = 1e-05
I1118 02:19:55.823277 22484 solver.cpp:209] Iteration 93, loss = 0.816454
I1118 02:19:55.823305 22484 solver.cpp:464] Iteration 93, lr = 1e-05
I1118 02:19:58.065980 22484 solver.cpp:209] Iteration 94, loss = 0.347245
I1118 02:19:58.066020 22484 solver.cpp:464] Iteration 94, lr = 1e-05
I1118 02:20:00.302458 22484 solver.cpp:209] Iteration 95, loss = 0.360073
I1118 02:20:00.302498 22484 solver.cpp:464] Iteration 95, lr = 1e-05
I1118 02:20:02.543818 22484 solver.cpp:209] Iteration 96, loss = 0.399038
I1118 02:20:02.543846 22484 solver.cpp:464] Iteration 96, lr = 1e-05
I1118 02:20:04.784560 22484 solver.cpp:209] Iteration 97, loss = 0.400258
I1118 02:20:04.784601 22484 solver.cpp:464] Iteration 97, lr = 1e-05
I1118 02:20:07.024783 22484 solver.cpp:209] Iteration 98, loss = 0.581418
I1118 02:20:07.024812 22484 solver.cpp:464] Iteration 98, lr = 1e-05
I1118 02:20:09.275686 22484 solver.cpp:209] Iteration 99, loss = 0.23776
I1118 02:20:09.275740 22484 solver.cpp:464] Iteration 99, lr = 1e-05
I1118 02:20:09.276327 22484 solver.cpp:264] Iteration 100, Testing net (#0)
I1118 02:20:23.393131 22484 solver.cpp:305] Test loss: 0.312411
I1118 02:20:23.393172 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:20:23.393179 22484 solver.cpp:319]            = 0.994197
I1118 02:20:23.393184 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:20:23.393188 22484 solver.cpp:318] mean_score = test_score[2] { = 8.46356e-37} / test_score[3] { = 59 }
I1118 02:20:23.393194 22484 solver.cpp:319]            = 1.4345e-38
I1118 02:20:23.393198 22484 solver.cpp:328]     Test net output #1: accuracy = 1.4345e-38
I1118 02:20:23.393203 22484 solver.cpp:332]     Test net output #2: accuracy = 0.892361
I1118 02:20:23.393206 22484 solver.cpp:334]     Test net output #3: accuracy = 0.497099
I1118 02:20:24.063616 22484 solver.cpp:209] Iteration 100, loss = 0.410304
I1118 02:20:24.063644 22484 solver.cpp:464] Iteration 100, lr = 1e-05
I1118 02:20:26.371572 22484 solver.cpp:209] Iteration 101, loss = 0.316401
I1118 02:20:26.371600 22484 solver.cpp:464] Iteration 101, lr = 1e-05
I1118 02:20:28.691228 22484 solver.cpp:209] Iteration 102, loss = 0.21889
I1118 02:20:28.691269 22484 solver.cpp:464] Iteration 102, lr = 1e-05
I1118 02:20:31.011765 22484 solver.cpp:209] Iteration 103, loss = 0.322333
I1118 02:20:31.011791 22484 solver.cpp:464] Iteration 103, lr = 1e-05
I1118 02:20:33.326041 22484 solver.cpp:209] Iteration 104, loss = 0.362083
I1118 02:20:33.326082 22484 solver.cpp:464] Iteration 104, lr = 1e-05
I1118 02:20:35.636486 22484 solver.cpp:209] Iteration 105, loss = 0.107883
I1118 02:20:35.636514 22484 solver.cpp:464] Iteration 105, lr = 1e-05
I1118 02:20:37.946738 22484 solver.cpp:209] Iteration 106, loss = 0.505242
I1118 02:20:37.946766 22484 solver.cpp:464] Iteration 106, lr = 1e-05
I1118 02:20:40.260498 22484 solver.cpp:209] Iteration 107, loss = 0.357894
I1118 02:20:40.260555 22484 solver.cpp:464] Iteration 107, lr = 1e-05
I1118 02:20:42.576550 22484 solver.cpp:209] Iteration 108, loss = 0.190596
I1118 02:20:42.576577 22484 solver.cpp:464] Iteration 108, lr = 1e-05
I1118 02:20:44.882388 22484 solver.cpp:209] Iteration 109, loss = 0.266366
I1118 02:20:44.882427 22484 solver.cpp:464] Iteration 109, lr = 1e-05
I1118 02:20:47.182119 22484 solver.cpp:209] Iteration 110, loss = 0.575149
I1118 02:20:47.182160 22484 solver.cpp:464] Iteration 110, lr = 1e-05
I1118 02:20:49.488735 22484 solver.cpp:209] Iteration 111, loss = 0.332772
I1118 02:20:49.488764 22484 solver.cpp:464] Iteration 111, lr = 1e-05
I1118 02:20:51.796511 22484 solver.cpp:209] Iteration 112, loss = 0.199178
I1118 02:20:51.796540 22484 solver.cpp:464] Iteration 112, lr = 1e-05
I1118 02:20:54.101994 22484 solver.cpp:209] Iteration 113, loss = 0.158061
I1118 02:20:54.102036 22484 solver.cpp:464] Iteration 113, lr = 1e-05
I1118 02:20:56.414476 22484 solver.cpp:209] Iteration 114, loss = 0.274505
I1118 02:20:56.414518 22484 solver.cpp:464] Iteration 114, lr = 1e-05
I1118 02:20:58.725800 22484 solver.cpp:209] Iteration 115, loss = 0.516189
I1118 02:20:58.725827 22484 solver.cpp:464] Iteration 115, lr = 1e-05
I1118 02:21:01.006618 22484 solver.cpp:209] Iteration 116, loss = 0.529277
I1118 02:21:01.006644 22484 solver.cpp:464] Iteration 116, lr = 1e-05
I1118 02:21:03.244199 22484 solver.cpp:209] Iteration 117, loss = 0.181821
I1118 02:21:03.244228 22484 solver.cpp:464] Iteration 117, lr = 1e-05
I1118 02:21:05.473760 22484 solver.cpp:209] Iteration 118, loss = 0.377051
I1118 02:21:05.473788 22484 solver.cpp:464] Iteration 118, lr = 1e-05
I1118 02:21:07.700568 22484 solver.cpp:209] Iteration 119, loss = 0.184941
I1118 02:21:07.700609 22484 solver.cpp:464] Iteration 119, lr = 1e-05
I1118 02:21:09.919915 22484 solver.cpp:209] Iteration 120, loss = 0.0597947
I1118 02:21:09.919945 22484 solver.cpp:464] Iteration 120, lr = 1e-05
I1118 02:21:12.143671 22484 solver.cpp:209] Iteration 121, loss = 0.131478
I1118 02:21:12.143714 22484 solver.cpp:464] Iteration 121, lr = 1e-05
I1118 02:21:14.376480 22484 solver.cpp:209] Iteration 122, loss = 0.31288
I1118 02:21:14.376520 22484 solver.cpp:464] Iteration 122, lr = 1e-05
I1118 02:21:16.610602 22484 solver.cpp:209] Iteration 123, loss = 0.0591864
I1118 02:21:16.610631 22484 solver.cpp:464] Iteration 123, lr = 1e-05
I1118 02:21:18.846117 22484 solver.cpp:209] Iteration 124, loss = 0.260875
I1118 02:21:18.846156 22484 solver.cpp:464] Iteration 124, lr = 1e-05
I1118 02:21:21.094673 22484 solver.cpp:209] Iteration 125, loss = 0.34305
I1118 02:21:21.094698 22484 solver.cpp:464] Iteration 125, lr = 1e-05
I1118 02:21:23.335089 22484 solver.cpp:209] Iteration 126, loss = 0.391966
I1118 02:21:23.335117 22484 solver.cpp:464] Iteration 126, lr = 1e-05
I1118 02:21:25.577605 22484 solver.cpp:209] Iteration 127, loss = 0.164627
I1118 02:21:25.577643 22484 solver.cpp:464] Iteration 127, lr = 1e-05
I1118 02:21:27.839125 22484 solver.cpp:209] Iteration 128, loss = 0.160803
I1118 02:21:27.839153 22484 solver.cpp:464] Iteration 128, lr = 1e-05
I1118 02:21:30.116529 22484 solver.cpp:209] Iteration 129, loss = 0.375363
I1118 02:21:30.116557 22484 solver.cpp:464] Iteration 129, lr = 1e-05
I1118 02:21:32.390144 22484 solver.cpp:209] Iteration 130, loss = 0.48057
I1118 02:21:32.390184 22484 solver.cpp:464] Iteration 130, lr = 1e-05
I1118 02:21:34.674219 22484 solver.cpp:209] Iteration 131, loss = 0.128543
I1118 02:21:34.674247 22484 solver.cpp:464] Iteration 131, lr = 1e-05
I1118 02:21:36.953552 22484 solver.cpp:209] Iteration 132, loss = 0.221777
I1118 02:21:36.953593 22484 solver.cpp:464] Iteration 132, lr = 1e-05
I1118 02:21:39.231083 22484 solver.cpp:209] Iteration 133, loss = 0.671542
I1118 02:21:39.231124 22484 solver.cpp:464] Iteration 133, lr = 1e-05
I1118 02:21:41.517863 22484 solver.cpp:209] Iteration 134, loss = 0.340967
I1118 02:21:41.517892 22484 solver.cpp:464] Iteration 134, lr = 1e-05
I1118 02:21:43.792096 22484 solver.cpp:209] Iteration 135, loss = 0.202982
I1118 02:21:43.792179 22484 solver.cpp:464] Iteration 135, lr = 1e-05
I1118 02:21:46.059861 22484 solver.cpp:209] Iteration 136, loss = 0.375197
I1118 02:21:46.059901 22484 solver.cpp:464] Iteration 136, lr = 1e-05
I1118 02:21:48.335831 22484 solver.cpp:209] Iteration 137, loss = 0.240573
I1118 02:21:48.335870 22484 solver.cpp:464] Iteration 137, lr = 1e-05
I1118 02:21:50.603667 22484 solver.cpp:209] Iteration 138, loss = 0.302735
I1118 02:21:50.603706 22484 solver.cpp:464] Iteration 138, lr = 1e-05
I1118 02:21:52.878429 22484 solver.cpp:209] Iteration 139, loss = 0.295614
I1118 02:21:52.878458 22484 solver.cpp:464] Iteration 139, lr = 1e-05
I1118 02:21:55.156852 22484 solver.cpp:209] Iteration 140, loss = 0.418466
I1118 02:21:55.156893 22484 solver.cpp:464] Iteration 140, lr = 1e-05
I1118 02:21:57.436390 22484 solver.cpp:209] Iteration 141, loss = 0.531812
I1118 02:21:57.436417 22484 solver.cpp:464] Iteration 141, lr = 1e-05
I1118 02:21:59.716207 22484 solver.cpp:209] Iteration 142, loss = 0.575745
I1118 02:21:59.716235 22484 solver.cpp:464] Iteration 142, lr = 1e-05
I1118 02:22:01.991267 22484 solver.cpp:209] Iteration 143, loss = 0.151571
I1118 02:22:01.991307 22484 solver.cpp:464] Iteration 143, lr = 1e-05
I1118 02:22:04.267858 22484 solver.cpp:209] Iteration 144, loss = 0.276611
I1118 02:22:04.267899 22484 solver.cpp:464] Iteration 144, lr = 1e-05
I1118 02:22:06.541965 22484 solver.cpp:209] Iteration 145, loss = 0.462462
I1118 02:22:06.541995 22484 solver.cpp:464] Iteration 145, lr = 1e-05
I1118 02:22:08.811080 22484 solver.cpp:209] Iteration 146, loss = 0.171127
I1118 02:22:08.811110 22484 solver.cpp:464] Iteration 146, lr = 1e-05
I1118 02:22:11.086354 22484 solver.cpp:209] Iteration 147, loss = 0.285351
I1118 02:22:11.086380 22484 solver.cpp:464] Iteration 147, lr = 1e-05
I1118 02:22:13.357808 22484 solver.cpp:209] Iteration 148, loss = 0.247623
I1118 02:22:13.357851 22484 solver.cpp:464] Iteration 148, lr = 1e-05
I1118 02:22:15.634407 22484 solver.cpp:209] Iteration 149, loss = 0.253946
I1118 02:22:15.634450 22484 solver.cpp:464] Iteration 149, lr = 1e-05
I1118 02:22:15.635054 22484 solver.cpp:264] Iteration 150, Testing net (#0)
I1118 02:22:29.844857 22484 solver.cpp:305] Test loss: 0.314595
I1118 02:22:29.844887 22484 solver.cpp:318] mean_score = test_score[0] { = 516} / test_score[1] { = 517 }
I1118 02:22:29.844907 22484 solver.cpp:319]            = 0.998066
I1118 02:22:29.844910 22484 solver.cpp:328]     Test net output #0: accuracy = 0.998066
I1118 02:22:29.844915 22484 solver.cpp:318] mean_score = test_score[2] { = 8.46356e-37} / test_score[3] { = 59 }
I1118 02:22:29.844920 22484 solver.cpp:319]            = 1.4345e-38
I1118 02:22:29.844924 22484 solver.cpp:328]     Test net output #1: accuracy = 1.4345e-38
I1118 02:22:29.844928 22484 solver.cpp:332]     Test net output #2: accuracy = 0.895833
I1118 02:22:29.844933 22484 solver.cpp:334]     Test net output #3: accuracy = 0.499033
I1118 02:22:30.505960 22484 solver.cpp:209] Iteration 150, loss = 0.261957
I1118 02:22:30.505987 22484 solver.cpp:464] Iteration 150, lr = 1e-05
I1118 02:22:32.774623 22484 solver.cpp:209] Iteration 151, loss = 0.407982
I1118 02:22:32.774652 22484 solver.cpp:464] Iteration 151, lr = 1e-05
I1118 02:22:35.045982 22484 solver.cpp:209] Iteration 152, loss = 0.727764
I1118 02:22:35.046023 22484 solver.cpp:464] Iteration 152, lr = 1e-05
I1118 02:22:37.318131 22484 solver.cpp:209] Iteration 153, loss = 0.346888
I1118 02:22:37.318173 22484 solver.cpp:464] Iteration 153, lr = 1e-05
I1118 02:22:39.605911 22484 solver.cpp:209] Iteration 154, loss = 0.344364
I1118 02:22:39.605939 22484 solver.cpp:464] Iteration 154, lr = 1e-05
I1118 02:22:41.883417 22484 solver.cpp:209] Iteration 155, loss = 0.36209
I1118 02:22:41.883458 22484 solver.cpp:464] Iteration 155, lr = 1e-05
I1118 02:22:44.162083 22484 solver.cpp:209] Iteration 156, loss = 0.422905
I1118 02:22:44.162125 22484 solver.cpp:464] Iteration 156, lr = 1e-05
I1118 02:22:46.445116 22484 solver.cpp:209] Iteration 157, loss = 0.153313
I1118 02:22:46.445199 22484 solver.cpp:464] Iteration 157, lr = 1e-05
I1118 02:22:48.720551 22484 solver.cpp:209] Iteration 158, loss = 0.412268
I1118 02:22:48.720592 22484 solver.cpp:464] Iteration 158, lr = 1e-05
I1118 02:22:50.995657 22484 solver.cpp:209] Iteration 159, loss = 0.272256
I1118 02:22:50.995698 22484 solver.cpp:464] Iteration 159, lr = 1e-05
I1118 02:22:53.264309 22484 solver.cpp:209] Iteration 160, loss = 0.36702
I1118 02:22:53.264350 22484 solver.cpp:464] Iteration 160, lr = 1e-05
I1118 02:22:55.534775 22484 solver.cpp:209] Iteration 161, loss = 0.322651
I1118 02:22:55.534803 22484 solver.cpp:464] Iteration 161, lr = 1e-05
I1118 02:22:57.810065 22484 solver.cpp:209] Iteration 162, loss = 0.344957
I1118 02:22:57.810093 22484 solver.cpp:464] Iteration 162, lr = 1e-05
I1118 02:23:00.087944 22484 solver.cpp:209] Iteration 163, loss = 0.428444
I1118 02:23:00.087985 22484 solver.cpp:464] Iteration 163, lr = 1e-05
I1118 02:23:02.368662 22484 solver.cpp:209] Iteration 164, loss = 0.279324
I1118 02:23:02.368691 22484 solver.cpp:464] Iteration 164, lr = 1e-05
I1118 02:23:04.645802 22484 solver.cpp:209] Iteration 165, loss = 0.63981
I1118 02:23:04.645843 22484 solver.cpp:464] Iteration 165, lr = 1e-05
I1118 02:23:06.925817 22484 solver.cpp:209] Iteration 166, loss = 0.447778
I1118 02:23:06.925858 22484 solver.cpp:464] Iteration 166, lr = 1e-05
I1118 02:23:09.201117 22484 solver.cpp:209] Iteration 167, loss = 0.792628
I1118 02:23:09.201156 22484 solver.cpp:464] Iteration 167, lr = 1e-05
I1118 02:23:11.441985 22484 solver.cpp:209] Iteration 168, loss = 0.344311
I1118 02:23:11.442025 22484 solver.cpp:464] Iteration 168, lr = 1e-05
I1118 02:23:13.686784 22484 solver.cpp:209] Iteration 169, loss = 0.457147
I1118 02:23:13.686812 22484 solver.cpp:464] Iteration 169, lr = 1e-05
I1118 02:23:15.925778 22484 solver.cpp:209] Iteration 170, loss = 0.286209
I1118 02:23:15.925819 22484 solver.cpp:464] Iteration 170, lr = 1e-05
I1118 02:23:18.170089 22484 solver.cpp:209] Iteration 171, loss = 0.369248
I1118 02:23:18.170143 22484 solver.cpp:464] Iteration 171, lr = 1e-05
I1118 02:23:20.442909 22484 solver.cpp:209] Iteration 172, loss = 0.301647
I1118 02:23:20.442937 22484 solver.cpp:464] Iteration 172, lr = 1e-05
I1118 02:23:22.687451 22484 solver.cpp:209] Iteration 173, loss = 0.650238
I1118 02:23:22.687480 22484 solver.cpp:464] Iteration 173, lr = 1e-05
I1118 02:23:24.940407 22484 solver.cpp:209] Iteration 174, loss = 0.376935
I1118 02:23:24.940448 22484 solver.cpp:464] Iteration 174, lr = 1e-05
I1118 02:23:27.185678 22484 solver.cpp:209] Iteration 175, loss = 0.415526
I1118 02:23:27.185719 22484 solver.cpp:464] Iteration 175, lr = 1e-05
I1118 02:23:29.426492 22484 solver.cpp:209] Iteration 176, loss = 0.285388
I1118 02:23:29.426534 22484 solver.cpp:464] Iteration 176, lr = 1e-05
I1118 02:23:31.668333 22484 solver.cpp:209] Iteration 177, loss = 0.604053
I1118 02:23:31.668361 22484 solver.cpp:464] Iteration 177, lr = 1e-05
I1118 02:23:33.907680 22484 solver.cpp:209] Iteration 178, loss = 0.10175
I1118 02:23:33.907721 22484 solver.cpp:464] Iteration 178, lr = 1e-05
I1118 02:23:36.159641 22484 solver.cpp:209] Iteration 179, loss = 0.521927
I1118 02:23:36.159682 22484 solver.cpp:464] Iteration 179, lr = 1e-05
I1118 02:23:38.406950 22484 solver.cpp:209] Iteration 180, loss = 0.171708
I1118 02:23:38.406978 22484 solver.cpp:464] Iteration 180, lr = 1e-05
I1118 02:23:40.651087 22484 solver.cpp:209] Iteration 181, loss = 0.522126
I1118 02:23:40.651128 22484 solver.cpp:464] Iteration 181, lr = 1e-05
I1118 02:23:42.915344 22484 solver.cpp:209] Iteration 182, loss = 0.307036
I1118 02:23:42.915372 22484 solver.cpp:464] Iteration 182, lr = 1e-05
I1118 02:23:45.159441 22484 solver.cpp:209] Iteration 183, loss = 0.269352
I1118 02:23:45.159482 22484 solver.cpp:464] Iteration 183, lr = 1e-05
I1118 02:23:47.403429 22484 solver.cpp:209] Iteration 184, loss = 0.283731
I1118 02:23:47.403456 22484 solver.cpp:464] Iteration 184, lr = 1e-05
I1118 02:23:49.644986 22484 solver.cpp:209] Iteration 185, loss = 0.354311
I1118 02:23:49.645045 22484 solver.cpp:464] Iteration 185, lr = 1e-05
I1118 02:23:51.892050 22484 solver.cpp:209] Iteration 186, loss = 0.396481
I1118 02:23:51.892091 22484 solver.cpp:464] Iteration 186, lr = 1e-05
I1118 02:23:54.132009 22484 solver.cpp:209] Iteration 187, loss = 0.443914
I1118 02:23:54.132050 22484 solver.cpp:464] Iteration 187, lr = 1e-05
I1118 02:23:56.377534 22484 solver.cpp:209] Iteration 188, loss = 0.650334
I1118 02:23:56.377575 22484 solver.cpp:464] Iteration 188, lr = 1e-05
I1118 02:23:58.630144 22484 solver.cpp:209] Iteration 189, loss = 0.483389
I1118 02:23:58.630173 22484 solver.cpp:464] Iteration 189, lr = 1e-05
I1118 02:24:00.876670 22484 solver.cpp:209] Iteration 190, loss = 0.27743
I1118 02:24:00.876698 22484 solver.cpp:464] Iteration 190, lr = 1e-05
I1118 02:24:03.118227 22484 solver.cpp:209] Iteration 191, loss = 0.135966
I1118 02:24:03.118268 22484 solver.cpp:464] Iteration 191, lr = 1e-05
I1118 02:24:05.358078 22484 solver.cpp:209] Iteration 192, loss = 0.1045
I1118 02:24:05.358119 22484 solver.cpp:464] Iteration 192, lr = 1e-05
I1118 02:24:07.596846 22484 solver.cpp:209] Iteration 193, loss = 0.441382
I1118 02:24:07.596875 22484 solver.cpp:464] Iteration 193, lr = 1e-05
I1118 02:24:09.838670 22484 solver.cpp:209] Iteration 194, loss = 0.337448
I1118 02:24:09.838699 22484 solver.cpp:464] Iteration 194, lr = 1e-05
I1118 02:24:12.097183 22484 solver.cpp:209] Iteration 195, loss = 0.341669
I1118 02:24:12.097230 22484 solver.cpp:464] Iteration 195, lr = 1e-05
I1118 02:24:14.341608 22484 solver.cpp:209] Iteration 196, loss = 0.423202
I1118 02:24:14.341639 22484 solver.cpp:464] Iteration 196, lr = 1e-05
I1118 02:24:16.588438 22484 solver.cpp:209] Iteration 197, loss = 0.290645
I1118 02:24:16.588469 22484 solver.cpp:464] Iteration 197, lr = 1e-05
I1118 02:24:18.826511 22484 solver.cpp:209] Iteration 198, loss = 0.267727
I1118 02:24:18.826540 22484 solver.cpp:464] Iteration 198, lr = 1e-05
I1118 02:24:21.067292 22484 solver.cpp:209] Iteration 199, loss = 0.410244
I1118 02:24:21.067369 22484 solver.cpp:464] Iteration 199, lr = 1e-05
I1118 02:24:21.067975 22484 solver.cpp:264] Iteration 200, Testing net (#0)
I1118 02:24:35.062384 22484 solver.cpp:305] Test loss: 0.304525
I1118 02:24:35.062424 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:24:35.062432 22484 solver.cpp:319]            = 0.994197
I1118 02:24:35.062436 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:24:35.062441 22484 solver.cpp:318] mean_score = test_score[2] { = 8.46356e-37} / test_score[3] { = 59 }
I1118 02:24:35.062448 22484 solver.cpp:319]            = 1.4345e-38
I1118 02:24:35.062450 22484 solver.cpp:328]     Test net output #1: accuracy = 1.4345e-38
I1118 02:24:35.062455 22484 solver.cpp:332]     Test net output #2: accuracy = 0.892361
I1118 02:24:35.062459 22484 solver.cpp:334]     Test net output #3: accuracy = 0.497099
I1118 02:24:35.712013 22484 solver.cpp:209] Iteration 200, loss = 0.314095
I1118 02:24:35.712041 22484 solver.cpp:464] Iteration 200, lr = 1e-05
I1118 02:24:37.968680 22484 solver.cpp:209] Iteration 201, loss = 0.540102
I1118 02:24:37.968710 22484 solver.cpp:464] Iteration 201, lr = 1e-05
I1118 02:24:40.236765 22484 solver.cpp:209] Iteration 202, loss = 0.174764
I1118 02:24:40.236806 22484 solver.cpp:464] Iteration 202, lr = 1e-05
I1118 02:24:42.510383 22484 solver.cpp:209] Iteration 203, loss = 0.51548
I1118 02:24:42.510411 22484 solver.cpp:464] Iteration 203, lr = 1e-05
I1118 02:24:44.789531 22484 solver.cpp:209] Iteration 204, loss = 0.200025
I1118 02:24:44.789561 22484 solver.cpp:464] Iteration 204, lr = 1e-05
I1118 02:24:47.067440 22484 solver.cpp:209] Iteration 205, loss = 0.227769
I1118 02:24:47.067482 22484 solver.cpp:464] Iteration 205, lr = 1e-05
I1118 02:24:49.357723 22484 solver.cpp:209] Iteration 206, loss = 0.237812
I1118 02:24:49.357764 22484 solver.cpp:464] Iteration 206, lr = 1e-05
I1118 02:24:51.637451 22484 solver.cpp:209] Iteration 207, loss = 0.455944
I1118 02:24:51.637533 22484 solver.cpp:464] Iteration 207, lr = 1e-05
I1118 02:24:53.911933 22484 solver.cpp:209] Iteration 208, loss = 0.145476
I1118 02:24:53.911973 22484 solver.cpp:464] Iteration 208, lr = 1e-05
I1118 02:24:56.178884 22484 solver.cpp:209] Iteration 209, loss = 0.358876
I1118 02:24:56.178925 22484 solver.cpp:464] Iteration 209, lr = 1e-05
I1118 02:24:58.389773 22484 solver.cpp:209] Iteration 210, loss = 0.342218
I1118 02:24:58.389816 22484 solver.cpp:464] Iteration 210, lr = 1e-05
I1118 02:25:00.585006 22484 solver.cpp:209] Iteration 211, loss = 0.165293
I1118 02:25:00.585036 22484 solver.cpp:464] Iteration 211, lr = 1e-05
I1118 02:25:02.779744 22484 solver.cpp:209] Iteration 212, loss = 0.511138
I1118 02:25:02.779785 22484 solver.cpp:464] Iteration 212, lr = 1e-05
I1118 02:25:04.977965 22484 solver.cpp:209] Iteration 213, loss = 0.405832
I1118 02:25:04.978006 22484 solver.cpp:464] Iteration 213, lr = 1e-05
I1118 02:25:07.178812 22484 solver.cpp:209] Iteration 214, loss = 0.189687
I1118 02:25:07.178843 22484 solver.cpp:464] Iteration 214, lr = 1e-05
I1118 02:25:09.374274 22484 solver.cpp:209] Iteration 215, loss = 0.178511
I1118 02:25:09.374305 22484 solver.cpp:464] Iteration 215, lr = 1e-05
I1118 02:25:11.572669 22484 solver.cpp:209] Iteration 216, loss = 0.295048
I1118 02:25:11.572698 22484 solver.cpp:464] Iteration 216, lr = 1e-05
I1118 02:25:13.796876 22484 solver.cpp:209] Iteration 217, loss = 0.182722
I1118 02:25:13.796917 22484 solver.cpp:464] Iteration 217, lr = 1e-05
I1118 02:25:16.036288 22484 solver.cpp:209] Iteration 218, loss = 0.427757
I1118 02:25:16.036316 22484 solver.cpp:464] Iteration 218, lr = 1e-05
I1118 02:25:18.292662 22484 solver.cpp:209] Iteration 219, loss = 0.574169
I1118 02:25:18.292701 22484 solver.cpp:464] Iteration 219, lr = 1e-05
I1118 02:25:20.540244 22484 solver.cpp:209] Iteration 220, loss = 0.175592
I1118 02:25:20.540273 22484 solver.cpp:464] Iteration 220, lr = 1e-05
I1118 02:25:22.791065 22484 solver.cpp:209] Iteration 221, loss = 0.43268
I1118 02:25:22.791110 22484 solver.cpp:464] Iteration 221, lr = 1e-05
I1118 02:25:25.037122 22484 solver.cpp:209] Iteration 222, loss = 0.117051
I1118 02:25:25.037150 22484 solver.cpp:464] Iteration 222, lr = 1e-05
I1118 02:25:27.280526 22484 solver.cpp:209] Iteration 223, loss = 0.138872
I1118 02:25:27.280555 22484 solver.cpp:464] Iteration 223, lr = 1e-05
I1118 02:25:29.524359 22484 solver.cpp:209] Iteration 224, loss = 0.208292
I1118 02:25:29.524400 22484 solver.cpp:464] Iteration 224, lr = 1e-05
I1118 02:25:31.762943 22484 solver.cpp:209] Iteration 225, loss = 0.248229
I1118 02:25:31.762984 22484 solver.cpp:464] Iteration 225, lr = 1e-05
I1118 02:25:34.008841 22484 solver.cpp:209] Iteration 226, loss = 0.125407
I1118 02:25:34.008882 22484 solver.cpp:464] Iteration 226, lr = 1e-05
I1118 02:25:36.258853 22484 solver.cpp:209] Iteration 227, loss = 0.38008
I1118 02:25:36.258896 22484 solver.cpp:464] Iteration 227, lr = 1e-05
I1118 02:25:38.503752 22484 solver.cpp:209] Iteration 228, loss = 0.295489
I1118 02:25:38.503782 22484 solver.cpp:464] Iteration 228, lr = 1e-05
I1118 02:25:40.766762 22484 solver.cpp:209] Iteration 229, loss = 0.239626
I1118 02:25:40.766788 22484 solver.cpp:464] Iteration 229, lr = 1e-05
I1118 02:25:43.013031 22484 solver.cpp:209] Iteration 230, loss = 0.140913
I1118 02:25:43.013070 22484 solver.cpp:464] Iteration 230, lr = 1e-05
I1118 02:25:45.260660 22484 solver.cpp:209] Iteration 231, loss = 0.18123
I1118 02:25:45.260690 22484 solver.cpp:464] Iteration 231, lr = 1e-05
I1118 02:25:47.501900 22484 solver.cpp:209] Iteration 232, loss = 0.377084
I1118 02:25:47.501929 22484 solver.cpp:464] Iteration 232, lr = 1e-05
I1118 02:25:49.742310 22484 solver.cpp:209] Iteration 233, loss = 0.420192
I1118 02:25:49.742338 22484 solver.cpp:464] Iteration 233, lr = 1e-05
I1118 02:25:51.987213 22484 solver.cpp:209] Iteration 234, loss = 0.10223
I1118 02:25:51.987254 22484 solver.cpp:464] Iteration 234, lr = 1e-05
I1118 02:25:54.240377 22484 solver.cpp:209] Iteration 235, loss = 0.301914
I1118 02:25:54.240464 22484 solver.cpp:464] Iteration 235, lr = 1e-05
I1118 02:25:56.496002 22484 solver.cpp:209] Iteration 236, loss = 0.60345
I1118 02:25:56.496031 22484 solver.cpp:464] Iteration 236, lr = 1e-05
I1118 02:25:58.743180 22484 solver.cpp:209] Iteration 237, loss = 0.174674
I1118 02:25:58.743221 22484 solver.cpp:464] Iteration 237, lr = 1e-05
I1118 02:26:00.988585 22484 solver.cpp:209] Iteration 238, loss = 0.354689
I1118 02:26:00.988612 22484 solver.cpp:464] Iteration 238, lr = 1e-05
I1118 02:26:03.229975 22484 solver.cpp:209] Iteration 239, loss = 0.249445
I1118 02:26:03.230017 22484 solver.cpp:464] Iteration 239, lr = 1e-05
I1118 02:26:05.456048 22484 solver.cpp:209] Iteration 240, loss = 0.288259
I1118 02:26:05.456089 22484 solver.cpp:464] Iteration 240, lr = 1e-05
I1118 02:26:07.687875 22484 solver.cpp:209] Iteration 241, loss = 0.221715
I1118 02:26:07.687917 22484 solver.cpp:464] Iteration 241, lr = 1e-05
I1118 02:26:09.941334 22484 solver.cpp:209] Iteration 242, loss = 0.282399
I1118 02:26:09.941375 22484 solver.cpp:464] Iteration 242, lr = 1e-05
I1118 02:26:12.169839 22484 solver.cpp:209] Iteration 243, loss = 0.396037
I1118 02:26:12.169880 22484 solver.cpp:464] Iteration 243, lr = 1e-05
I1118 02:26:14.409965 22484 solver.cpp:209] Iteration 244, loss = 0.694914
I1118 02:26:14.409993 22484 solver.cpp:464] Iteration 244, lr = 1e-05
I1118 02:26:16.637025 22484 solver.cpp:209] Iteration 245, loss = 0.369316
I1118 02:26:16.637054 22484 solver.cpp:464] Iteration 245, lr = 1e-05
I1118 02:26:18.867930 22484 solver.cpp:209] Iteration 246, loss = 0.18715
I1118 02:26:18.867971 22484 solver.cpp:464] Iteration 246, lr = 1e-05
I1118 02:26:21.089875 22484 solver.cpp:209] Iteration 247, loss = 0.203218
I1118 02:26:21.089913 22484 solver.cpp:464] Iteration 247, lr = 1e-05
I1118 02:26:23.321481 22484 solver.cpp:209] Iteration 248, loss = 0.42298
I1118 02:26:23.321522 22484 solver.cpp:464] Iteration 248, lr = 1e-05
I1118 02:26:25.555068 22484 solver.cpp:209] Iteration 249, loss = 0.120007
I1118 02:26:25.555111 22484 solver.cpp:464] Iteration 249, lr = 1e-05
I1118 02:26:25.555706 22484 solver.cpp:264] Iteration 250, Testing net (#0)
I1118 02:26:39.435003 22484 solver.cpp:305] Test loss: 0.291168
I1118 02:26:39.435045 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:26:39.435053 22484 solver.cpp:319]            = 0.994197
I1118 02:26:39.435057 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:26:39.435061 22484 solver.cpp:318] mean_score = test_score[2] { = 8.46356e-37} / test_score[3] { = 59 }
I1118 02:26:39.435067 22484 solver.cpp:319]            = 1.4345e-38
I1118 02:26:39.435071 22484 solver.cpp:328]     Test net output #1: accuracy = 1.4345e-38
I1118 02:26:39.435075 22484 solver.cpp:332]     Test net output #2: accuracy = 0.892361
I1118 02:26:39.435080 22484 solver.cpp:334]     Test net output #3: accuracy = 0.497099
I1118 02:26:40.079885 22484 solver.cpp:209] Iteration 250, loss = 0.301802
I1118 02:26:40.079926 22484 solver.cpp:464] Iteration 250, lr = 1e-05
I1118 02:26:42.325654 22484 solver.cpp:209] Iteration 251, loss = 0.246108
I1118 02:26:42.325696 22484 solver.cpp:464] Iteration 251, lr = 1e-05
I1118 02:26:44.575592 22484 solver.cpp:209] Iteration 252, loss = 0.184101
I1118 02:26:44.575634 22484 solver.cpp:464] Iteration 252, lr = 1e-05
I1118 02:26:46.849653 22484 solver.cpp:209] Iteration 253, loss = 0.340302
I1118 02:26:46.849681 22484 solver.cpp:464] Iteration 253, lr = 1e-05
I1118 02:26:49.097141 22484 solver.cpp:209] Iteration 254, loss = 0.447671
I1118 02:26:49.097182 22484 solver.cpp:464] Iteration 254, lr = 1e-05
I1118 02:26:51.340559 22484 solver.cpp:209] Iteration 255, loss = 0.477429
I1118 02:26:51.340587 22484 solver.cpp:464] Iteration 255, lr = 1e-05
I1118 02:26:53.579048 22484 solver.cpp:209] Iteration 256, loss = 0.346182
I1118 02:26:53.579077 22484 solver.cpp:464] Iteration 256, lr = 1e-05
I1118 02:26:55.820415 22484 solver.cpp:209] Iteration 257, loss = 0.187602
I1118 02:26:55.820472 22484 solver.cpp:464] Iteration 257, lr = 1e-05
I1118 02:26:58.063240 22484 solver.cpp:209] Iteration 258, loss = 0.344049
I1118 02:26:58.063268 22484 solver.cpp:464] Iteration 258, lr = 1e-05
I1118 02:27:00.309502 22484 solver.cpp:209] Iteration 259, loss = 0.288345
I1118 02:27:00.309532 22484 solver.cpp:464] Iteration 259, lr = 1e-05
I1118 02:27:02.549558 22484 solver.cpp:209] Iteration 260, loss = 0.194954
I1118 02:27:02.549599 22484 solver.cpp:464] Iteration 260, lr = 1e-05
I1118 02:27:04.794544 22484 solver.cpp:209] Iteration 261, loss = 0.242345
I1118 02:27:04.794595 22484 solver.cpp:464] Iteration 261, lr = 1e-05
I1118 02:27:07.030985 22484 solver.cpp:209] Iteration 262, loss = 0.267193
I1118 02:27:07.031013 22484 solver.cpp:464] Iteration 262, lr = 1e-05
I1118 02:27:09.271320 22484 solver.cpp:209] Iteration 263, loss = 0.224859
I1118 02:27:09.271361 22484 solver.cpp:464] Iteration 263, lr = 1e-05
I1118 02:27:11.514569 22484 solver.cpp:209] Iteration 264, loss = 0.375025
I1118 02:27:11.514619 22484 solver.cpp:464] Iteration 264, lr = 1e-05
I1118 02:27:13.755357 22484 solver.cpp:209] Iteration 265, loss = 0.253626
I1118 02:27:13.755385 22484 solver.cpp:464] Iteration 265, lr = 1e-05
I1118 02:27:16.006176 22484 solver.cpp:209] Iteration 266, loss = 0.325896
I1118 02:27:16.006206 22484 solver.cpp:464] Iteration 266, lr = 1e-05
I1118 02:27:18.250638 22484 solver.cpp:209] Iteration 267, loss = 0.236156
I1118 02:27:18.250665 22484 solver.cpp:464] Iteration 267, lr = 1e-05
I1118 02:27:20.493818 22484 solver.cpp:209] Iteration 268, loss = 0.7019
I1118 02:27:20.493859 22484 solver.cpp:464] Iteration 268, lr = 1e-05
I1118 02:27:22.722712 22484 solver.cpp:209] Iteration 269, loss = 0.400385
I1118 02:27:22.722740 22484 solver.cpp:464] Iteration 269, lr = 1e-05
I1118 02:27:24.948811 22484 solver.cpp:209] Iteration 270, loss = 0.678102
I1118 02:27:24.948839 22484 solver.cpp:464] Iteration 270, lr = 1e-05
I1118 02:27:27.170279 22484 solver.cpp:209] Iteration 271, loss = 0.293281
I1118 02:27:27.170336 22484 solver.cpp:464] Iteration 271, lr = 1e-05
I1118 02:27:29.394295 22484 solver.cpp:209] Iteration 272, loss = 0.449142
I1118 02:27:29.394336 22484 solver.cpp:464] Iteration 272, lr = 1e-05
I1118 02:27:31.625458 22484 solver.cpp:209] Iteration 273, loss = 0.384878
I1118 02:27:31.625500 22484 solver.cpp:464] Iteration 273, lr = 1e-05
I1118 02:27:33.860903 22484 solver.cpp:209] Iteration 274, loss = 0.356188
I1118 02:27:33.860944 22484 solver.cpp:464] Iteration 274, lr = 1e-05
I1118 02:27:36.097972 22484 solver.cpp:209] Iteration 275, loss = 0.385555
I1118 02:27:36.098001 22484 solver.cpp:464] Iteration 275, lr = 1e-05
I1118 02:27:38.326515 22484 solver.cpp:209] Iteration 276, loss = 0.626058
I1118 02:27:38.326555 22484 solver.cpp:464] Iteration 276, lr = 1e-05
I1118 02:27:40.547585 22484 solver.cpp:209] Iteration 277, loss = 0.386183
I1118 02:27:40.547613 22484 solver.cpp:464] Iteration 277, lr = 1e-05
I1118 02:27:42.769495 22484 solver.cpp:209] Iteration 278, loss = 0.367589
I1118 02:27:42.769523 22484 solver.cpp:464] Iteration 278, lr = 1e-05
I1118 02:27:44.998214 22484 solver.cpp:209] Iteration 279, loss = 0.393845
I1118 02:27:44.998241 22484 solver.cpp:464] Iteration 279, lr = 1e-05
I1118 02:27:47.235398 22484 solver.cpp:209] Iteration 280, loss = 0.474629
I1118 02:27:47.235437 22484 solver.cpp:464] Iteration 280, lr = 1e-05
I1118 02:27:49.465927 22484 solver.cpp:209] Iteration 281, loss = 0.217837
I1118 02:27:49.465970 22484 solver.cpp:464] Iteration 281, lr = 1e-05
I1118 02:27:51.695704 22484 solver.cpp:209] Iteration 282, loss = 0.39551
I1118 02:27:51.695746 22484 solver.cpp:464] Iteration 282, lr = 1e-05
I1118 02:27:53.921422 22484 solver.cpp:209] Iteration 283, loss = 0.131381
I1118 02:27:53.921452 22484 solver.cpp:464] Iteration 283, lr = 1e-05
I1118 02:27:56.145051 22484 solver.cpp:209] Iteration 284, loss = 0.749028
I1118 02:27:56.145092 22484 solver.cpp:464] Iteration 284, lr = 1e-05
I1118 02:27:58.386128 22484 solver.cpp:209] Iteration 285, loss = 0.248842
I1118 02:27:58.386214 22484 solver.cpp:464] Iteration 285, lr = 1e-05
I1118 02:28:00.608021 22484 solver.cpp:209] Iteration 286, loss = 0.375915
I1118 02:28:00.608050 22484 solver.cpp:464] Iteration 286, lr = 1e-05
I1118 02:28:02.841109 22484 solver.cpp:209] Iteration 287, loss = 0.220032
I1118 02:28:02.841150 22484 solver.cpp:464] Iteration 287, lr = 1e-05
I1118 02:28:05.071872 22484 solver.cpp:209] Iteration 288, loss = 0.435245
I1118 02:28:05.071914 22484 solver.cpp:464] Iteration 288, lr = 1e-05
I1118 02:28:07.302438 22484 solver.cpp:209] Iteration 289, loss = 0.222237
I1118 02:28:07.302466 22484 solver.cpp:464] Iteration 289, lr = 1e-05
I1118 02:28:09.533346 22484 solver.cpp:209] Iteration 290, loss = 0.450946
I1118 02:28:09.533386 22484 solver.cpp:464] Iteration 290, lr = 1e-05
I1118 02:28:11.755630 22484 solver.cpp:209] Iteration 291, loss = 0.576071
I1118 02:28:11.755659 22484 solver.cpp:464] Iteration 291, lr = 1e-05
I1118 02:28:13.978019 22484 solver.cpp:209] Iteration 292, loss = 0.540953
I1118 02:28:13.978049 22484 solver.cpp:464] Iteration 292, lr = 1e-05
I1118 02:28:16.205377 22484 solver.cpp:209] Iteration 293, loss = 0.175012
I1118 02:28:16.205406 22484 solver.cpp:464] Iteration 293, lr = 1e-05
I1118 02:28:18.442487 22484 solver.cpp:209] Iteration 294, loss = 0.14494
I1118 02:28:18.442515 22484 solver.cpp:464] Iteration 294, lr = 1e-05
I1118 02:28:20.676826 22484 solver.cpp:209] Iteration 295, loss = 0.117513
I1118 02:28:20.676867 22484 solver.cpp:464] Iteration 295, lr = 1e-05
I1118 02:28:22.907392 22484 solver.cpp:209] Iteration 296, loss = 0.567134
I1118 02:28:22.907433 22484 solver.cpp:464] Iteration 296, lr = 1e-05
I1118 02:28:25.129447 22484 solver.cpp:209] Iteration 297, loss = 0.217004
I1118 02:28:25.129475 22484 solver.cpp:464] Iteration 297, lr = 1e-05
I1118 02:28:27.355715 22484 solver.cpp:209] Iteration 298, loss = 0.409105
I1118 02:28:27.355743 22484 solver.cpp:464] Iteration 298, lr = 1e-05
I1118 02:28:29.578517 22484 solver.cpp:209] Iteration 299, loss = 0.378387
I1118 02:28:29.578611 22484 solver.cpp:464] Iteration 299, lr = 1e-05
I1118 02:28:29.579241 22484 solver.cpp:264] Iteration 300, Testing net (#0)
I1118 02:28:43.527340 22484 solver.cpp:305] Test loss: 0.29179
I1118 02:28:43.527369 22484 solver.cpp:318] mean_score = test_score[0] { = 513} / test_score[1] { = 517 }
I1118 02:28:43.527389 22484 solver.cpp:319]            = 0.992263
I1118 02:28:43.527392 22484 solver.cpp:328]     Test net output #0: accuracy = 0.992263
I1118 02:28:43.527397 22484 solver.cpp:318] mean_score = test_score[2] { = 1} / test_score[3] { = 59 }
I1118 02:28:43.527402 22484 solver.cpp:319]            = 0.0169492
I1118 02:28:43.527405 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0169492
I1118 02:28:43.527410 22484 solver.cpp:332]     Test net output #2: accuracy = 0.892361
I1118 02:28:43.527415 22484 solver.cpp:334]     Test net output #3: accuracy = 0.504606
I1118 02:28:44.181234 22484 solver.cpp:209] Iteration 300, loss = 0.308045
I1118 02:28:44.181262 22484 solver.cpp:464] Iteration 300, lr = 1e-05
I1118 02:28:46.424860 22484 solver.cpp:209] Iteration 301, loss = 0.271855
I1118 02:28:46.424890 22484 solver.cpp:464] Iteration 301, lr = 1e-05
I1118 02:28:48.664422 22484 solver.cpp:209] Iteration 302, loss = 0.354377
I1118 02:28:48.664450 22484 solver.cpp:464] Iteration 302, lr = 1e-05
I1118 02:28:50.923122 22484 solver.cpp:209] Iteration 303, loss = 0.392479
I1118 02:28:50.923147 22484 solver.cpp:464] Iteration 303, lr = 1e-05
I1118 02:28:53.173045 22484 solver.cpp:209] Iteration 304, loss = 0.31185
I1118 02:28:53.173085 22484 solver.cpp:464] Iteration 304, lr = 1e-05
I1118 02:28:55.422260 22484 solver.cpp:209] Iteration 305, loss = 0.180684
I1118 02:28:55.422288 22484 solver.cpp:464] Iteration 305, lr = 1e-05
I1118 02:28:57.668711 22484 solver.cpp:209] Iteration 306, loss = 0.316489
I1118 02:28:57.668751 22484 solver.cpp:464] Iteration 306, lr = 1e-05
I1118 02:28:59.901871 22484 solver.cpp:209] Iteration 307, loss = 0.138284
I1118 02:28:59.901919 22484 solver.cpp:464] Iteration 307, lr = 1e-05
I1118 02:29:02.149229 22484 solver.cpp:209] Iteration 308, loss = 0.168766
I1118 02:29:02.149256 22484 solver.cpp:464] Iteration 308, lr = 1e-05
I1118 02:29:04.389816 22484 solver.cpp:209] Iteration 309, loss = 0.220392
I1118 02:29:04.389844 22484 solver.cpp:464] Iteration 309, lr = 1e-05
I1118 02:29:06.638902 22484 solver.cpp:209] Iteration 310, loss = 0.421727
I1118 02:29:06.638932 22484 solver.cpp:464] Iteration 310, lr = 1e-05
I1118 02:29:08.879701 22484 solver.cpp:209] Iteration 311, loss = 0.213179
I1118 02:29:08.879729 22484 solver.cpp:464] Iteration 311, lr = 1e-05
I1118 02:29:11.104351 22484 solver.cpp:209] Iteration 312, loss = 0.346152
I1118 02:29:11.104389 22484 solver.cpp:464] Iteration 312, lr = 1e-05
I1118 02:29:13.336475 22484 solver.cpp:209] Iteration 313, loss = 0.273025
I1118 02:29:13.336516 22484 solver.cpp:464] Iteration 313, lr = 1e-05
I1118 02:29:15.564255 22484 solver.cpp:209] Iteration 314, loss = 0.242506
I1118 02:29:15.564297 22484 solver.cpp:464] Iteration 314, lr = 1e-05
I1118 02:29:17.789053 22484 solver.cpp:209] Iteration 315, loss = 0.337365
I1118 02:29:17.789094 22484 solver.cpp:464] Iteration 315, lr = 1e-05
I1118 02:29:20.017328 22484 solver.cpp:209] Iteration 316, loss = 0.4545
I1118 02:29:20.017369 22484 solver.cpp:464] Iteration 316, lr = 1e-05
I1118 02:29:22.239337 22484 solver.cpp:209] Iteration 317, loss = 0.156893
I1118 02:29:22.239367 22484 solver.cpp:464] Iteration 317, lr = 1e-05
I1118 02:29:24.472123 22484 solver.cpp:209] Iteration 318, loss = 0.172729
I1118 02:29:24.472163 22484 solver.cpp:464] Iteration 318, lr = 1e-05
I1118 02:29:26.707273 22484 solver.cpp:209] Iteration 319, loss = 0.283759
I1118 02:29:26.707314 22484 solver.cpp:464] Iteration 319, lr = 1e-05
I1118 02:29:28.934011 22484 solver.cpp:209] Iteration 320, loss = 0.410631
I1118 02:29:28.934039 22484 solver.cpp:464] Iteration 320, lr = 1e-05
I1118 02:29:31.159512 22484 solver.cpp:209] Iteration 321, loss = 0.274025
I1118 02:29:31.159550 22484 solver.cpp:464] Iteration 321, lr = 1e-05
I1118 02:29:33.376754 22484 solver.cpp:209] Iteration 322, loss = 0.538308
I1118 02:29:33.376781 22484 solver.cpp:464] Iteration 322, lr = 1e-05
I1118 02:29:35.605403 22484 solver.cpp:209] Iteration 323, loss = 0.183186
I1118 02:29:35.605432 22484 solver.cpp:464] Iteration 323, lr = 1e-05
I1118 02:29:37.857321 22484 solver.cpp:209] Iteration 324, loss = 0.489152
I1118 02:29:37.857362 22484 solver.cpp:464] Iteration 324, lr = 1e-05
I1118 02:29:40.086669 22484 solver.cpp:209] Iteration 325, loss = 0.0663229
I1118 02:29:40.086699 22484 solver.cpp:464] Iteration 325, lr = 1e-05
I1118 02:29:42.319545 22484 solver.cpp:209] Iteration 326, loss = 0.15788
I1118 02:29:42.319587 22484 solver.cpp:464] Iteration 326, lr = 1e-05
I1118 02:29:44.541538 22484 solver.cpp:209] Iteration 327, loss = 0.240283
I1118 02:29:44.541568 22484 solver.cpp:464] Iteration 327, lr = 1e-05
I1118 02:29:46.767827 22484 solver.cpp:209] Iteration 328, loss = 0.288281
I1118 02:29:46.767868 22484 solver.cpp:464] Iteration 328, lr = 1e-05
I1118 02:29:48.991169 22484 solver.cpp:209] Iteration 329, loss = 0.230677
I1118 02:29:48.991210 22484 solver.cpp:464] Iteration 329, lr = 1e-05
I1118 02:29:51.213502 22484 solver.cpp:209] Iteration 330, loss = 0.270605
I1118 02:29:51.213531 22484 solver.cpp:464] Iteration 330, lr = 1e-05
I1118 02:29:53.454447 22484 solver.cpp:209] Iteration 331, loss = 0.225214
I1118 02:29:53.454475 22484 solver.cpp:464] Iteration 331, lr = 1e-05
I1118 02:29:55.685195 22484 solver.cpp:209] Iteration 332, loss = 0.279956
I1118 02:29:55.685235 22484 solver.cpp:464] Iteration 332, lr = 1e-05
I1118 02:29:57.915642 22484 solver.cpp:209] Iteration 333, loss = 0.147377
I1118 02:29:57.915671 22484 solver.cpp:464] Iteration 333, lr = 1e-05
I1118 02:30:00.144156 22484 solver.cpp:209] Iteration 334, loss = 0.180659
I1118 02:30:00.144198 22484 solver.cpp:464] Iteration 334, lr = 1e-05
I1118 02:30:02.364835 22484 solver.cpp:209] Iteration 335, loss = 0.43056
I1118 02:30:02.364917 22484 solver.cpp:464] Iteration 335, lr = 1e-05
I1118 02:30:04.595551 22484 solver.cpp:209] Iteration 336, loss = 0.42438
I1118 02:30:04.595593 22484 solver.cpp:464] Iteration 336, lr = 1e-05
I1118 02:30:06.822444 22484 solver.cpp:209] Iteration 337, loss = 0.0669043
I1118 02:30:06.822473 22484 solver.cpp:464] Iteration 337, lr = 1e-05
I1118 02:30:09.054235 22484 solver.cpp:209] Iteration 338, loss = 0.440684
I1118 02:30:09.054262 22484 solver.cpp:464] Iteration 338, lr = 1e-05
I1118 02:30:11.290485 22484 solver.cpp:209] Iteration 339, loss = 0.575568
I1118 02:30:11.290516 22484 solver.cpp:464] Iteration 339, lr = 1e-05
I1118 02:30:13.520544 22484 solver.cpp:209] Iteration 340, loss = 0.197338
I1118 02:30:13.520575 22484 solver.cpp:464] Iteration 340, lr = 1e-05
I1118 02:30:15.749465 22484 solver.cpp:209] Iteration 341, loss = 0.318931
I1118 02:30:15.749497 22484 solver.cpp:464] Iteration 341, lr = 1e-05
I1118 02:30:17.972554 22484 solver.cpp:209] Iteration 342, loss = 0.299724
I1118 02:30:17.972589 22484 solver.cpp:464] Iteration 342, lr = 1e-05
I1118 02:30:20.196377 22484 solver.cpp:209] Iteration 343, loss = 0.321527
I1118 02:30:20.196406 22484 solver.cpp:464] Iteration 343, lr = 1e-05
I1118 02:30:22.426252 22484 solver.cpp:209] Iteration 344, loss = 0.139274
I1118 02:30:22.426280 22484 solver.cpp:464] Iteration 344, lr = 1e-05
I1118 02:30:24.662996 22484 solver.cpp:209] Iteration 345, loss = 0.423959
I1118 02:30:24.663036 22484 solver.cpp:464] Iteration 345, lr = 1e-05
I1118 02:30:26.898942 22484 solver.cpp:209] Iteration 346, loss = 0.239139
I1118 02:30:26.898972 22484 solver.cpp:464] Iteration 346, lr = 1e-05
I1118 02:30:29.136425 22484 solver.cpp:209] Iteration 347, loss = 0.749293
I1118 02:30:29.136453 22484 solver.cpp:464] Iteration 347, lr = 1e-05
I1118 02:30:31.359602 22484 solver.cpp:209] Iteration 348, loss = 0.333553
I1118 02:30:31.359632 22484 solver.cpp:464] Iteration 348, lr = 1e-05
I1118 02:30:33.587332 22484 solver.cpp:209] Iteration 349, loss = 0.0876178
I1118 02:30:33.587376 22484 solver.cpp:464] Iteration 349, lr = 1e-05
I1118 02:30:33.587992 22484 solver.cpp:264] Iteration 350, Testing net (#0)
I1118 02:30:47.460196 22484 solver.cpp:305] Test loss: 0.294368
I1118 02:30:47.460237 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:30:47.460244 22484 solver.cpp:319]            = 0.994197
I1118 02:30:47.460249 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:30:47.460253 22484 solver.cpp:318] mean_score = test_score[2] { = 1} / test_score[3] { = 59 }
I1118 02:30:47.460258 22484 solver.cpp:319]            = 0.0169492
I1118 02:30:47.460263 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0169492
I1118 02:30:47.460266 22484 solver.cpp:332]     Test net output #2: accuracy = 0.894097
I1118 02:30:47.460270 22484 solver.cpp:334]     Test net output #3: accuracy = 0.505573
I1118 02:30:48.106848 22484 solver.cpp:209] Iteration 350, loss = 0.355022
I1118 02:30:48.106875 22484 solver.cpp:464] Iteration 350, lr = 1e-05
I1118 02:30:50.347875 22484 solver.cpp:209] Iteration 351, loss = 0.272233
I1118 02:30:50.347918 22484 solver.cpp:464] Iteration 351, lr = 1e-05
I1118 02:30:52.589108 22484 solver.cpp:209] Iteration 352, loss = 0.281102
I1118 02:30:52.589136 22484 solver.cpp:464] Iteration 352, lr = 1e-05
I1118 02:30:54.833858 22484 solver.cpp:209] Iteration 353, loss = 0.172216
I1118 02:30:54.833900 22484 solver.cpp:464] Iteration 353, lr = 1e-05
I1118 02:30:57.078459 22484 solver.cpp:209] Iteration 354, loss = 0.24176
I1118 02:30:57.078487 22484 solver.cpp:464] Iteration 354, lr = 1e-05
I1118 02:30:59.327298 22484 solver.cpp:209] Iteration 355, loss = 0.132694
I1118 02:30:59.327327 22484 solver.cpp:464] Iteration 355, lr = 1e-05
I1118 02:31:01.594951 22484 solver.cpp:209] Iteration 356, loss = 0.332859
I1118 02:31:01.594980 22484 solver.cpp:464] Iteration 356, lr = 1e-05
I1118 02:31:03.841846 22484 solver.cpp:209] Iteration 357, loss = 0.634826
I1118 02:31:03.841895 22484 solver.cpp:464] Iteration 357, lr = 1e-05
I1118 02:31:06.080523 22484 solver.cpp:209] Iteration 358, loss = 0.299093
I1118 02:31:06.080553 22484 solver.cpp:464] Iteration 358, lr = 1e-05
I1118 02:31:08.317663 22484 solver.cpp:209] Iteration 359, loss = 0.423044
I1118 02:31:08.317692 22484 solver.cpp:464] Iteration 359, lr = 1e-05
I1118 02:31:10.559525 22484 solver.cpp:209] Iteration 360, loss = 0.292807
I1118 02:31:10.559566 22484 solver.cpp:464] Iteration 360, lr = 1e-05
I1118 02:31:12.807783 22484 solver.cpp:209] Iteration 361, loss = 0.294395
I1118 02:31:12.807823 22484 solver.cpp:464] Iteration 361, lr = 1e-05
I1118 02:31:15.054741 22484 solver.cpp:209] Iteration 362, loss = 0.176638
I1118 02:31:15.054771 22484 solver.cpp:464] Iteration 362, lr = 1e-05
I1118 02:31:17.278471 22484 solver.cpp:209] Iteration 363, loss = 0.199019
I1118 02:31:17.278501 22484 solver.cpp:464] Iteration 363, lr = 1e-05
I1118 02:31:19.503844 22484 solver.cpp:209] Iteration 364, loss = 0.440572
I1118 02:31:19.503873 22484 solver.cpp:464] Iteration 364, lr = 1e-05
I1118 02:31:21.724074 22484 solver.cpp:209] Iteration 365, loss = 0.226387
I1118 02:31:21.724117 22484 solver.cpp:464] Iteration 365, lr = 1e-05
I1118 02:31:23.916056 22484 solver.cpp:209] Iteration 366, loss = 0.362655
I1118 02:31:23.916085 22484 solver.cpp:464] Iteration 366, lr = 1e-05
I1118 02:31:26.108010 22484 solver.cpp:209] Iteration 367, loss = 0.193207
I1118 02:31:26.108037 22484 solver.cpp:464] Iteration 367, lr = 1e-05
I1118 02:31:28.316295 22484 solver.cpp:209] Iteration 368, loss = 0.289738
I1118 02:31:28.316323 22484 solver.cpp:464] Iteration 368, lr = 1e-05
I1118 02:31:30.509708 22484 solver.cpp:209] Iteration 369, loss = 0.324153
I1118 02:31:30.509748 22484 solver.cpp:464] Iteration 369, lr = 1e-05
I1118 02:31:32.705330 22484 solver.cpp:209] Iteration 370, loss = 0.628323
I1118 02:31:32.705371 22484 solver.cpp:464] Iteration 370, lr = 1e-05
I1118 02:31:34.894249 22484 solver.cpp:209] Iteration 371, loss = 0.587959
I1118 02:31:34.894309 22484 solver.cpp:464] Iteration 371, lr = 1e-05
I1118 02:31:37.099627 22484 solver.cpp:209] Iteration 372, loss = 0.691526
I1118 02:31:37.099656 22484 solver.cpp:464] Iteration 372, lr = 1e-05
I1118 02:31:39.322429 22484 solver.cpp:209] Iteration 373, loss = 0.194852
I1118 02:31:39.322458 22484 solver.cpp:464] Iteration 373, lr = 1e-05
I1118 02:31:41.552186 22484 solver.cpp:209] Iteration 374, loss = 0.274544
I1118 02:31:41.552214 22484 solver.cpp:464] Iteration 374, lr = 1e-05
I1118 02:31:43.780197 22484 solver.cpp:209] Iteration 375, loss = 0.458209
I1118 02:31:43.780238 22484 solver.cpp:464] Iteration 375, lr = 1e-05
I1118 02:31:46.005844 22484 solver.cpp:209] Iteration 376, loss = 0.520883
I1118 02:31:46.005874 22484 solver.cpp:464] Iteration 376, lr = 1e-05
I1118 02:31:48.231839 22484 solver.cpp:209] Iteration 377, loss = 0.402082
I1118 02:31:48.231879 22484 solver.cpp:464] Iteration 377, lr = 1e-05
I1118 02:31:50.454210 22484 solver.cpp:209] Iteration 378, loss = 0.396475
I1118 02:31:50.454239 22484 solver.cpp:464] Iteration 378, lr = 1e-05
I1118 02:31:52.677268 22484 solver.cpp:209] Iteration 379, loss = 0.553608
I1118 02:31:52.677309 22484 solver.cpp:464] Iteration 379, lr = 1e-05
I1118 02:31:54.905968 22484 solver.cpp:209] Iteration 380, loss = 0.572923
I1118 02:31:54.905998 22484 solver.cpp:464] Iteration 380, lr = 1e-05
I1118 02:31:57.136246 22484 solver.cpp:209] Iteration 381, loss = 0.306231
I1118 02:31:57.136276 22484 solver.cpp:464] Iteration 381, lr = 1e-05
I1118 02:31:59.370173 22484 solver.cpp:209] Iteration 382, loss = 0.369635
I1118 02:31:59.370203 22484 solver.cpp:464] Iteration 382, lr = 1e-05
I1118 02:32:01.591343 22484 solver.cpp:209] Iteration 383, loss = 0.251114
I1118 02:32:01.591373 22484 solver.cpp:464] Iteration 383, lr = 1e-05
I1118 02:32:03.814467 22484 solver.cpp:209] Iteration 384, loss = 0.29424
I1118 02:32:03.814497 22484 solver.cpp:464] Iteration 384, lr = 1e-05
I1118 02:32:06.036927 22484 solver.cpp:209] Iteration 385, loss = 0.266733
I1118 02:32:06.037014 22484 solver.cpp:464] Iteration 385, lr = 1e-05
I1118 02:32:08.261385 22484 solver.cpp:209] Iteration 386, loss = 0.262831
I1118 02:32:08.261415 22484 solver.cpp:464] Iteration 386, lr = 1e-05
I1118 02:32:10.491627 22484 solver.cpp:209] Iteration 387, loss = 0.586076
I1118 02:32:10.491654 22484 solver.cpp:464] Iteration 387, lr = 1e-05
I1118 02:32:12.717531 22484 solver.cpp:209] Iteration 388, loss = 0.168192
I1118 02:32:12.717558 22484 solver.cpp:464] Iteration 388, lr = 1e-05
I1118 02:32:14.944228 22484 solver.cpp:209] Iteration 389, loss = 0.386044
I1118 02:32:14.944257 22484 solver.cpp:464] Iteration 389, lr = 1e-05
I1118 02:32:17.167147 22484 solver.cpp:209] Iteration 390, loss = 0.122217
I1118 02:32:17.167176 22484 solver.cpp:464] Iteration 390, lr = 1e-05
I1118 02:32:19.391861 22484 solver.cpp:209] Iteration 391, loss = 0.398419
I1118 02:32:19.391901 22484 solver.cpp:464] Iteration 391, lr = 1e-05
I1118 02:32:21.619935 22484 solver.cpp:209] Iteration 392, loss = 0.382328
I1118 02:32:21.619963 22484 solver.cpp:464] Iteration 392, lr = 1e-05
I1118 02:32:23.838021 22484 solver.cpp:209] Iteration 393, loss = 0.504732
I1118 02:32:23.838063 22484 solver.cpp:464] Iteration 393, lr = 1e-05
I1118 02:32:26.068645 22484 solver.cpp:209] Iteration 394, loss = 0.472987
I1118 02:32:26.068673 22484 solver.cpp:464] Iteration 394, lr = 1e-05
I1118 02:32:28.315474 22484 solver.cpp:209] Iteration 395, loss = 0.44645
I1118 02:32:28.315503 22484 solver.cpp:464] Iteration 395, lr = 1e-05
I1118 02:32:30.545167 22484 solver.cpp:209] Iteration 396, loss = 0.104655
I1118 02:32:30.545208 22484 solver.cpp:464] Iteration 396, lr = 1e-05
I1118 02:32:32.772707 22484 solver.cpp:209] Iteration 397, loss = 0.111586
I1118 02:32:32.772738 22484 solver.cpp:464] Iteration 397, lr = 1e-05
I1118 02:32:34.992730 22484 solver.cpp:209] Iteration 398, loss = 0.28874
I1118 02:32:34.992758 22484 solver.cpp:464] Iteration 398, lr = 1e-05
I1118 02:32:37.219867 22484 solver.cpp:209] Iteration 399, loss = 0.329631
I1118 02:32:37.219928 22484 solver.cpp:464] Iteration 399, lr = 1e-05
I1118 02:32:37.220510 22484 solver.cpp:264] Iteration 400, Testing net (#0)
I1118 02:32:51.102838 22484 solver.cpp:305] Test loss: 0.28421
I1118 02:32:51.102867 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:32:51.102874 22484 solver.cpp:319]            = 0.994197
I1118 02:32:51.102880 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:32:51.102884 22484 solver.cpp:318] mean_score = test_score[2] { = 3} / test_score[3] { = 59 }
I1118 02:32:51.102888 22484 solver.cpp:319]            = 0.0508475
I1118 02:32:51.102892 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0508475
I1118 02:32:51.102897 22484 solver.cpp:332]     Test net output #2: accuracy = 0.897569
I1118 02:32:51.102903 22484 solver.cpp:334]     Test net output #3: accuracy = 0.522522
I1118 02:32:51.750105 22484 solver.cpp:209] Iteration 400, loss = 0.24109
I1118 02:32:51.750134 22484 solver.cpp:464] Iteration 400, lr = 1e-05
I1118 02:32:53.977840 22484 solver.cpp:209] Iteration 401, loss = 0.635284
I1118 02:32:53.977869 22484 solver.cpp:464] Iteration 401, lr = 1e-05
I1118 02:32:56.203328 22484 solver.cpp:209] Iteration 402, loss = 0.0824944
I1118 02:32:56.203371 22484 solver.cpp:464] Iteration 402, lr = 1e-05
I1118 02:32:58.435098 22484 solver.cpp:209] Iteration 403, loss = 0.322886
I1118 02:32:58.435138 22484 solver.cpp:464] Iteration 403, lr = 1e-05
I1118 02:33:00.661258 22484 solver.cpp:209] Iteration 404, loss = 0.275901
I1118 02:33:00.661288 22484 solver.cpp:464] Iteration 404, lr = 1e-05
I1118 02:33:02.896040 22484 solver.cpp:209] Iteration 405, loss = 0.326492
I1118 02:33:02.896081 22484 solver.cpp:464] Iteration 405, lr = 1e-05
I1118 02:33:05.124019 22484 solver.cpp:209] Iteration 406, loss = 0.398211
I1118 02:33:05.124047 22484 solver.cpp:464] Iteration 406, lr = 1e-05
I1118 02:33:07.352717 22484 solver.cpp:209] Iteration 407, loss = 0.287481
I1118 02:33:07.352802 22484 solver.cpp:464] Iteration 407, lr = 1e-05
I1118 02:33:09.573715 22484 solver.cpp:209] Iteration 408, loss = 0.308023
I1118 02:33:09.573745 22484 solver.cpp:464] Iteration 408, lr = 1e-05
I1118 02:33:11.803057 22484 solver.cpp:209] Iteration 409, loss = 0.253888
I1118 02:33:11.803086 22484 solver.cpp:464] Iteration 409, lr = 1e-05
I1118 02:33:14.036638 22484 solver.cpp:209] Iteration 410, loss = 0.163428
I1118 02:33:14.036669 22484 solver.cpp:464] Iteration 410, lr = 1e-05
I1118 02:33:16.273883 22484 solver.cpp:209] Iteration 411, loss = 0.297365
I1118 02:33:16.273926 22484 solver.cpp:464] Iteration 411, lr = 1e-05
I1118 02:33:18.505444 22484 solver.cpp:209] Iteration 412, loss = 0.189636
I1118 02:33:18.505484 22484 solver.cpp:464] Iteration 412, lr = 1e-05
I1118 02:33:20.729226 22484 solver.cpp:209] Iteration 413, loss = 0.222039
I1118 02:33:20.729267 22484 solver.cpp:464] Iteration 413, lr = 1e-05
I1118 02:33:22.951872 22484 solver.cpp:209] Iteration 414, loss = 0.409937
I1118 02:33:22.951901 22484 solver.cpp:464] Iteration 414, lr = 1e-05
I1118 02:33:25.181398 22484 solver.cpp:209] Iteration 415, loss = 0.221554
I1118 02:33:25.181427 22484 solver.cpp:464] Iteration 415, lr = 1e-05
I1118 02:33:27.417486 22484 solver.cpp:209] Iteration 416, loss = 0.231411
I1118 02:33:27.417515 22484 solver.cpp:464] Iteration 416, lr = 1e-05
I1118 02:33:29.653775 22484 solver.cpp:209] Iteration 417, loss = 0.198026
I1118 02:33:29.653803 22484 solver.cpp:464] Iteration 417, lr = 1e-05
I1118 02:33:31.881606 22484 solver.cpp:209] Iteration 418, loss = 0.278006
I1118 02:33:31.881636 22484 solver.cpp:464] Iteration 418, lr = 1e-05
I1118 02:33:34.107409 22484 solver.cpp:209] Iteration 419, loss = 0.482497
I1118 02:33:34.107439 22484 solver.cpp:464] Iteration 419, lr = 1e-05
I1118 02:33:36.339359 22484 solver.cpp:209] Iteration 420, loss = 0.152651
I1118 02:33:36.339388 22484 solver.cpp:464] Iteration 420, lr = 1e-05
I1118 02:33:38.565580 22484 solver.cpp:209] Iteration 421, loss = 0.188379
I1118 02:33:38.565665 22484 solver.cpp:464] Iteration 421, lr = 1e-05
I1118 02:33:40.793826 22484 solver.cpp:209] Iteration 422, loss = 0.119678
I1118 02:33:40.793854 22484 solver.cpp:464] Iteration 422, lr = 1e-05
I1118 02:33:43.023128 22484 solver.cpp:209] Iteration 423, loss = 0.433023
I1118 02:33:43.023159 22484 solver.cpp:464] Iteration 423, lr = 1e-05
I1118 02:33:45.256992 22484 solver.cpp:209] Iteration 424, loss = 0.347278
I1118 02:33:45.257035 22484 solver.cpp:464] Iteration 424, lr = 1e-05
I1118 02:33:47.497375 22484 solver.cpp:209] Iteration 425, loss = 0.324157
I1118 02:33:47.497417 22484 solver.cpp:464] Iteration 425, lr = 1e-05
I1118 02:33:49.730144 22484 solver.cpp:209] Iteration 426, loss = 0.121041
I1118 02:33:49.730185 22484 solver.cpp:464] Iteration 426, lr = 1e-05
I1118 02:33:51.964555 22484 solver.cpp:209] Iteration 427, loss = 0.336158
I1118 02:33:51.964596 22484 solver.cpp:464] Iteration 427, lr = 1e-05
I1118 02:33:54.191376 22484 solver.cpp:209] Iteration 428, loss = 0.0675306
I1118 02:33:54.191406 22484 solver.cpp:464] Iteration 428, lr = 1e-05
I1118 02:33:56.415390 22484 solver.cpp:209] Iteration 429, loss = 0.1075
I1118 02:33:56.415432 22484 solver.cpp:464] Iteration 429, lr = 1e-05
I1118 02:33:58.657032 22484 solver.cpp:209] Iteration 430, loss = 0.212483
I1118 02:33:58.657062 22484 solver.cpp:464] Iteration 430, lr = 1e-05
I1118 02:34:00.890185 22484 solver.cpp:209] Iteration 431, loss = 0.234979
I1118 02:34:00.890210 22484 solver.cpp:464] Iteration 431, lr = 1e-05
I1118 02:34:03.126613 22484 solver.cpp:209] Iteration 432, loss = 0.315808
I1118 02:34:03.126644 22484 solver.cpp:464] Iteration 432, lr = 1e-05
I1118 02:34:05.355556 22484 solver.cpp:209] Iteration 433, loss = 0.218611
I1118 02:34:05.355586 22484 solver.cpp:464] Iteration 433, lr = 1e-05
I1118 02:34:07.583483 22484 solver.cpp:209] Iteration 434, loss = 0.264091
I1118 02:34:07.583513 22484 solver.cpp:464] Iteration 434, lr = 1e-05
I1118 02:34:09.807404 22484 solver.cpp:209] Iteration 435, loss = 0.210622
I1118 02:34:09.807494 22484 solver.cpp:464] Iteration 435, lr = 1e-05
I1118 02:34:12.029710 22484 solver.cpp:209] Iteration 436, loss = 0.168054
I1118 02:34:12.029750 22484 solver.cpp:464] Iteration 436, lr = 1e-05
I1118 02:34:14.259655 22484 solver.cpp:209] Iteration 437, loss = 0.303683
I1118 02:34:14.259696 22484 solver.cpp:464] Iteration 437, lr = 1e-05
I1118 02:34:16.490906 22484 solver.cpp:209] Iteration 438, loss = 0.415745
I1118 02:34:16.490936 22484 solver.cpp:464] Iteration 438, lr = 1e-05
I1118 02:34:18.723289 22484 solver.cpp:209] Iteration 439, loss = 0.258738
I1118 02:34:18.723319 22484 solver.cpp:464] Iteration 439, lr = 1e-05
I1118 02:34:20.951938 22484 solver.cpp:209] Iteration 440, loss = 0.0882362
I1118 02:34:20.951966 22484 solver.cpp:464] Iteration 440, lr = 1e-05
I1118 02:34:23.171268 22484 solver.cpp:209] Iteration 441, loss = 0.678559
I1118 02:34:23.171310 22484 solver.cpp:464] Iteration 441, lr = 1e-05
I1118 02:34:25.402360 22484 solver.cpp:209] Iteration 442, loss = 0.308999
I1118 02:34:25.402390 22484 solver.cpp:464] Iteration 442, lr = 1e-05
I1118 02:34:27.631963 22484 solver.cpp:209] Iteration 443, loss = 0.232237
I1118 02:34:27.631991 22484 solver.cpp:464] Iteration 443, lr = 1e-05
I1118 02:34:29.868939 22484 solver.cpp:209] Iteration 444, loss = 0.329389
I1118 02:34:29.868968 22484 solver.cpp:464] Iteration 444, lr = 1e-05
I1118 02:34:32.099483 22484 solver.cpp:209] Iteration 445, loss = 0.16756
I1118 02:34:32.099514 22484 solver.cpp:464] Iteration 445, lr = 1e-05
I1118 02:34:34.330010 22484 solver.cpp:209] Iteration 446, loss = 0.407221
I1118 02:34:34.330041 22484 solver.cpp:464] Iteration 446, lr = 1e-05
I1118 02:34:36.563205 22484 solver.cpp:209] Iteration 447, loss = 0.231109
I1118 02:34:36.563235 22484 solver.cpp:464] Iteration 447, lr = 1e-05
I1118 02:34:38.785797 22484 solver.cpp:209] Iteration 448, loss = 0.3167
I1118 02:34:38.785827 22484 solver.cpp:464] Iteration 448, lr = 1e-05
I1118 02:34:41.011490 22484 solver.cpp:209] Iteration 449, loss = 0.437581
I1118 02:34:41.011533 22484 solver.cpp:464] Iteration 449, lr = 1e-05
I1118 02:34:41.012125 22484 solver.cpp:264] Iteration 450, Testing net (#0)
I1118 02:34:54.745389 22484 solver.cpp:305] Test loss: 0.277857
I1118 02:34:54.745419 22484 solver.cpp:318] mean_score = test_score[0] { = 513} / test_score[1] { = 517 }
I1118 02:34:54.745426 22484 solver.cpp:319]            = 0.992263
I1118 02:34:54.745431 22484 solver.cpp:328]     Test net output #0: accuracy = 0.992263
I1118 02:34:54.745435 22484 solver.cpp:318] mean_score = test_score[2] { = 1} / test_score[3] { = 59 }
I1118 02:34:54.745440 22484 solver.cpp:319]            = 0.0169492
I1118 02:34:54.745445 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0169492
I1118 02:34:54.745448 22484 solver.cpp:332]     Test net output #2: accuracy = 0.892361
I1118 02:34:54.745452 22484 solver.cpp:334]     Test net output #3: accuracy = 0.504606
I1118 02:34:55.383169 22484 solver.cpp:209] Iteration 450, loss = 0.526844
I1118 02:34:55.383211 22484 solver.cpp:464] Iteration 450, lr = 1e-05
I1118 02:34:57.609388 22484 solver.cpp:209] Iteration 451, loss = 0.197751
I1118 02:34:57.609415 22484 solver.cpp:464] Iteration 451, lr = 1e-05
I1118 02:34:59.839789 22484 solver.cpp:209] Iteration 452, loss = 0.0838347
I1118 02:34:59.839830 22484 solver.cpp:464] Iteration 452, lr = 1e-05
I1118 02:35:02.077164 22484 solver.cpp:209] Iteration 453, loss = 0.283189
I1118 02:35:02.077205 22484 solver.cpp:464] Iteration 453, lr = 1e-05
I1118 02:35:04.327848 22484 solver.cpp:209] Iteration 454, loss = 0.335183
I1118 02:35:04.327877 22484 solver.cpp:464] Iteration 454, lr = 1e-05
I1118 02:35:06.565863 22484 solver.cpp:209] Iteration 455, loss = 0.261249
I1118 02:35:06.565891 22484 solver.cpp:464] Iteration 455, lr = 1e-05
I1118 02:35:08.807636 22484 solver.cpp:209] Iteration 456, loss = 0.227448
I1118 02:35:08.807663 22484 solver.cpp:464] Iteration 456, lr = 1e-05
I1118 02:35:11.048112 22484 solver.cpp:209] Iteration 457, loss = 0.149679
I1118 02:35:11.048189 22484 solver.cpp:464] Iteration 457, lr = 1e-05
I1118 02:35:13.290902 22484 solver.cpp:209] Iteration 458, loss = 0.168015
I1118 02:35:13.290931 22484 solver.cpp:464] Iteration 458, lr = 1e-05
I1118 02:35:15.542345 22484 solver.cpp:209] Iteration 459, loss = 0.228954
I1118 02:35:15.542372 22484 solver.cpp:464] Iteration 459, lr = 1e-05
I1118 02:35:17.786056 22484 solver.cpp:209] Iteration 460, loss = 0.658961
I1118 02:35:17.786085 22484 solver.cpp:464] Iteration 460, lr = 1e-05
I1118 02:35:20.037117 22484 solver.cpp:209] Iteration 461, loss = 0.547916
I1118 02:35:20.037147 22484 solver.cpp:464] Iteration 461, lr = 1e-05
I1118 02:35:22.260114 22484 solver.cpp:209] Iteration 462, loss = 0.325037
I1118 02:35:22.260155 22484 solver.cpp:464] Iteration 462, lr = 1e-05
I1118 02:35:24.485388 22484 solver.cpp:209] Iteration 463, loss = 0.199646
I1118 02:35:24.485417 22484 solver.cpp:464] Iteration 463, lr = 1e-05
I1118 02:35:26.708969 22484 solver.cpp:209] Iteration 464, loss = 0.45722
I1118 02:35:26.708998 22484 solver.cpp:464] Iteration 464, lr = 1e-05
I1118 02:35:28.935618 22484 solver.cpp:209] Iteration 465, loss = 0.117459
I1118 02:35:28.935658 22484 solver.cpp:464] Iteration 465, lr = 1e-05
I1118 02:35:31.170099 22484 solver.cpp:209] Iteration 466, loss = 0.183137
I1118 02:35:31.170136 22484 solver.cpp:464] Iteration 466, lr = 1e-05
I1118 02:35:33.395748 22484 solver.cpp:209] Iteration 467, loss = 0.373203
I1118 02:35:33.395791 22484 solver.cpp:464] Iteration 467, lr = 1e-05
I1118 02:35:35.628640 22484 solver.cpp:209] Iteration 468, loss = 0.194231
I1118 02:35:35.628670 22484 solver.cpp:464] Iteration 468, lr = 1e-05
I1118 02:35:37.854962 22484 solver.cpp:209] Iteration 469, loss = 0.372335
I1118 02:35:37.855005 22484 solver.cpp:464] Iteration 469, lr = 1e-05
I1118 02:35:40.075731 22484 solver.cpp:209] Iteration 470, loss = 0.149942
I1118 02:35:40.075759 22484 solver.cpp:464] Iteration 470, lr = 1e-05
I1118 02:35:42.304747 22484 solver.cpp:209] Iteration 471, loss = 0.212332
I1118 02:35:42.304795 22484 solver.cpp:464] Iteration 471, lr = 1e-05
I1118 02:35:44.529238 22484 solver.cpp:209] Iteration 472, loss = 0.427756
I1118 02:35:44.529279 22484 solver.cpp:464] Iteration 472, lr = 1e-05
I1118 02:35:46.788750 22484 solver.cpp:209] Iteration 473, loss = 0.484772
I1118 02:35:46.788791 22484 solver.cpp:464] Iteration 473, lr = 1e-05
I1118 02:35:49.019561 22484 solver.cpp:209] Iteration 474, loss = 0.4124
I1118 02:35:49.019590 22484 solver.cpp:464] Iteration 474, lr = 1e-05
I1118 02:35:51.247793 22484 solver.cpp:209] Iteration 475, loss = 0.640085
I1118 02:35:51.247822 22484 solver.cpp:464] Iteration 475, lr = 1e-05
I1118 02:35:53.478212 22484 solver.cpp:209] Iteration 476, loss = 0.258781
I1118 02:35:53.478253 22484 solver.cpp:464] Iteration 476, lr = 1e-05
I1118 02:35:55.704139 22484 solver.cpp:209] Iteration 477, loss = 0.53024
I1118 02:35:55.704180 22484 solver.cpp:464] Iteration 477, lr = 1e-05
I1118 02:35:57.932962 22484 solver.cpp:209] Iteration 478, loss = 0.25325
I1118 02:35:57.932992 22484 solver.cpp:464] Iteration 478, lr = 1e-05
I1118 02:36:00.161092 22484 solver.cpp:209] Iteration 479, loss = 0.566556
I1118 02:36:00.161134 22484 solver.cpp:464] Iteration 479, lr = 1e-05
I1118 02:36:02.393225 22484 solver.cpp:209] Iteration 480, loss = 0.38127
I1118 02:36:02.393267 22484 solver.cpp:464] Iteration 480, lr = 1e-05
I1118 02:36:04.632275 22484 solver.cpp:209] Iteration 481, loss = 0.244289
I1118 02:36:04.632316 22484 solver.cpp:464] Iteration 481, lr = 1e-05
I1118 02:36:06.859436 22484 solver.cpp:209] Iteration 482, loss = 0.38024
I1118 02:36:06.859467 22484 solver.cpp:464] Iteration 482, lr = 1e-05
I1118 02:36:09.085711 22484 solver.cpp:209] Iteration 483, loss = 0.413354
I1118 02:36:09.085739 22484 solver.cpp:464] Iteration 483, lr = 1e-05
I1118 02:36:11.309955 22484 solver.cpp:209] Iteration 484, loss = 0.274362
I1118 02:36:11.309984 22484 solver.cpp:464] Iteration 484, lr = 1e-05
I1118 02:36:13.509759 22484 solver.cpp:209] Iteration 485, loss = 0.494111
I1118 02:36:13.509819 22484 solver.cpp:464] Iteration 485, lr = 1e-05
I1118 02:36:15.707852 22484 solver.cpp:209] Iteration 486, loss = 0.212144
I1118 02:36:15.707895 22484 solver.cpp:464] Iteration 486, lr = 1e-05
I1118 02:36:17.905618 22484 solver.cpp:209] Iteration 487, loss = 0.296478
I1118 02:36:17.905647 22484 solver.cpp:464] Iteration 487, lr = 1e-05
I1118 02:36:20.103479 22484 solver.cpp:209] Iteration 488, loss = 0.257586
I1118 02:36:20.103508 22484 solver.cpp:464] Iteration 488, lr = 1e-05
I1118 02:36:22.297624 22484 solver.cpp:209] Iteration 489, loss = 0.336444
I1118 02:36:22.297667 22484 solver.cpp:464] Iteration 489, lr = 1e-05
I1118 02:36:24.485247 22484 solver.cpp:209] Iteration 490, loss = 0.426254
I1118 02:36:24.485288 22484 solver.cpp:464] Iteration 490, lr = 1e-05
I1118 02:36:26.673089 22484 solver.cpp:209] Iteration 491, loss = 0.164551
I1118 02:36:26.673118 22484 solver.cpp:464] Iteration 491, lr = 1e-05
I1118 02:36:28.869307 22484 solver.cpp:209] Iteration 492, loss = 0.282296
I1118 02:36:28.869338 22484 solver.cpp:464] Iteration 492, lr = 1e-05
I1118 02:36:31.064656 22484 solver.cpp:209] Iteration 493, loss = 0.181749
I1118 02:36:31.064694 22484 solver.cpp:464] Iteration 493, lr = 1e-05
I1118 02:36:33.279557 22484 solver.cpp:209] Iteration 494, loss = 0.366174
I1118 02:36:33.279587 22484 solver.cpp:464] Iteration 494, lr = 1e-05
I1118 02:36:35.498273 22484 solver.cpp:209] Iteration 495, loss = 0.392782
I1118 02:36:35.498316 22484 solver.cpp:464] Iteration 495, lr = 1e-05
I1118 02:36:37.721982 22484 solver.cpp:209] Iteration 496, loss = 0.508109
I1118 02:36:37.722013 22484 solver.cpp:464] Iteration 496, lr = 1e-05
I1118 02:36:39.946212 22484 solver.cpp:209] Iteration 497, loss = 0.550672
I1118 02:36:39.946254 22484 solver.cpp:464] Iteration 497, lr = 1e-05
I1118 02:36:42.170742 22484 solver.cpp:209] Iteration 498, loss = 0.311055
I1118 02:36:42.170773 22484 solver.cpp:464] Iteration 498, lr = 1e-05
I1118 02:36:44.409575 22484 solver.cpp:209] Iteration 499, loss = 0.10087
I1118 02:36:44.409661 22484 solver.cpp:464] Iteration 499, lr = 1e-05
I1118 02:36:44.410264 22484 solver.cpp:264] Iteration 500, Testing net (#0)
I1118 02:36:58.295583 22484 solver.cpp:305] Test loss: 0.276304
I1118 02:36:58.295613 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:36:58.295631 22484 solver.cpp:319]            = 0.994197
I1118 02:36:58.295636 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:36:58.295640 22484 solver.cpp:318] mean_score = test_score[2] { = 1} / test_score[3] { = 59 }
I1118 02:36:58.295645 22484 solver.cpp:319]            = 0.0169492
I1118 02:36:58.295649 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0169492
I1118 02:36:58.295653 22484 solver.cpp:332]     Test net output #2: accuracy = 0.894097
I1118 02:36:58.295657 22484 solver.cpp:334]     Test net output #3: accuracy = 0.505573
I1118 02:36:58.942845 22484 solver.cpp:209] Iteration 500, loss = 0.136537
I1118 02:36:58.942878 22484 solver.cpp:464] Iteration 500, lr = 1e-05
I1118 02:37:01.177650 22484 solver.cpp:209] Iteration 501, loss = 0.245156
I1118 02:37:01.177676 22484 solver.cpp:464] Iteration 501, lr = 1e-05
I1118 02:37:03.408826 22484 solver.cpp:209] Iteration 502, loss = 0.287026
I1118 02:37:03.408856 22484 solver.cpp:464] Iteration 502, lr = 1e-05
I1118 02:37:05.639333 22484 solver.cpp:209] Iteration 503, loss = 0.275313
I1118 02:37:05.639375 22484 solver.cpp:464] Iteration 503, lr = 1e-05
I1118 02:37:07.856832 22484 solver.cpp:209] Iteration 504, loss = 0.653596
I1118 02:37:07.856873 22484 solver.cpp:464] Iteration 504, lr = 1e-05
I1118 02:37:10.077328 22484 solver.cpp:209] Iteration 505, loss = 0.0871341
I1118 02:37:10.077370 22484 solver.cpp:464] Iteration 505, lr = 1e-05
I1118 02:37:12.304057 22484 solver.cpp:209] Iteration 506, loss = 0.305396
I1118 02:37:12.304086 22484 solver.cpp:464] Iteration 506, lr = 1e-05
I1118 02:37:14.557206 22484 solver.cpp:209] Iteration 507, loss = 0.282042
I1118 02:37:14.557289 22484 solver.cpp:464] Iteration 507, lr = 1e-05
I1118 02:37:16.787363 22484 solver.cpp:209] Iteration 508, loss = 0.27044
I1118 02:37:16.787405 22484 solver.cpp:464] Iteration 508, lr = 1e-05
I1118 02:37:19.021601 22484 solver.cpp:209] Iteration 509, loss = 0.549796
I1118 02:37:19.021643 22484 solver.cpp:464] Iteration 509, lr = 1e-05
I1118 02:37:21.245721 22484 solver.cpp:209] Iteration 510, loss = 0.213858
I1118 02:37:21.245751 22484 solver.cpp:464] Iteration 510, lr = 1e-05
I1118 02:37:23.472633 22484 solver.cpp:209] Iteration 511, loss = 0.259428
I1118 02:37:23.472676 22484 solver.cpp:464] Iteration 511, lr = 1e-05
I1118 02:37:25.698833 22484 solver.cpp:209] Iteration 512, loss = 0.234109
I1118 02:37:25.698863 22484 solver.cpp:464] Iteration 512, lr = 1e-05
I1118 02:37:27.926013 22484 solver.cpp:209] Iteration 513, loss = 0.154806
I1118 02:37:27.926043 22484 solver.cpp:464] Iteration 513, lr = 1e-05
I1118 02:37:30.163208 22484 solver.cpp:209] Iteration 514, loss = 0.331513
I1118 02:37:30.163238 22484 solver.cpp:464] Iteration 514, lr = 1e-05
I1118 02:37:32.390671 22484 solver.cpp:209] Iteration 515, loss = 0.307345
I1118 02:37:32.390700 22484 solver.cpp:464] Iteration 515, lr = 1e-05
I1118 02:37:34.630218 22484 solver.cpp:209] Iteration 516, loss = 0.0904763
I1118 02:37:34.630249 22484 solver.cpp:464] Iteration 516, lr = 1e-05
I1118 02:37:36.856853 22484 solver.cpp:209] Iteration 517, loss = 0.298035
I1118 02:37:36.856881 22484 solver.cpp:464] Iteration 517, lr = 1e-05
I1118 02:37:39.081784 22484 solver.cpp:209] Iteration 518, loss = 0.289935
I1118 02:37:39.081812 22484 solver.cpp:464] Iteration 518, lr = 1e-05
I1118 02:37:41.306748 22484 solver.cpp:209] Iteration 519, loss = 0.200111
I1118 02:37:41.306779 22484 solver.cpp:464] Iteration 519, lr = 1e-05
I1118 02:37:43.530277 22484 solver.cpp:209] Iteration 520, loss = 0.251753
I1118 02:37:43.530319 22484 solver.cpp:464] Iteration 520, lr = 1e-05
I1118 02:37:45.769590 22484 solver.cpp:209] Iteration 521, loss = 0.435118
I1118 02:37:45.769678 22484 solver.cpp:464] Iteration 521, lr = 1e-05
I1118 02:37:48.002785 22484 solver.cpp:209] Iteration 522, loss = 0.384099
I1118 02:37:48.002815 22484 solver.cpp:464] Iteration 522, lr = 1e-05
I1118 02:37:50.232098 22484 solver.cpp:209] Iteration 523, loss = 0.152421
I1118 02:37:50.232127 22484 solver.cpp:464] Iteration 523, lr = 1e-05
I1118 02:37:52.461633 22484 solver.cpp:209] Iteration 524, loss = 0.208491
I1118 02:37:52.461675 22484 solver.cpp:464] Iteration 524, lr = 1e-05
I1118 02:37:54.680716 22484 solver.cpp:209] Iteration 525, loss = 0.293703
I1118 02:37:54.680758 22484 solver.cpp:464] Iteration 525, lr = 1e-05
I1118 02:37:56.914405 22484 solver.cpp:209] Iteration 526, loss = 0.457469
I1118 02:37:56.914435 22484 solver.cpp:464] Iteration 526, lr = 1e-05
I1118 02:37:59.141968 22484 solver.cpp:209] Iteration 527, loss = 0.535666
I1118 02:37:59.141999 22484 solver.cpp:464] Iteration 527, lr = 1e-05
I1118 02:38:01.380321 22484 solver.cpp:209] Iteration 528, loss = 0.196984
I1118 02:38:01.380350 22484 solver.cpp:464] Iteration 528, lr = 1e-05
I1118 02:38:03.610646 22484 solver.cpp:209] Iteration 529, loss = 0.297878
I1118 02:38:03.610677 22484 solver.cpp:464] Iteration 529, lr = 1e-05
I1118 02:38:05.841800 22484 solver.cpp:209] Iteration 530, loss = 0.23559
I1118 02:38:05.841830 22484 solver.cpp:464] Iteration 530, lr = 1e-05
I1118 02:38:08.068744 22484 solver.cpp:209] Iteration 531, loss = 0.0923647
I1118 02:38:08.068785 22484 solver.cpp:464] Iteration 531, lr = 1e-05
I1118 02:38:10.286216 22484 solver.cpp:209] Iteration 532, loss = 0.136065
I1118 02:38:10.286247 22484 solver.cpp:464] Iteration 532, lr = 1e-05
I1118 02:38:12.516176 22484 solver.cpp:209] Iteration 533, loss = 0.174771
I1118 02:38:12.516217 22484 solver.cpp:464] Iteration 533, lr = 1e-05
I1118 02:38:14.741981 22484 solver.cpp:209] Iteration 534, loss = 0.222354
I1118 02:38:14.742009 22484 solver.cpp:464] Iteration 534, lr = 1e-05
I1118 02:38:16.974030 22484 solver.cpp:209] Iteration 535, loss = 0.323585
I1118 02:38:16.974115 22484 solver.cpp:464] Iteration 535, lr = 1e-05
I1118 02:38:19.205564 22484 solver.cpp:209] Iteration 536, loss = 0.42843
I1118 02:38:19.205592 22484 solver.cpp:464] Iteration 536, lr = 1e-05
I1118 02:38:21.436506 22484 solver.cpp:209] Iteration 537, loss = 0.272401
I1118 02:38:21.436547 22484 solver.cpp:464] Iteration 537, lr = 1e-05
I1118 02:38:23.666465 22484 solver.cpp:209] Iteration 538, loss = 0.189519
I1118 02:38:23.666493 22484 solver.cpp:464] Iteration 538, lr = 1e-05
I1118 02:38:25.893446 22484 solver.cpp:209] Iteration 539, loss = 0.12691
I1118 02:38:25.893486 22484 solver.cpp:464] Iteration 539, lr = 1e-05
I1118 02:38:28.120870 22484 solver.cpp:209] Iteration 540, loss = 0.355146
I1118 02:38:28.120909 22484 solver.cpp:464] Iteration 540, lr = 1e-05
I1118 02:38:30.349355 22484 solver.cpp:209] Iteration 541, loss = 0.302178
I1118 02:38:30.349385 22484 solver.cpp:464] Iteration 541, lr = 1e-05
I1118 02:38:32.583999 22484 solver.cpp:209] Iteration 542, loss = 0.213691
I1118 02:38:32.584028 22484 solver.cpp:464] Iteration 542, lr = 1e-05
I1118 02:38:34.818141 22484 solver.cpp:209] Iteration 543, loss = 0.146135
I1118 02:38:34.818172 22484 solver.cpp:464] Iteration 543, lr = 1e-05
I1118 02:38:37.058262 22484 solver.cpp:209] Iteration 544, loss = 0.610281
I1118 02:38:37.058305 22484 solver.cpp:464] Iteration 544, lr = 1e-05
I1118 02:38:39.280910 22484 solver.cpp:209] Iteration 545, loss = 0.230122
I1118 02:38:39.280951 22484 solver.cpp:464] Iteration 545, lr = 1e-05
I1118 02:38:41.509044 22484 solver.cpp:209] Iteration 546, loss = 0.195926
I1118 02:38:41.509073 22484 solver.cpp:464] Iteration 546, lr = 1e-05
I1118 02:38:43.725798 22484 solver.cpp:209] Iteration 547, loss = 0.39115
I1118 02:38:43.725827 22484 solver.cpp:464] Iteration 547, lr = 1e-05
I1118 02:38:45.920815 22484 solver.cpp:209] Iteration 548, loss = 0.208155
I1118 02:38:45.920856 22484 solver.cpp:464] Iteration 548, lr = 1e-05
I1118 02:38:48.118062 22484 solver.cpp:209] Iteration 549, loss = 0.247072
I1118 02:38:48.118154 22484 solver.cpp:464] Iteration 549, lr = 1e-05
I1118 02:38:48.118774 22484 solver.cpp:264] Iteration 550, Testing net (#0)
I1118 02:39:01.876068 22484 solver.cpp:305] Test loss: 0.27189
I1118 02:39:01.876097 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:39:01.876116 22484 solver.cpp:319]            = 0.994197
I1118 02:39:01.876121 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:39:01.876126 22484 solver.cpp:318] mean_score = test_score[2] { = 3} / test_score[3] { = 59 }
I1118 02:39:01.876130 22484 solver.cpp:319]            = 0.0508475
I1118 02:39:01.876133 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0508475
I1118 02:39:01.876138 22484 solver.cpp:332]     Test net output #2: accuracy = 0.897569
I1118 02:39:01.876142 22484 solver.cpp:334]     Test net output #3: accuracy = 0.522522
I1118 02:39:02.526640 22484 solver.cpp:209] Iteration 550, loss = 0.312749
I1118 02:39:02.526669 22484 solver.cpp:464] Iteration 550, lr = 1e-05
I1118 02:39:04.768365 22484 solver.cpp:209] Iteration 551, loss = 0.392
I1118 02:39:04.768394 22484 solver.cpp:464] Iteration 551, lr = 1e-05
I1118 02:39:07.009124 22484 solver.cpp:209] Iteration 552, loss = 0.534805
I1118 02:39:07.009166 22484 solver.cpp:464] Iteration 552, lr = 1e-05
I1118 02:39:09.249028 22484 solver.cpp:209] Iteration 553, loss = 0.539368
I1118 02:39:09.249058 22484 solver.cpp:464] Iteration 553, lr = 1e-05
I1118 02:39:11.488111 22484 solver.cpp:209] Iteration 554, loss = 0.168577
I1118 02:39:11.488139 22484 solver.cpp:464] Iteration 554, lr = 1e-05
I1118 02:39:13.727861 22484 solver.cpp:209] Iteration 555, loss = 0.248267
I1118 02:39:13.727902 22484 solver.cpp:464] Iteration 555, lr = 1e-05
I1118 02:39:15.971390 22484 solver.cpp:209] Iteration 556, loss = 0.298343
I1118 02:39:15.971431 22484 solver.cpp:464] Iteration 556, lr = 1e-05
I1118 02:39:18.219605 22484 solver.cpp:209] Iteration 557, loss = 0.324426
I1118 02:39:18.219683 22484 solver.cpp:464] Iteration 557, lr = 1e-05
I1118 02:39:20.467043 22484 solver.cpp:209] Iteration 558, loss = 0.183736
I1118 02:39:20.467072 22484 solver.cpp:464] Iteration 558, lr = 1e-05
I1118 02:39:22.711336 22484 solver.cpp:209] Iteration 559, loss = 0.245356
I1118 02:39:22.711366 22484 solver.cpp:464] Iteration 559, lr = 1e-05
I1118 02:39:24.947476 22484 solver.cpp:209] Iteration 560, loss = 0.167013
I1118 02:39:24.947516 22484 solver.cpp:464] Iteration 560, lr = 1e-05
I1118 02:39:27.168114 22484 solver.cpp:209] Iteration 561, loss = 0.22258
I1118 02:39:27.168143 22484 solver.cpp:464] Iteration 561, lr = 1e-05
I1118 02:39:29.359304 22484 solver.cpp:209] Iteration 562, loss = 0.266004
I1118 02:39:29.359333 22484 solver.cpp:464] Iteration 562, lr = 1e-05
I1118 02:39:31.564018 22484 solver.cpp:209] Iteration 563, loss = 0.442859
I1118 02:39:31.564046 22484 solver.cpp:464] Iteration 563, lr = 1e-05
I1118 02:39:33.768738 22484 solver.cpp:209] Iteration 564, loss = 0.393671
I1118 02:39:33.768779 22484 solver.cpp:464] Iteration 564, lr = 1e-05
I1118 02:39:35.968541 22484 solver.cpp:209] Iteration 565, loss = 0.21673
I1118 02:39:35.968582 22484 solver.cpp:464] Iteration 565, lr = 1e-05
I1118 02:39:38.156546 22484 solver.cpp:209] Iteration 566, loss = 0.257557
I1118 02:39:38.156586 22484 solver.cpp:464] Iteration 566, lr = 1e-05
I1118 02:39:40.347610 22484 solver.cpp:209] Iteration 567, loss = 0.305444
I1118 02:39:40.347652 22484 solver.cpp:464] Iteration 567, lr = 1e-05
I1118 02:39:42.543267 22484 solver.cpp:209] Iteration 568, loss = 0.0856566
I1118 02:39:42.543298 22484 solver.cpp:464] Iteration 568, lr = 1e-05
I1118 02:39:44.745076 22484 solver.cpp:209] Iteration 569, loss = 0.309996
I1118 02:39:44.745116 22484 solver.cpp:464] Iteration 569, lr = 1e-05
I1118 02:39:46.943967 22484 solver.cpp:209] Iteration 570, loss = 0.217448
I1118 02:39:46.944008 22484 solver.cpp:464] Iteration 570, lr = 1e-05
I1118 02:39:49.138631 22484 solver.cpp:209] Iteration 571, loss = 0.250117
I1118 02:39:49.138697 22484 solver.cpp:464] Iteration 571, lr = 1e-05
I1118 02:39:51.326216 22484 solver.cpp:209] Iteration 572, loss = 0.394604
I1118 02:39:51.326258 22484 solver.cpp:464] Iteration 572, lr = 1e-05
I1118 02:39:53.517195 22484 solver.cpp:209] Iteration 573, loss = 0.161766
I1118 02:39:53.517225 22484 solver.cpp:464] Iteration 573, lr = 1e-05
I1118 02:39:55.744570 22484 solver.cpp:209] Iteration 574, loss = 0.291657
I1118 02:39:55.744598 22484 solver.cpp:464] Iteration 574, lr = 1e-05
I1118 02:39:57.973949 22484 solver.cpp:209] Iteration 575, loss = 0.308278
I1118 02:39:57.973991 22484 solver.cpp:464] Iteration 575, lr = 1e-05
I1118 02:40:00.202078 22484 solver.cpp:209] Iteration 576, loss = 0.466415
I1118 02:40:00.202108 22484 solver.cpp:464] Iteration 576, lr = 1e-05
I1118 02:40:02.427943 22484 solver.cpp:209] Iteration 577, loss = 0.492092
I1118 02:40:02.427973 22484 solver.cpp:464] Iteration 577, lr = 1e-05
I1118 02:40:04.662559 22484 solver.cpp:209] Iteration 578, loss = 0.51133
I1118 02:40:04.662608 22484 solver.cpp:464] Iteration 578, lr = 1e-05
I1118 02:40:06.887205 22484 solver.cpp:209] Iteration 579, loss = 0.386495
I1118 02:40:06.887235 22484 solver.cpp:464] Iteration 579, lr = 1e-05
I1118 02:40:09.111419 22484 solver.cpp:209] Iteration 580, loss = 0.279315
I1118 02:40:09.111449 22484 solver.cpp:464] Iteration 580, lr = 1e-05
I1118 02:40:11.332841 22484 solver.cpp:209] Iteration 581, loss = 0.4217
I1118 02:40:11.332883 22484 solver.cpp:464] Iteration 581, lr = 1e-05
I1118 02:40:13.567886 22484 solver.cpp:209] Iteration 582, loss = 0.352394
I1118 02:40:13.567929 22484 solver.cpp:464] Iteration 582, lr = 1e-05
I1118 02:40:15.800935 22484 solver.cpp:209] Iteration 583, loss = 0.337523
I1118 02:40:15.800966 22484 solver.cpp:464] Iteration 583, lr = 1e-05
I1118 02:40:18.036576 22484 solver.cpp:209] Iteration 584, loss = 0.347805
I1118 02:40:18.036617 22484 solver.cpp:464] Iteration 584, lr = 1e-05
I1118 02:40:20.259708 22484 solver.cpp:209] Iteration 585, loss = 0.359124
I1118 02:40:20.259793 22484 solver.cpp:464] Iteration 585, lr = 1e-05
I1118 02:40:22.480316 22484 solver.cpp:209] Iteration 586, loss = 0.433133
I1118 02:40:22.480358 22484 solver.cpp:464] Iteration 586, lr = 1e-05
I1118 02:40:24.701846 22484 solver.cpp:209] Iteration 587, loss = 0.247499
I1118 02:40:24.701875 22484 solver.cpp:464] Iteration 587, lr = 1e-05
I1118 02:40:26.928186 22484 solver.cpp:209] Iteration 588, loss = 0.525623
I1118 02:40:26.928228 22484 solver.cpp:464] Iteration 588, lr = 1e-05
I1118 02:40:29.169483 22484 solver.cpp:209] Iteration 589, loss = 0.072804
I1118 02:40:29.169527 22484 solver.cpp:464] Iteration 589, lr = 1e-05
I1118 02:40:31.393448 22484 solver.cpp:209] Iteration 590, loss = 0.388641
I1118 02:40:31.393491 22484 solver.cpp:464] Iteration 590, lr = 1e-05
I1118 02:40:33.628118 22484 solver.cpp:209] Iteration 591, loss = 0.141847
I1118 02:40:33.628147 22484 solver.cpp:464] Iteration 591, lr = 1e-05
I1118 02:40:35.846520 22484 solver.cpp:209] Iteration 592, loss = 0.328476
I1118 02:40:35.846562 22484 solver.cpp:464] Iteration 592, lr = 1e-05
I1118 02:40:38.070858 22484 solver.cpp:209] Iteration 593, loss = 0.329166
I1118 02:40:38.070888 22484 solver.cpp:464] Iteration 593, lr = 1e-05
I1118 02:40:40.297333 22484 solver.cpp:209] Iteration 594, loss = 0.23409
I1118 02:40:40.297361 22484 solver.cpp:464] Iteration 594, lr = 1e-05
I1118 02:40:42.524408 22484 solver.cpp:209] Iteration 595, loss = 0.29419
I1118 02:40:42.524438 22484 solver.cpp:464] Iteration 595, lr = 1e-05
I1118 02:40:44.752614 22484 solver.cpp:209] Iteration 596, loss = 0.240309
I1118 02:40:44.752641 22484 solver.cpp:464] Iteration 596, lr = 1e-05
I1118 02:40:46.976819 22484 solver.cpp:209] Iteration 597, loss = 0.352423
I1118 02:40:46.976861 22484 solver.cpp:464] Iteration 597, lr = 1e-05
I1118 02:40:49.209059 22484 solver.cpp:209] Iteration 598, loss = 0.423954
I1118 02:40:49.209100 22484 solver.cpp:464] Iteration 598, lr = 1e-05
I1118 02:40:51.434401 22484 solver.cpp:209] Iteration 599, loss = 0.610466
I1118 02:40:51.434464 22484 solver.cpp:464] Iteration 599, lr = 1e-05
I1118 02:40:51.435063 22484 solver.cpp:264] Iteration 600, Testing net (#0)
I1118 02:41:05.306845 22484 solver.cpp:305] Test loss: 0.277809
I1118 02:41:05.306888 22484 solver.cpp:318] mean_score = test_score[0] { = 512} / test_score[1] { = 517 }
I1118 02:41:05.306895 22484 solver.cpp:319]            = 0.990329
I1118 02:41:05.306900 22484 solver.cpp:328]     Test net output #0: accuracy = 0.990329
I1118 02:41:05.306903 22484 solver.cpp:318] mean_score = test_score[2] { = 3} / test_score[3] { = 59 }
I1118 02:41:05.306908 22484 solver.cpp:319]            = 0.0508475
I1118 02:41:05.306911 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0508475
I1118 02:41:05.306916 22484 solver.cpp:332]     Test net output #2: accuracy = 0.894097
I1118 02:41:05.306921 22484 solver.cpp:334]     Test net output #3: accuracy = 0.520588
I1118 02:41:05.952486 22484 solver.cpp:209] Iteration 600, loss = 0.404745
I1118 02:41:05.952514 22484 solver.cpp:464] Iteration 600, lr = 1e-05
I1118 02:41:08.179061 22484 solver.cpp:209] Iteration 601, loss = 0.267888
I1118 02:41:08.179101 22484 solver.cpp:464] Iteration 601, lr = 1e-05
I1118 02:41:10.402699 22484 solver.cpp:209] Iteration 602, loss = 0.132612
I1118 02:41:10.402729 22484 solver.cpp:464] Iteration 602, lr = 1e-05
I1118 02:41:12.632043 22484 solver.cpp:209] Iteration 603, loss = 0.0780788
I1118 02:41:12.632084 22484 solver.cpp:464] Iteration 603, lr = 1e-05
I1118 02:41:14.864220 22484 solver.cpp:209] Iteration 604, loss = 0.368828
I1118 02:41:14.864248 22484 solver.cpp:464] Iteration 604, lr = 1e-05
I1118 02:41:17.096317 22484 solver.cpp:209] Iteration 605, loss = 0.339414
I1118 02:41:17.096345 22484 solver.cpp:464] Iteration 605, lr = 1e-05
I1118 02:41:19.329107 22484 solver.cpp:209] Iteration 606, loss = 0.263959
I1118 02:41:19.329136 22484 solver.cpp:464] Iteration 606, lr = 1e-05
I1118 02:41:21.550766 22484 solver.cpp:209] Iteration 607, loss = 0.484028
I1118 02:41:21.550847 22484 solver.cpp:464] Iteration 607, lr = 1e-05
I1118 02:41:23.779652 22484 solver.cpp:209] Iteration 608, loss = 0.275533
I1118 02:41:23.779693 22484 solver.cpp:464] Iteration 608, lr = 1e-05
I1118 02:41:26.005231 22484 solver.cpp:209] Iteration 609, loss = 0.259809
I1118 02:41:26.005260 22484 solver.cpp:464] Iteration 609, lr = 1e-05
I1118 02:41:28.234096 22484 solver.cpp:209] Iteration 610, loss = 0.2373
I1118 02:41:28.234122 22484 solver.cpp:464] Iteration 610, lr = 1e-05
I1118 02:41:30.474227 22484 solver.cpp:209] Iteration 611, loss = 0.304998
I1118 02:41:30.474270 22484 solver.cpp:464] Iteration 611, lr = 1e-05
I1118 02:41:32.704695 22484 solver.cpp:209] Iteration 612, loss = 0.484327
I1118 02:41:32.704725 22484 solver.cpp:464] Iteration 612, lr = 1e-05
I1118 02:41:34.943336 22484 solver.cpp:209] Iteration 613, loss = 0.0825974
I1118 02:41:34.943378 22484 solver.cpp:464] Iteration 613, lr = 1e-05
I1118 02:41:37.168457 22484 solver.cpp:209] Iteration 614, loss = 0.407971
I1118 02:41:37.168486 22484 solver.cpp:464] Iteration 614, lr = 1e-05
I1118 02:41:39.390244 22484 solver.cpp:209] Iteration 615, loss = 0.0874336
I1118 02:41:39.390287 22484 solver.cpp:464] Iteration 615, lr = 1e-05
I1118 02:41:41.616262 22484 solver.cpp:209] Iteration 616, loss = 0.175974
I1118 02:41:41.616303 22484 solver.cpp:464] Iteration 616, lr = 1e-05
I1118 02:41:43.840237 22484 solver.cpp:209] Iteration 617, loss = 0.307595
I1118 02:41:43.840266 22484 solver.cpp:464] Iteration 617, lr = 1e-05
I1118 02:41:46.084980 22484 solver.cpp:209] Iteration 618, loss = 0.259327
I1118 02:41:46.085021 22484 solver.cpp:464] Iteration 618, lr = 1e-05
I1118 02:41:48.313299 22484 solver.cpp:209] Iteration 619, loss = 0.114383
I1118 02:41:48.313339 22484 solver.cpp:464] Iteration 619, lr = 1e-05
I1118 02:41:50.544579 22484 solver.cpp:209] Iteration 620, loss = 0.648082
I1118 02:41:50.544608 22484 solver.cpp:464] Iteration 620, lr = 1e-05
I1118 02:41:52.771833 22484 solver.cpp:209] Iteration 621, loss = 0.291869
I1118 02:41:52.771896 22484 solver.cpp:464] Iteration 621, lr = 1e-05
I1118 02:41:54.994336 22484 solver.cpp:209] Iteration 622, loss = 0.282227
I1118 02:41:54.994379 22484 solver.cpp:464] Iteration 622, lr = 1e-05
I1118 02:41:57.221678 22484 solver.cpp:209] Iteration 623, loss = 0.234106
I1118 02:41:57.221706 22484 solver.cpp:464] Iteration 623, lr = 1e-05
I1118 02:41:59.449465 22484 solver.cpp:209] Iteration 624, loss = 0.432893
I1118 02:41:59.449496 22484 solver.cpp:464] Iteration 624, lr = 1e-05
I1118 02:42:01.675143 22484 solver.cpp:209] Iteration 625, loss = 0.295954
I1118 02:42:01.675171 22484 solver.cpp:464] Iteration 625, lr = 1e-05
I1118 02:42:03.905925 22484 solver.cpp:209] Iteration 626, loss = 0.134903
I1118 02:42:03.905956 22484 solver.cpp:464] Iteration 626, lr = 1e-05
I1118 02:42:06.160027 22484 solver.cpp:209] Iteration 627, loss = 0.214118
I1118 02:42:06.160056 22484 solver.cpp:464] Iteration 627, lr = 1e-05
I1118 02:42:08.390301 22484 solver.cpp:209] Iteration 628, loss = 0.167572
I1118 02:42:08.390329 22484 solver.cpp:464] Iteration 628, lr = 1e-05
I1118 02:42:10.615680 22484 solver.cpp:209] Iteration 629, loss = 0.423029
I1118 02:42:10.615710 22484 solver.cpp:464] Iteration 629, lr = 1e-05
I1118 02:42:12.836632 22484 solver.cpp:209] Iteration 630, loss = 0.588032
I1118 02:42:12.836663 22484 solver.cpp:464] Iteration 630, lr = 1e-05
I1118 02:42:15.067005 22484 solver.cpp:209] Iteration 631, loss = 0.183321
I1118 02:42:15.067047 22484 solver.cpp:464] Iteration 631, lr = 1e-05
I1118 02:42:17.296838 22484 solver.cpp:209] Iteration 632, loss = 0.471667
I1118 02:42:17.296869 22484 solver.cpp:464] Iteration 632, lr = 1e-05
I1118 02:42:19.533588 22484 solver.cpp:209] Iteration 633, loss = 0.209936
I1118 02:42:19.533618 22484 solver.cpp:464] Iteration 633, lr = 1e-05
I1118 02:42:21.762169 22484 solver.cpp:209] Iteration 634, loss = 0.0810393
I1118 02:42:21.762199 22484 solver.cpp:464] Iteration 634, lr = 1e-05
I1118 02:42:23.987598 22484 solver.cpp:209] Iteration 635, loss = 0.1163
I1118 02:42:23.987660 22484 solver.cpp:464] Iteration 635, lr = 1e-05
I1118 02:42:26.217872 22484 solver.cpp:209] Iteration 636, loss = 0.332058
I1118 02:42:26.217901 22484 solver.cpp:464] Iteration 636, lr = 1e-05
I1118 02:42:28.440959 22484 solver.cpp:209] Iteration 637, loss = 0.0703065
I1118 02:42:28.440989 22484 solver.cpp:464] Iteration 637, lr = 1e-05
I1118 02:42:30.670789 22484 solver.cpp:209] Iteration 638, loss = 0.320292
I1118 02:42:30.670817 22484 solver.cpp:464] Iteration 638, lr = 1e-05
I1118 02:42:32.902405 22484 solver.cpp:209] Iteration 639, loss = 0.186323
I1118 02:42:32.902436 22484 solver.cpp:464] Iteration 639, lr = 1e-05
I1118 02:42:35.131043 22484 solver.cpp:209] Iteration 640, loss = 0.235864
I1118 02:42:35.131073 22484 solver.cpp:464] Iteration 640, lr = 1e-05
I1118 02:42:37.365685 22484 solver.cpp:209] Iteration 641, loss = 0.150509
I1118 02:42:37.365727 22484 solver.cpp:464] Iteration 641, lr = 1e-05
I1118 02:42:39.588554 22484 solver.cpp:209] Iteration 642, loss = 0.124829
I1118 02:42:39.588595 22484 solver.cpp:464] Iteration 642, lr = 1e-05
I1118 02:42:41.793465 22484 solver.cpp:209] Iteration 643, loss = 0.348673
I1118 02:42:41.793508 22484 solver.cpp:464] Iteration 643, lr = 1e-05
I1118 02:42:43.987447 22484 solver.cpp:209] Iteration 644, loss = 0.326354
I1118 02:42:43.987478 22484 solver.cpp:464] Iteration 644, lr = 1e-05
I1118 02:42:46.211453 22484 solver.cpp:209] Iteration 645, loss = 0.1218
I1118 02:42:46.211493 22484 solver.cpp:464] Iteration 645, lr = 1e-05
I1118 02:42:48.412432 22484 solver.cpp:209] Iteration 646, loss = 0.328137
I1118 02:42:48.412462 22484 solver.cpp:464] Iteration 646, lr = 1e-05
I1118 02:42:50.607532 22484 solver.cpp:209] Iteration 647, loss = 0.530794
I1118 02:42:50.607563 22484 solver.cpp:464] Iteration 647, lr = 1e-05
I1118 02:42:52.804271 22484 solver.cpp:209] Iteration 648, loss = 0.206855
I1118 02:42:52.804301 22484 solver.cpp:464] Iteration 648, lr = 1e-05
I1118 02:42:54.993011 22484 solver.cpp:209] Iteration 649, loss = 0.307014
I1118 02:42:54.993065 22484 solver.cpp:464] Iteration 649, lr = 1e-05
I1118 02:42:54.993684 22484 solver.cpp:264] Iteration 650, Testing net (#0)
I1118 02:43:08.842376 22484 solver.cpp:305] Test loss: 0.276269
I1118 02:43:08.842418 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 02:43:08.842425 22484 solver.cpp:319]            = 0.994197
I1118 02:43:08.842430 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 02:43:08.842434 22484 solver.cpp:318] mean_score = test_score[2] { = 2} / test_score[3] { = 59 }
I1118 02:43:08.842439 22484 solver.cpp:319]            = 0.0338983
I1118 02:43:08.842443 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0338983
I1118 02:43:08.842447 22484 solver.cpp:332]     Test net output #2: accuracy = 0.895833
I1118 02:43:08.842452 22484 solver.cpp:334]     Test net output #3: accuracy = 0.514048
I1118 02:43:09.503968 22484 solver.cpp:209] Iteration 650, loss = 0.250762
I1118 02:43:09.504008 22484 solver.cpp:464] Iteration 650, lr = 1e-05
I1118 02:43:11.773882 22484 solver.cpp:209] Iteration 651, loss = 0.272231
I1118 02:43:11.773912 22484 solver.cpp:464] Iteration 651, lr = 1e-05
I1118 02:43:14.047350 22484 solver.cpp:209] Iteration 652, loss = 0.243149
I1118 02:43:14.047392 22484 solver.cpp:464] Iteration 652, lr = 1e-05
I1118 02:43:16.322906 22484 solver.cpp:209] Iteration 653, loss = 0.266451
I1118 02:43:16.322937 22484 solver.cpp:464] Iteration 653, lr = 1e-05
I1118 02:43:18.567611 22484 solver.cpp:209] Iteration 654, loss = 0.33338
I1118 02:43:18.567641 22484 solver.cpp:464] Iteration 654, lr = 1e-05
I1118 02:43:20.814492 22484 solver.cpp:209] Iteration 655, loss = 0.37226
I1118 02:43:20.814519 22484 solver.cpp:464] Iteration 655, lr = 1e-05
I1118 02:43:23.090054 22484 solver.cpp:209] Iteration 656, loss = 0.383088
I1118 02:43:23.090096 22484 solver.cpp:464] Iteration 656, lr = 1e-05
I1118 02:43:25.320157 22484 solver.cpp:209] Iteration 657, loss = 0.144052
I1118 02:43:25.320242 22484 solver.cpp:464] Iteration 657, lr = 1e-05
I1118 02:43:27.544168 22484 solver.cpp:209] Iteration 658, loss = 0.205449
I1118 02:43:27.544209 22484 solver.cpp:464] Iteration 658, lr = 1e-05
I1118 02:43:29.731498 22484 solver.cpp:209] Iteration 659, loss = 0.270464
I1118 02:43:29.731526 22484 solver.cpp:464] Iteration 659, lr = 1e-05
I1118 02:43:31.931092 22484 solver.cpp:209] Iteration 660, loss = 0.199901
I1118 02:43:31.931121 22484 solver.cpp:464] Iteration 660, lr = 1e-05
I1118 02:43:34.123002 22484 solver.cpp:209] Iteration 661, loss = 0.239191
I1118 02:43:34.123030 22484 solver.cpp:464] Iteration 661, lr = 1e-05
I1118 02:43:36.321486 22484 solver.cpp:209] Iteration 662, loss = 0.186388
I1118 02:43:36.321528 22484 solver.cpp:464] Iteration 662, lr = 1e-05
I1118 02:43:38.511523 22484 solver.cpp:209] Iteration 663, loss = 0.162135
I1118 02:43:38.511564 22484 solver.cpp:464] Iteration 663, lr = 1e-05
I1118 02:43:40.698431 22484 solver.cpp:209] Iteration 664, loss = 0.26244
I1118 02:43:40.698473 22484 solver.cpp:464] Iteration 664, lr = 1e-05
I1118 02:43:42.893213 22484 solver.cpp:209] Iteration 665, loss = 0.363417
I1118 02:43:42.893242 22484 solver.cpp:464] Iteration 665, lr = 1e-05
I1118 02:43:45.096540 22484 solver.cpp:209] Iteration 666, loss = 0.345534
I1118 02:43:45.096582 22484 solver.cpp:464] Iteration 666, lr = 1e-05
I1118 02:43:47.294030 22484 solver.cpp:209] Iteration 667, loss = 0.423814
I1118 02:43:47.294072 22484 solver.cpp:464] Iteration 667, lr = 1e-05
I1118 02:43:49.495409 22484 solver.cpp:209] Iteration 668, loss = 0.238721
I1118 02:43:49.495439 22484 solver.cpp:464] Iteration 668, lr = 1e-05
I1118 02:43:51.686755 22484 solver.cpp:209] Iteration 669, loss = 0.294267
I1118 02:43:51.686785 22484 solver.cpp:464] Iteration 669, lr = 1e-05
I1118 02:43:53.882979 22484 solver.cpp:209] Iteration 670, loss = 0.259867
I1118 02:43:53.883009 22484 solver.cpp:464] Iteration 670, lr = 1e-05
I1118 02:43:56.078758 22484 solver.cpp:209] Iteration 671, loss = 0.111596
I1118 02:43:56.078811 22484 solver.cpp:464] Iteration 671, lr = 1e-05
I1118 02:43:58.306922 22484 solver.cpp:209] Iteration 672, loss = 0.334878
I1118 02:43:58.306962 22484 solver.cpp:464] Iteration 672, lr = 1e-05
I1118 02:44:00.536656 22484 solver.cpp:209] Iteration 673, loss = 0.262581
I1118 02:44:00.536686 22484 solver.cpp:464] Iteration 673, lr = 1e-05
I1118 02:44:02.763828 22484 solver.cpp:209] Iteration 674, loss = 0.254131
I1118 02:44:02.763869 22484 solver.cpp:464] Iteration 674, lr = 1e-05
I1118 02:44:04.994061 22484 solver.cpp:209] Iteration 675, loss = 0.36024
I1118 02:44:04.994091 22484 solver.cpp:464] Iteration 675, lr = 1e-05
I1118 02:44:07.215899 22484 solver.cpp:209] Iteration 676, loss = 0.271304
I1118 02:44:07.215939 22484 solver.cpp:464] Iteration 676, lr = 1e-05
I1118 02:44:09.441470 22484 solver.cpp:209] Iteration 677, loss = 0.318311
I1118 02:44:09.441512 22484 solver.cpp:464] Iteration 677, lr = 1e-05
I1118 02:44:11.669822 22484 solver.cpp:209] Iteration 678, loss = 0.233696
I1118 02:44:11.669863 22484 solver.cpp:464] Iteration 678, lr = 1e-05
I1118 02:44:13.908206 22484 solver.cpp:209] Iteration 679, loss = 0.564011
I1118 02:44:13.908248 22484 solver.cpp:464] Iteration 679, lr = 1e-05
I1118 02:44:16.143667 22484 solver.cpp:209] Iteration 680, loss = 0.457592
I1118 02:44:16.143697 22484 solver.cpp:464] Iteration 680, lr = 1e-05
I1118 02:44:18.376308 22484 solver.cpp:209] Iteration 681, loss = 0.614986
I1118 02:44:18.376349 22484 solver.cpp:464] Iteration 681, lr = 1e-05
I1118 02:44:20.605252 22484 solver.cpp:209] Iteration 682, loss = 0.336424
I1118 02:44:20.605281 22484 solver.cpp:464] Iteration 682, lr = 1e-05
I1118 02:44:22.829303 22484 solver.cpp:209] Iteration 683, loss = 0.308559
I1118 02:44:22.829332 22484 solver.cpp:464] Iteration 683, lr = 1e-05
I1118 02:44:25.047011 22484 solver.cpp:209] Iteration 684, loss = 0.531211
I1118 02:44:25.047040 22484 solver.cpp:464] Iteration 684, lr = 1e-05
I1118 02:44:27.275864 22484 solver.cpp:209] Iteration 685, loss = 0.471386
I1118 02:44:27.275950 22484 solver.cpp:464] Iteration 685, lr = 1e-05
I1118 02:44:29.513475 22484 solver.cpp:209] Iteration 686, loss = 0.285791
I1118 02:44:29.513504 22484 solver.cpp:464] Iteration 686, lr = 1e-05
I1118 02:44:31.746567 22484 solver.cpp:209] Iteration 687, loss = 0.557609
I1118 02:44:31.746616 22484 solver.cpp:464] Iteration 687, lr = 1e-05
I1118 02:44:33.981015 22484 solver.cpp:209] Iteration 688, loss = 0.275134
I1118 02:44:33.981057 22484 solver.cpp:464] Iteration 688, lr = 1e-05
I1118 02:44:36.203825 22484 solver.cpp:209] Iteration 689, loss = 0.442172
I1118 02:44:36.203855 22484 solver.cpp:464] Iteration 689, lr = 1e-05
I1118 02:44:38.433457 22484 solver.cpp:209] Iteration 690, loss = 0.283536
I1118 02:44:38.433486 22484 solver.cpp:464] Iteration 690, lr = 1e-05
I1118 02:44:40.657683 22484 solver.cpp:209] Iteration 691, loss = 0.479039
I1118 02:44:40.657711 22484 solver.cpp:464] Iteration 691, lr = 1e-05
I1118 02:44:42.893640 22484 solver.cpp:209] Iteration 692, loss = 0.121109
I1118 02:44:42.893668 22484 solver.cpp:464] Iteration 692, lr = 1e-05
I1118 02:44:45.126169 22484 solver.cpp:209] Iteration 693, loss = 0.529415
I1118 02:44:45.126210 22484 solver.cpp:464] Iteration 693, lr = 1e-05
I1118 02:44:47.352807 22484 solver.cpp:209] Iteration 694, loss = 0.108414
I1118 02:44:47.352849 22484 solver.cpp:464] Iteration 694, lr = 1e-05
I1118 02:44:49.587108 22484 solver.cpp:209] Iteration 695, loss = 0.534364
I1118 02:44:49.587137 22484 solver.cpp:464] Iteration 695, lr = 1e-05
I1118 02:44:51.812417 22484 solver.cpp:209] Iteration 696, loss = 0.186144
I1118 02:44:51.812446 22484 solver.cpp:464] Iteration 696, lr = 1e-05
I1118 02:44:54.041159 22484 solver.cpp:209] Iteration 697, loss = 0.219914
I1118 02:44:54.041200 22484 solver.cpp:464] Iteration 697, lr = 1e-05
I1118 02:44:56.266824 22484 solver.cpp:209] Iteration 698, loss = 0.227552
I1118 02:44:56.266854 22484 solver.cpp:464] Iteration 698, lr = 1e-05
I1118 02:44:58.491578 22484 solver.cpp:209] Iteration 699, loss = 0.397076
I1118 02:44:58.491669 22484 solver.cpp:464] Iteration 699, lr = 1e-05
I1118 02:44:58.492255 22484 solver.cpp:264] Iteration 700, Testing net (#0)
I1118 02:45:12.380970 22484 solver.cpp:305] Test loss: 0.280146
I1118 02:45:12.380998 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 02:45:12.381018 22484 solver.cpp:319]            = 0.98646
I1118 02:45:12.381023 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 02:45:12.381027 22484 solver.cpp:318] mean_score = test_score[2] { = 1} / test_score[3] { = 59 }
I1118 02:45:12.381032 22484 solver.cpp:319]            = 0.0169492
I1118 02:45:12.381036 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0169492
I1118 02:45:12.381041 22484 solver.cpp:332]     Test net output #2: accuracy = 0.887153
I1118 02:45:12.381044 22484 solver.cpp:334]     Test net output #3: accuracy = 0.501705
I1118 02:45:13.028095 22484 solver.cpp:209] Iteration 700, loss = 0.224097
I1118 02:45:13.028123 22484 solver.cpp:464] Iteration 700, lr = 1e-05
I1118 02:45:15.264735 22484 solver.cpp:209] Iteration 701, loss = 0.292804
I1118 02:45:15.264765 22484 solver.cpp:464] Iteration 701, lr = 1e-05
I1118 02:45:17.492952 22484 solver.cpp:209] Iteration 702, loss = 0.590326
I1118 02:45:17.492982 22484 solver.cpp:464] Iteration 702, lr = 1e-05
I1118 02:45:19.725392 22484 solver.cpp:209] Iteration 703, loss = 0.506528
I1118 02:45:19.725432 22484 solver.cpp:464] Iteration 703, lr = 1e-05
I1118 02:45:21.955071 22484 solver.cpp:209] Iteration 704, loss = 0.172957
I1118 02:45:21.955101 22484 solver.cpp:464] Iteration 704, lr = 1e-05
I1118 02:45:24.178777 22484 solver.cpp:209] Iteration 705, loss = 0.0798835
I1118 02:45:24.178808 22484 solver.cpp:464] Iteration 705, lr = 1e-05
I1118 02:45:26.408666 22484 solver.cpp:209] Iteration 706, loss = 0.0984778
I1118 02:45:26.408697 22484 solver.cpp:464] Iteration 706, lr = 1e-05
I1118 02:45:28.640038 22484 solver.cpp:209] Iteration 707, loss = 0.3228
I1118 02:45:28.640125 22484 solver.cpp:464] Iteration 707, lr = 1e-05
I1118 02:45:30.879031 22484 solver.cpp:209] Iteration 708, loss = 0.244287
I1118 02:45:30.879070 22484 solver.cpp:464] Iteration 708, lr = 1e-05
I1118 02:45:33.110276 22484 solver.cpp:209] Iteration 709, loss = 0.34138
I1118 02:45:33.110317 22484 solver.cpp:464] Iteration 709, lr = 1e-05
I1118 02:45:35.341117 22484 solver.cpp:209] Iteration 710, loss = 0.382529
I1118 02:45:35.341158 22484 solver.cpp:464] Iteration 710, lr = 1e-05
I1118 02:45:37.567939 22484 solver.cpp:209] Iteration 711, loss = 0.250103
I1118 02:45:37.567970 22484 solver.cpp:464] Iteration 711, lr = 1e-05
I1118 02:45:39.786813 22484 solver.cpp:209] Iteration 712, loss = 0.261553
I1118 02:45:39.786842 22484 solver.cpp:464] Iteration 712, lr = 1e-05
I1118 02:45:42.013838 22484 solver.cpp:209] Iteration 713, loss = 0.327923
I1118 02:45:42.013878 22484 solver.cpp:464] Iteration 713, lr = 1e-05
I1118 02:45:44.240161 22484 solver.cpp:209] Iteration 714, loss = 0.387908
I1118 02:45:44.240190 22484 solver.cpp:464] Iteration 714, lr = 1e-05
I1118 02:45:46.496469 22484 solver.cpp:209] Iteration 715, loss = 0.304962
I1118 02:45:46.496498 22484 solver.cpp:464] Iteration 715, lr = 1e-05
I1118 02:45:48.732108 22484 solver.cpp:209] Iteration 716, loss = 0.169315
I1118 02:45:48.732151 22484 solver.cpp:464] Iteration 716, lr = 1e-05
I1118 02:45:50.967432 22484 solver.cpp:209] Iteration 717, loss = 0.463911
I1118 02:45:50.967459 22484 solver.cpp:464] Iteration 717, lr = 1e-05
I1118 02:45:53.194458 22484 solver.cpp:209] Iteration 718, loss = 0.174892
I1118 02:45:53.194488 22484 solver.cpp:464] Iteration 718, lr = 1e-05
I1118 02:45:55.420578 22484 solver.cpp:209] Iteration 719, loss = 0.222463
I1118 02:45:55.420606 22484 solver.cpp:464] Iteration 719, lr = 1e-05
I1118 02:45:57.645650 22484 solver.cpp:209] Iteration 720, loss = 0.287338
I1118 02:45:57.645678 22484 solver.cpp:464] Iteration 720, lr = 1e-05
I1118 02:45:59.874294 22484 solver.cpp:209] Iteration 721, loss = 0.299522
I1118 02:45:59.874387 22484 solver.cpp:464] Iteration 721, lr = 1e-05
I1118 02:46:02.112633 22484 solver.cpp:209] Iteration 722, loss = 0.086736
I1118 02:46:02.112663 22484 solver.cpp:464] Iteration 722, lr = 1e-05
I1118 02:46:04.341151 22484 solver.cpp:209] Iteration 723, loss = 0.404363
I1118 02:46:04.341181 22484 solver.cpp:464] Iteration 723, lr = 1e-05
I1118 02:46:06.579435 22484 solver.cpp:209] Iteration 724, loss = 0.33619
I1118 02:46:06.579478 22484 solver.cpp:464] Iteration 724, lr = 1e-05
I1118 02:46:08.801136 22484 solver.cpp:209] Iteration 725, loss = 0.194959
I1118 02:46:08.801167 22484 solver.cpp:464] Iteration 725, lr = 1e-05
I1118 02:46:11.031568 22484 solver.cpp:209] Iteration 726, loss = 0.390776
I1118 02:46:11.031605 22484 solver.cpp:464] Iteration 726, lr = 1e-05
I1118 02:46:13.257791 22484 solver.cpp:209] Iteration 727, loss = 0.297165
I1118 02:46:13.257822 22484 solver.cpp:464] Iteration 727, lr = 1e-05
I1118 02:46:15.479022 22484 solver.cpp:209] Iteration 728, loss = 0.216973
I1118 02:46:15.479064 22484 solver.cpp:464] Iteration 728, lr = 1e-05
I1118 02:46:17.713970 22484 solver.cpp:209] Iteration 729, loss = 0.131492
I1118 02:46:17.714011 22484 solver.cpp:464] Iteration 729, lr = 1e-05
I1118 02:46:19.943182 22484 solver.cpp:209] Iteration 730, loss = 0.263668
I1118 02:46:19.943212 22484 solver.cpp:464] Iteration 730, lr = 1e-05
I1118 02:46:22.178222 22484 solver.cpp:209] Iteration 731, loss = 0.316503
I1118 02:46:22.178251 22484 solver.cpp:464] Iteration 731, lr = 1e-05
I1118 02:46:24.403343 22484 solver.cpp:209] Iteration 732, loss = 0.294409
I1118 02:46:24.403383 22484 solver.cpp:464] Iteration 732, lr = 1e-05
I1118 02:46:26.627109 22484 solver.cpp:209] Iteration 733, loss = 0.334336
I1118 02:46:26.627137 22484 solver.cpp:464] Iteration 733, lr = 1e-05
I1118 02:46:28.851690 22484 solver.cpp:209] Iteration 734, loss = 0.133865
I1118 02:46:28.851719 22484 solver.cpp:464] Iteration 734, lr = 1e-05
I1118 02:46:31.070716 22484 solver.cpp:209] Iteration 735, loss = 0.393967
I1118 02:46:31.070780 22484 solver.cpp:464] Iteration 735, lr = 1e-05
I1118 02:46:33.311089 22484 solver.cpp:209] Iteration 736, loss = 0.106679
I1118 02:46:33.311130 22484 solver.cpp:464] Iteration 736, lr = 1e-05
I1118 02:46:35.541502 22484 solver.cpp:209] Iteration 737, loss = 0.128976
I1118 02:46:35.541533 22484 solver.cpp:464] Iteration 737, lr = 1e-05
I1118 02:46:37.774646 22484 solver.cpp:209] Iteration 738, loss = 0.151415
I1118 02:46:37.774675 22484 solver.cpp:464] Iteration 738, lr = 1e-05
I1118 02:46:39.998946 22484 solver.cpp:209] Iteration 739, loss = 0.262087
I1118 02:46:39.998975 22484 solver.cpp:464] Iteration 739, lr = 1e-05
I1118 02:46:42.220218 22484 solver.cpp:209] Iteration 740, loss = 0.0567653
I1118 02:46:42.220260 22484 solver.cpp:464] Iteration 740, lr = 1e-05
I1118 02:46:44.446811 22484 solver.cpp:209] Iteration 741, loss = 0.270102
I1118 02:46:44.446842 22484 solver.cpp:464] Iteration 741, lr = 1e-05
I1118 02:46:46.676726 22484 solver.cpp:209] Iteration 742, loss = 0.242032
I1118 02:46:46.676756 22484 solver.cpp:464] Iteration 742, lr = 1e-05
I1118 02:46:48.908162 22484 solver.cpp:209] Iteration 743, loss = 0.22516
I1118 02:46:48.908205 22484 solver.cpp:464] Iteration 743, lr = 1e-05
I1118 02:46:51.143400 22484 solver.cpp:209] Iteration 744, loss = 0.106935
I1118 02:46:51.143439 22484 solver.cpp:464] Iteration 744, lr = 1e-05
I1118 02:46:53.370012 22484 solver.cpp:209] Iteration 745, loss = 0.22237
I1118 02:46:53.370043 22484 solver.cpp:464] Iteration 745, lr = 1e-05
I1118 02:46:55.600404 22484 solver.cpp:209] Iteration 746, loss = 0.288353
I1118 02:46:55.600445 22484 solver.cpp:464] Iteration 746, lr = 1e-05
I1118 02:46:57.828445 22484 solver.cpp:209] Iteration 747, loss = 0.398844
I1118 02:46:57.828486 22484 solver.cpp:464] Iteration 747, lr = 1e-05
I1118 02:47:00.051370 22484 solver.cpp:209] Iteration 748, loss = 0.0650761
I1118 02:47:00.051399 22484 solver.cpp:464] Iteration 748, lr = 1e-05
I1118 02:47:02.278251 22484 solver.cpp:209] Iteration 749, loss = 0.381525
I1118 02:47:02.278317 22484 solver.cpp:464] Iteration 749, lr = 1e-05
I1118 02:47:02.278913 22484 solver.cpp:264] Iteration 750, Testing net (#0)
I1118 02:47:16.025725 22484 solver.cpp:305] Test loss: 0.270771
I1118 02:47:16.025768 22484 solver.cpp:318] mean_score = test_score[0] { = 513} / test_score[1] { = 517 }
I1118 02:47:16.025774 22484 solver.cpp:319]            = 0.992263
I1118 02:47:16.025779 22484 solver.cpp:328]     Test net output #0: accuracy = 0.992263
I1118 02:47:16.025784 22484 solver.cpp:318] mean_score = test_score[2] { = 3} / test_score[3] { = 59 }
I1118 02:47:16.025789 22484 solver.cpp:319]            = 0.0508475
I1118 02:47:16.025792 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0508475
I1118 02:47:16.025797 22484 solver.cpp:332]     Test net output #2: accuracy = 0.895833
I1118 02:47:16.025801 22484 solver.cpp:334]     Test net output #3: accuracy = 0.521555
I1118 02:47:16.671402 22484 solver.cpp:209] Iteration 750, loss = 0.466089
I1118 02:47:16.671432 22484 solver.cpp:464] Iteration 750, lr = 1e-05
I1118 02:47:18.909214 22484 solver.cpp:209] Iteration 751, loss = 0.15556
I1118 02:47:18.909256 22484 solver.cpp:464] Iteration 751, lr = 1e-05
I1118 02:47:21.162385 22484 solver.cpp:209] Iteration 752, loss = 0.314987
I1118 02:47:21.162422 22484 solver.cpp:464] Iteration 752, lr = 1e-05
I1118 02:47:23.404527 22484 solver.cpp:209] Iteration 753, loss = 0.37131
I1118 02:47:23.404569 22484 solver.cpp:464] Iteration 753, lr = 1e-05
I1118 02:47:25.645095 22484 solver.cpp:209] Iteration 754, loss = 0.229293
I1118 02:47:25.645134 22484 solver.cpp:464] Iteration 754, lr = 1e-05
I1118 02:47:27.891996 22484 solver.cpp:209] Iteration 755, loss = 0.249737
I1118 02:47:27.892025 22484 solver.cpp:464] Iteration 755, lr = 1e-05
I1118 02:47:30.133396 22484 solver.cpp:209] Iteration 756, loss = 0.284148
I1118 02:47:30.133425 22484 solver.cpp:464] Iteration 756, lr = 1e-05
I1118 02:47:32.382935 22484 solver.cpp:209] Iteration 757, loss = 0.379076
I1118 02:47:32.382987 22484 solver.cpp:464] Iteration 757, lr = 1e-05
I1118 02:47:34.631006 22484 solver.cpp:209] Iteration 758, loss = 0.490881
I1118 02:47:34.631036 22484 solver.cpp:464] Iteration 758, lr = 1e-05
I1118 02:47:36.877173 22484 solver.cpp:209] Iteration 759, loss = 0.399226
I1118 02:47:36.877204 22484 solver.cpp:464] Iteration 759, lr = 1e-05
I1118 02:47:39.138985 22484 solver.cpp:209] Iteration 760, loss = 0.132556
I1118 02:47:39.139025 22484 solver.cpp:464] Iteration 760, lr = 1e-05
I1118 02:47:41.331455 22484 solver.cpp:209] Iteration 761, loss = 0.239998
I1118 02:47:41.331497 22484 solver.cpp:464] Iteration 761, lr = 1e-05
I1118 02:47:43.525926 22484 solver.cpp:209] Iteration 762, loss = 0.312708
I1118 02:47:43.525967 22484 solver.cpp:464] Iteration 762, lr = 1e-05
I1118 02:47:45.714532 22484 solver.cpp:209] Iteration 763, loss = 0.159266
I1118 02:47:45.714561 22484 solver.cpp:464] Iteration 763, lr = 1e-05
I1118 02:47:47.906460 22484 solver.cpp:209] Iteration 764, loss = 0.227466
I1118 02:47:47.906501 22484 solver.cpp:464] Iteration 764, lr = 1e-05
I1118 02:47:50.104426 22484 solver.cpp:209] Iteration 765, loss = 0.165296
I1118 02:47:50.104467 22484 solver.cpp:464] Iteration 765, lr = 1e-05
I1118 02:47:52.299160 22484 solver.cpp:209] Iteration 766, loss = 0.131086
I1118 02:47:52.299201 22484 solver.cpp:464] Iteration 766, lr = 1e-05
I1118 02:47:54.492420 22484 solver.cpp:209] Iteration 767, loss = 0.36111
I1118 02:47:54.492460 22484 solver.cpp:464] Iteration 767, lr = 1e-05
I1118 02:47:56.687249 22484 solver.cpp:209] Iteration 768, loss = 0.439331
I1118 02:47:56.687278 22484 solver.cpp:464] Iteration 768, lr = 1e-05
I1118 02:47:58.878530 22484 solver.cpp:209] Iteration 769, loss = 0.361985
I1118 02:47:58.878572 22484 solver.cpp:464] Iteration 769, lr = 1e-05
I1118 02:48:01.085779 22484 solver.cpp:209] Iteration 770, loss = 0.436468
I1118 02:48:01.085805 22484 solver.cpp:464] Iteration 770, lr = 1e-05
I1118 02:48:03.279119 22484 solver.cpp:209] Iteration 771, loss = 0.19699
I1118 02:48:03.279211 22484 solver.cpp:464] Iteration 771, lr = 1e-05
I1118 02:48:05.501195 22484 solver.cpp:209] Iteration 772, loss = 0.385898
I1118 02:48:05.501237 22484 solver.cpp:464] Iteration 772, lr = 1e-05
I1118 02:48:07.725582 22484 solver.cpp:209] Iteration 773, loss = 0.21923
I1118 02:48:07.725612 22484 solver.cpp:464] Iteration 773, lr = 1e-05
I1118 02:48:09.947504 22484 solver.cpp:209] Iteration 774, loss = 0.15848
I1118 02:48:09.947545 22484 solver.cpp:464] Iteration 774, lr = 1e-05
I1118 02:48:12.176372 22484 solver.cpp:209] Iteration 775, loss = 0.363203
I1118 02:48:12.176400 22484 solver.cpp:464] Iteration 775, lr = 1e-05
I1118 02:48:14.402442 22484 solver.cpp:209] Iteration 776, loss = 0.269187
I1118 02:48:14.402472 22484 solver.cpp:464] Iteration 776, lr = 1e-05
I1118 02:48:16.632760 22484 solver.cpp:209] Iteration 777, loss = 0.279504
I1118 02:48:16.632802 22484 solver.cpp:464] Iteration 777, lr = 1e-05
I1118 02:48:18.866219 22484 solver.cpp:209] Iteration 778, loss = 0.191293
I1118 02:48:18.866261 22484 solver.cpp:464] Iteration 778, lr = 1e-05
I1118 02:48:21.093679 22484 solver.cpp:209] Iteration 779, loss = 0.284239
I1118 02:48:21.093706 22484 solver.cpp:464] Iteration 779, lr = 1e-05
I1118 02:48:23.319588 22484 solver.cpp:209] Iteration 780, loss = 0.351695
I1118 02:48:23.319617 22484 solver.cpp:464] Iteration 780, lr = 1e-05
I1118 02:48:25.541539 22484 solver.cpp:209] Iteration 781, loss = 0.348576
I1118 02:48:25.541568 22484 solver.cpp:464] Iteration 781, lr = 1e-05
I1118 02:48:27.771553 22484 solver.cpp:209] Iteration 782, loss = 0.455917
I1118 02:48:27.771582 22484 solver.cpp:464] Iteration 782, lr = 1e-05
I1118 02:48:29.999661 22484 solver.cpp:209] Iteration 783, loss = 0.444347
I1118 02:48:29.999691 22484 solver.cpp:464] Iteration 783, lr = 1e-05
I1118 02:48:32.233651 22484 solver.cpp:209] Iteration 784, loss = 0.507595
I1118 02:48:32.233693 22484 solver.cpp:464] Iteration 784, lr = 1e-05
I1118 02:48:34.468950 22484 solver.cpp:209] Iteration 785, loss = 0.328538
I1118 02:48:34.469038 22484 solver.cpp:464] Iteration 785, lr = 1e-05
I1118 02:48:36.694785 22484 solver.cpp:209] Iteration 786, loss = 0.332423
I1118 02:48:36.694815 22484 solver.cpp:464] Iteration 786, lr = 1e-05
I1118 02:48:38.925307 22484 solver.cpp:209] Iteration 787, loss = 0.408012
I1118 02:48:38.925336 22484 solver.cpp:464] Iteration 787, lr = 1e-05
I1118 02:48:41.147029 22484 solver.cpp:209] Iteration 788, loss = 0.260221
I1118 02:48:41.147066 22484 solver.cpp:464] Iteration 788, lr = 1e-05
I1118 02:48:43.376718 22484 solver.cpp:209] Iteration 789, loss = 0.350452
I1118 02:48:43.376759 22484 solver.cpp:464] Iteration 789, lr = 1e-05
I1118 02:48:45.607975 22484 solver.cpp:209] Iteration 790, loss = 0.410265
I1118 02:48:45.608016 22484 solver.cpp:464] Iteration 790, lr = 1e-05
I1118 02:48:47.837563 22484 solver.cpp:209] Iteration 791, loss = 0.354039
I1118 02:48:47.837605 22484 solver.cpp:464] Iteration 791, lr = 1e-05
I1118 02:48:50.072623 22484 solver.cpp:209] Iteration 792, loss = 0.342064
I1118 02:48:50.072664 22484 solver.cpp:464] Iteration 792, lr = 1e-05
I1118 02:48:52.298946 22484 solver.cpp:209] Iteration 793, loss = 0.304427
I1118 02:48:52.298974 22484 solver.cpp:464] Iteration 793, lr = 1e-05
I1118 02:48:54.524863 22484 solver.cpp:209] Iteration 794, loss = 0.327539
I1118 02:48:54.524893 22484 solver.cpp:464] Iteration 794, lr = 1e-05
I1118 02:48:56.748555 22484 solver.cpp:209] Iteration 795, loss = 0.160993
I1118 02:48:56.748596 22484 solver.cpp:464] Iteration 795, lr = 1e-05
I1118 02:48:58.972435 22484 solver.cpp:209] Iteration 796, loss = 0.338676
I1118 02:48:58.972476 22484 solver.cpp:464] Iteration 796, lr = 1e-05
I1118 02:49:01.209553 22484 solver.cpp:209] Iteration 797, loss = 0.234655
I1118 02:49:01.209579 22484 solver.cpp:464] Iteration 797, lr = 1e-05
I1118 02:49:03.439136 22484 solver.cpp:209] Iteration 798, loss = 0.477148
I1118 02:49:03.439165 22484 solver.cpp:464] Iteration 798, lr = 1e-05
I1118 02:49:05.668602 22484 solver.cpp:209] Iteration 799, loss = 0.166305
I1118 02:49:05.668695 22484 solver.cpp:464] Iteration 799, lr = 1e-05
I1118 02:49:05.669306 22484 solver.cpp:264] Iteration 800, Testing net (#0)
I1118 02:49:19.550776 22484 solver.cpp:305] Test loss: 0.265837
I1118 02:49:19.550806 22484 solver.cpp:318] mean_score = test_score[0] { = 509} / test_score[1] { = 517 }
I1118 02:49:19.550812 22484 solver.cpp:319]            = 0.984526
I1118 02:49:19.550817 22484 solver.cpp:328]     Test net output #0: accuracy = 0.984526
I1118 02:49:19.550822 22484 solver.cpp:318] mean_score = test_score[2] { = 7} / test_score[3] { = 59 }
I1118 02:49:19.550827 22484 solver.cpp:319]            = 0.118644
I1118 02:49:19.550830 22484 solver.cpp:328]     Test net output #1: accuracy = 0.118644
I1118 02:49:19.550834 22484 solver.cpp:332]     Test net output #2: accuracy = 0.895833
I1118 02:49:19.550838 22484 solver.cpp:334]     Test net output #3: accuracy = 0.551585
I1118 02:49:20.198407 22484 solver.cpp:209] Iteration 800, loss = 0.220664
I1118 02:49:20.198448 22484 solver.cpp:464] Iteration 800, lr = 1e-05
I1118 02:49:22.441462 22484 solver.cpp:209] Iteration 801, loss = 0.149811
I1118 02:49:22.441494 22484 solver.cpp:464] Iteration 801, lr = 1e-05
I1118 02:49:24.664369 22484 solver.cpp:209] Iteration 802, loss = 0.298552
I1118 02:49:24.664398 22484 solver.cpp:464] Iteration 802, lr = 1e-05
I1118 02:49:26.887251 22484 solver.cpp:209] Iteration 803, loss = 0.223546
I1118 02:49:26.887292 22484 solver.cpp:464] Iteration 803, lr = 1e-05
I1118 02:49:29.112197 22484 solver.cpp:209] Iteration 804, loss = 0.531023
I1118 02:49:29.112241 22484 solver.cpp:464] Iteration 804, lr = 1e-05
I1118 02:49:31.340190 22484 solver.cpp:209] Iteration 805, loss = 0.432576
I1118 02:49:31.340219 22484 solver.cpp:464] Iteration 805, lr = 1e-05
I1118 02:49:33.578133 22484 solver.cpp:209] Iteration 806, loss = 0.45667
I1118 02:49:33.578174 22484 solver.cpp:464] Iteration 806, lr = 1e-05
I1118 02:49:35.806164 22484 solver.cpp:209] Iteration 807, loss = 0.147004
I1118 02:49:35.806216 22484 solver.cpp:464] Iteration 807, lr = 1e-05
I1118 02:49:38.035411 22484 solver.cpp:209] Iteration 808, loss = 0.131756
I1118 02:49:38.035439 22484 solver.cpp:464] Iteration 808, lr = 1e-05
I1118 02:49:40.262148 22484 solver.cpp:209] Iteration 809, loss = 0.130642
I1118 02:49:40.262190 22484 solver.cpp:464] Iteration 809, lr = 1e-05
I1118 02:49:42.488354 22484 solver.cpp:209] Iteration 810, loss = 0.470571
I1118 02:49:42.488396 22484 solver.cpp:464] Iteration 810, lr = 1e-05
I1118 02:49:44.714862 22484 solver.cpp:209] Iteration 811, loss = 0.186858
I1118 02:49:44.714905 22484 solver.cpp:464] Iteration 811, lr = 1e-05
I1118 02:49:46.936985 22484 solver.cpp:209] Iteration 812, loss = 0.631425
I1118 02:49:46.937016 22484 solver.cpp:464] Iteration 812, lr = 1e-05
I1118 02:49:49.167510 22484 solver.cpp:209] Iteration 813, loss = 0.211503
I1118 02:49:49.167539 22484 solver.cpp:464] Iteration 813, lr = 1e-05
I1118 02:49:51.397339 22484 solver.cpp:209] Iteration 814, loss = 0.380765
I1118 02:49:51.397382 22484 solver.cpp:464] Iteration 814, lr = 1e-05
I1118 02:49:53.629340 22484 solver.cpp:209] Iteration 815, loss = 0.277584
I1118 02:49:53.629370 22484 solver.cpp:464] Iteration 815, lr = 1e-05
I1118 02:49:55.854495 22484 solver.cpp:209] Iteration 816, loss = 0.307638
I1118 02:49:55.854523 22484 solver.cpp:464] Iteration 816, lr = 1e-05
I1118 02:49:58.087795 22484 solver.cpp:209] Iteration 817, loss = 0.343481
I1118 02:49:58.087823 22484 solver.cpp:464] Iteration 817, lr = 1e-05
I1118 02:50:00.312127 22484 solver.cpp:209] Iteration 818, loss = 0.313638
I1118 02:50:00.312156 22484 solver.cpp:464] Iteration 818, lr = 1e-05
I1118 02:50:02.536259 22484 solver.cpp:209] Iteration 819, loss = 0.273998
I1118 02:50:02.536301 22484 solver.cpp:464] Iteration 819, lr = 1e-05
I1118 02:50:04.768843 22484 solver.cpp:209] Iteration 820, loss = 0.20764
I1118 02:50:04.768885 22484 solver.cpp:464] Iteration 820, lr = 1e-05
I1118 02:50:07.001168 22484 solver.cpp:209] Iteration 821, loss = 0.142716
I1118 02:50:07.001235 22484 solver.cpp:464] Iteration 821, lr = 1e-05
I1118 02:50:09.236330 22484 solver.cpp:209] Iteration 822, loss = 0.253836
I1118 02:50:09.236371 22484 solver.cpp:464] Iteration 822, lr = 1e-05
I1118 02:50:11.459993 22484 solver.cpp:209] Iteration 823, loss = 0.229707
I1118 02:50:11.460034 22484 solver.cpp:464] Iteration 823, lr = 1e-05
I1118 02:50:13.680893 22484 solver.cpp:209] Iteration 824, loss = 0.280473
I1118 02:50:13.680922 22484 solver.cpp:464] Iteration 824, lr = 1e-05
I1118 02:50:15.906566 22484 solver.cpp:209] Iteration 825, loss = 0.278986
I1118 02:50:15.906615 22484 solver.cpp:464] Iteration 825, lr = 1e-05
I1118 02:50:18.131971 22484 solver.cpp:209] Iteration 826, loss = 0.319457
I1118 02:50:18.132011 22484 solver.cpp:464] Iteration 826, lr = 1e-05
I1118 02:50:20.368212 22484 solver.cpp:209] Iteration 827, loss = 0.264919
I1118 02:50:20.368242 22484 solver.cpp:464] Iteration 827, lr = 1e-05
I1118 02:50:22.595260 22484 solver.cpp:209] Iteration 828, loss = 0.278831
I1118 02:50:22.595290 22484 solver.cpp:464] Iteration 828, lr = 1e-05
I1118 02:50:24.816531 22484 solver.cpp:209] Iteration 829, loss = 0.289566
I1118 02:50:24.816562 22484 solver.cpp:464] Iteration 829, lr = 1e-05
I1118 02:50:27.040556 22484 solver.cpp:209] Iteration 830, loss = 0.345891
I1118 02:50:27.040585 22484 solver.cpp:464] Iteration 830, lr = 1e-05
I1118 02:50:29.269608 22484 solver.cpp:209] Iteration 831, loss = 0.134575
I1118 02:50:29.269637 22484 solver.cpp:464] Iteration 831, lr = 1e-05
I1118 02:50:31.500509 22484 solver.cpp:209] Iteration 832, loss = 0.264763
I1118 02:50:31.500550 22484 solver.cpp:464] Iteration 832, lr = 1e-05
I1118 02:50:33.730420 22484 solver.cpp:209] Iteration 833, loss = 0.136962
I1118 02:50:33.730450 22484 solver.cpp:464] Iteration 833, lr = 1e-05
I1118 02:50:35.963955 22484 solver.cpp:209] Iteration 834, loss = 0.354272
I1118 02:50:35.963999 22484 solver.cpp:464] Iteration 834, lr = 1e-05
I1118 02:50:38.197672 22484 solver.cpp:209] Iteration 835, loss = 0.206177
I1118 02:50:38.197724 22484 solver.cpp:464] Iteration 835, lr = 1e-05
I1118 02:50:40.423151 22484 solver.cpp:209] Iteration 836, loss = 0.474429
I1118 02:50:40.423192 22484 solver.cpp:464] Iteration 836, lr = 1e-05
I1118 02:50:42.649377 22484 solver.cpp:209] Iteration 837, loss = 0.109346
I1118 02:50:42.649406 22484 solver.cpp:464] Iteration 837, lr = 1e-05
I1118 02:50:44.870535 22484 solver.cpp:209] Iteration 838, loss = 0.470794
I1118 02:50:44.870565 22484 solver.cpp:464] Iteration 838, lr = 1e-05
I1118 02:50:47.090625 22484 solver.cpp:209] Iteration 839, loss = 0.0632398
I1118 02:50:47.090656 22484 solver.cpp:464] Iteration 839, lr = 1e-05
I1118 02:50:49.295349 22484 solver.cpp:209] Iteration 840, loss = 0.155448
I1118 02:50:49.295377 22484 solver.cpp:464] Iteration 840, lr = 1e-05
I1118 02:50:51.491961 22484 solver.cpp:209] Iteration 841, loss = 0.168836
I1118 02:50:51.491989 22484 solver.cpp:464] Iteration 841, lr = 1e-05
I1118 02:50:53.691411 22484 solver.cpp:209] Iteration 842, loss = 0.204515
I1118 02:50:53.691440 22484 solver.cpp:464] Iteration 842, lr = 1e-05
I1118 02:50:55.875072 22484 solver.cpp:209] Iteration 843, loss = 0.210602
I1118 02:50:55.875102 22484 solver.cpp:464] Iteration 843, lr = 1e-05
I1118 02:50:58.068521 22484 solver.cpp:209] Iteration 844, loss = 0.299158
I1118 02:50:58.068562 22484 solver.cpp:464] Iteration 844, lr = 1e-05
I1118 02:51:00.266868 22484 solver.cpp:209] Iteration 845, loss = 0.227347
I1118 02:51:00.266911 22484 solver.cpp:464] Iteration 845, lr = 1e-05
I1118 02:51:02.460943 22484 solver.cpp:209] Iteration 846, loss = 0.208457
I1118 02:51:02.460984 22484 solver.cpp:464] Iteration 846, lr = 1e-05
I1118 02:51:04.665949 22484 solver.cpp:209] Iteration 847, loss = 0.132126
I1118 02:51:04.665978 22484 solver.cpp:464] Iteration 847, lr = 1e-05
I1118 02:51:06.852382 22484 solver.cpp:209] Iteration 848, loss = 0.222485
I1118 02:51:06.852423 22484 solver.cpp:464] Iteration 848, lr = 1e-05
I1118 02:51:09.041774 22484 solver.cpp:209] Iteration 849, loss = 0.437431
I1118 02:51:09.041828 22484 solver.cpp:464] Iteration 849, lr = 1e-05
I1118 02:51:09.042423 22484 solver.cpp:264] Iteration 850, Testing net (#0)
I1118 02:51:22.945938 22484 solver.cpp:305] Test loss: 0.268168
I1118 02:51:22.945968 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 02:51:22.945974 22484 solver.cpp:319]            = 0.98646
I1118 02:51:22.945979 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 02:51:22.945983 22484 solver.cpp:318] mean_score = test_score[2] { = 4} / test_score[3] { = 59 }
I1118 02:51:22.945988 22484 solver.cpp:319]            = 0.0677966
I1118 02:51:22.945992 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0677966
I1118 02:51:22.945996 22484 solver.cpp:332]     Test net output #2: accuracy = 0.892361
I1118 02:51:22.946001 22484 solver.cpp:334]     Test net output #3: accuracy = 0.527128
I1118 02:51:23.605588 22484 solver.cpp:209] Iteration 850, loss = 0.220426
I1118 02:51:23.605629 22484 solver.cpp:464] Iteration 850, lr = 1e-05
I1118 02:51:25.873693 22484 solver.cpp:209] Iteration 851, loss = 0.0701878
I1118 02:51:25.873723 22484 solver.cpp:464] Iteration 851, lr = 1e-05
I1118 02:51:28.150650 22484 solver.cpp:209] Iteration 852, loss = 0.575209
I1118 02:51:28.150679 22484 solver.cpp:464] Iteration 852, lr = 1e-05
I1118 02:51:30.423532 22484 solver.cpp:209] Iteration 853, loss = 0.436574
I1118 02:51:30.423562 22484 solver.cpp:464] Iteration 853, lr = 1e-05
I1118 02:51:32.701853 22484 solver.cpp:209] Iteration 854, loss = 0.0835326
I1118 02:51:32.701884 22484 solver.cpp:464] Iteration 854, lr = 1e-05
I1118 02:51:34.946072 22484 solver.cpp:209] Iteration 855, loss = 0.258263
I1118 02:51:34.946102 22484 solver.cpp:464] Iteration 855, lr = 1e-05
I1118 02:51:37.173292 22484 solver.cpp:209] Iteration 856, loss = 0.29244
I1118 02:51:37.173333 22484 solver.cpp:464] Iteration 856, lr = 1e-05
I1118 02:51:39.376673 22484 solver.cpp:209] Iteration 857, loss = 0.354302
I1118 02:51:39.376739 22484 solver.cpp:464] Iteration 857, lr = 1e-05
I1118 02:51:41.568166 22484 solver.cpp:209] Iteration 858, loss = 0.162569
I1118 02:51:41.568197 22484 solver.cpp:464] Iteration 858, lr = 1e-05
I1118 02:51:43.761236 22484 solver.cpp:209] Iteration 859, loss = 0.286483
I1118 02:51:43.761277 22484 solver.cpp:464] Iteration 859, lr = 1e-05
I1118 02:51:45.951092 22484 solver.cpp:209] Iteration 860, loss = 0.29423
I1118 02:51:45.951134 22484 solver.cpp:464] Iteration 860, lr = 1e-05
I1118 02:51:48.144296 22484 solver.cpp:209] Iteration 861, loss = 0.557806
I1118 02:51:48.144325 22484 solver.cpp:464] Iteration 861, lr = 1e-05
I1118 02:51:50.343595 22484 solver.cpp:209] Iteration 862, loss = 0.201639
I1118 02:51:50.343624 22484 solver.cpp:464] Iteration 862, lr = 1e-05
I1118 02:51:52.537957 22484 solver.cpp:209] Iteration 863, loss = 0.0902678
I1118 02:51:52.537987 22484 solver.cpp:464] Iteration 863, lr = 1e-05
I1118 02:51:54.730415 22484 solver.cpp:209] Iteration 864, loss = 0.242173
I1118 02:51:54.730444 22484 solver.cpp:464] Iteration 864, lr = 1e-05
I1118 02:51:56.917160 22484 solver.cpp:209] Iteration 865, loss = 0.195511
I1118 02:51:56.917201 22484 solver.cpp:464] Iteration 865, lr = 1e-05
I1118 02:51:59.113418 22484 solver.cpp:209] Iteration 866, loss = 0.30178
I1118 02:51:59.113461 22484 solver.cpp:464] Iteration 866, lr = 1e-05
I1118 02:52:01.308084 22484 solver.cpp:209] Iteration 867, loss = 0.200823
I1118 02:52:01.308115 22484 solver.cpp:464] Iteration 867, lr = 1e-05
I1118 02:52:03.526410 22484 solver.cpp:209] Iteration 868, loss = 0.158294
I1118 02:52:03.526440 22484 solver.cpp:464] Iteration 868, lr = 1e-05
I1118 02:52:05.720899 22484 solver.cpp:209] Iteration 869, loss = 0.155365
I1118 02:52:05.720940 22484 solver.cpp:464] Iteration 869, lr = 1e-05
I1118 02:52:07.919663 22484 solver.cpp:209] Iteration 870, loss = 0.328364
I1118 02:52:07.919692 22484 solver.cpp:464] Iteration 870, lr = 1e-05
I1118 02:52:10.142479 22484 solver.cpp:209] Iteration 871, loss = 0.488938
I1118 02:52:10.142546 22484 solver.cpp:464] Iteration 871, lr = 1e-05
I1118 02:52:12.392686 22484 solver.cpp:209] Iteration 872, loss = 0.300237
I1118 02:52:12.392729 22484 solver.cpp:464] Iteration 872, lr = 1e-05
I1118 02:52:14.627672 22484 solver.cpp:209] Iteration 873, loss = 0.326726
I1118 02:52:14.627713 22484 solver.cpp:464] Iteration 873, lr = 1e-05
I1118 02:52:16.859493 22484 solver.cpp:209] Iteration 874, loss = 0.237393
I1118 02:52:16.859522 22484 solver.cpp:464] Iteration 874, lr = 1e-05
I1118 02:52:19.085949 22484 solver.cpp:209] Iteration 875, loss = 0.285614
I1118 02:52:19.085993 22484 solver.cpp:464] Iteration 875, lr = 1e-05
I1118 02:52:21.308859 22484 solver.cpp:209] Iteration 876, loss = 0.166925
I1118 02:52:21.308902 22484 solver.cpp:464] Iteration 876, lr = 1e-05
I1118 02:52:23.539011 22484 solver.cpp:209] Iteration 877, loss = 0.168064
I1118 02:52:23.539041 22484 solver.cpp:464] Iteration 877, lr = 1e-05
I1118 02:52:25.765326 22484 solver.cpp:209] Iteration 878, loss = 0.413199
I1118 02:52:25.765367 22484 solver.cpp:464] Iteration 878, lr = 1e-05
I1118 02:52:27.996639 22484 solver.cpp:209] Iteration 879, loss = 0.278565
I1118 02:52:27.996669 22484 solver.cpp:464] Iteration 879, lr = 1e-05
I1118 02:52:30.222847 22484 solver.cpp:209] Iteration 880, loss = 0.211371
I1118 02:52:30.222877 22484 solver.cpp:464] Iteration 880, lr = 1e-05
I1118 02:52:32.448290 22484 solver.cpp:209] Iteration 881, loss = 0.237159
I1118 02:52:32.448319 22484 solver.cpp:464] Iteration 881, lr = 1e-05
I1118 02:52:34.682663 22484 solver.cpp:209] Iteration 882, loss = 0.26029
I1118 02:52:34.682693 22484 solver.cpp:464] Iteration 882, lr = 1e-05
I1118 02:52:36.910722 22484 solver.cpp:209] Iteration 883, loss = 0.342075
I1118 02:52:36.910753 22484 solver.cpp:464] Iteration 883, lr = 1e-05
I1118 02:52:39.134380 22484 solver.cpp:209] Iteration 884, loss = 0.572658
I1118 02:52:39.134421 22484 solver.cpp:464] Iteration 884, lr = 1e-05
I1118 02:52:41.355984 22484 solver.cpp:209] Iteration 885, loss = 0.316299
I1118 02:52:41.356036 22484 solver.cpp:464] Iteration 885, lr = 1e-05
I1118 02:52:43.579314 22484 solver.cpp:209] Iteration 886, loss = 0.729911
I1118 02:52:43.579355 22484 solver.cpp:464] Iteration 886, lr = 1e-05
I1118 02:52:45.809542 22484 solver.cpp:209] Iteration 887, loss = 0.167427
I1118 02:52:45.809571 22484 solver.cpp:464] Iteration 887, lr = 1e-05
I1118 02:52:48.043422 22484 solver.cpp:209] Iteration 888, loss = 0.299987
I1118 02:52:48.043462 22484 solver.cpp:464] Iteration 888, lr = 1e-05
I1118 02:52:50.272160 22484 solver.cpp:209] Iteration 889, loss = 0.441961
I1118 02:52:50.272202 22484 solver.cpp:464] Iteration 889, lr = 1e-05
I1118 02:52:52.497468 22484 solver.cpp:209] Iteration 890, loss = 0.42489
I1118 02:52:52.497498 22484 solver.cpp:464] Iteration 890, lr = 1e-05
I1118 02:52:54.721163 22484 solver.cpp:209] Iteration 891, loss = 0.234416
I1118 02:52:54.721192 22484 solver.cpp:464] Iteration 891, lr = 1e-05
I1118 02:52:56.944067 22484 solver.cpp:209] Iteration 892, loss = 0.35414
I1118 02:52:56.944108 22484 solver.cpp:464] Iteration 892, lr = 1e-05
I1118 02:52:59.177724 22484 solver.cpp:209] Iteration 893, loss = 0.404555
I1118 02:52:59.177767 22484 solver.cpp:464] Iteration 893, lr = 1e-05
I1118 02:53:01.410290 22484 solver.cpp:209] Iteration 894, loss = 0.523873
I1118 02:53:01.410320 22484 solver.cpp:464] Iteration 894, lr = 1e-05
I1118 02:53:03.636955 22484 solver.cpp:209] Iteration 895, loss = 0.320769
I1118 02:53:03.636996 22484 solver.cpp:464] Iteration 895, lr = 1e-05
I1118 02:53:05.863638 22484 solver.cpp:209] Iteration 896, loss = 0.376001
I1118 02:53:05.863667 22484 solver.cpp:464] Iteration 896, lr = 1e-05
I1118 02:53:08.083693 22484 solver.cpp:209] Iteration 897, loss = 0.258449
I1118 02:53:08.083732 22484 solver.cpp:464] Iteration 897, lr = 1e-05
I1118 02:53:10.315875 22484 solver.cpp:209] Iteration 898, loss = 0.375469
I1118 02:53:10.315917 22484 solver.cpp:464] Iteration 898, lr = 1e-05
I1118 02:53:12.539849 22484 solver.cpp:209] Iteration 899, loss = 0.222689
I1118 02:53:12.539904 22484 solver.cpp:464] Iteration 899, lr = 1e-05
I1118 02:53:12.540498 22484 solver.cpp:264] Iteration 900, Testing net (#0)
I1118 02:53:26.420608 22484 solver.cpp:305] Test loss: 0.26287
I1118 02:53:26.420646 22484 solver.cpp:318] mean_score = test_score[0] { = 509} / test_score[1] { = 517 }
I1118 02:53:26.420655 22484 solver.cpp:319]            = 0.984526
I1118 02:53:26.420658 22484 solver.cpp:328]     Test net output #0: accuracy = 0.984526
I1118 02:53:26.420663 22484 solver.cpp:318] mean_score = test_score[2] { = 7} / test_score[3] { = 59 }
I1118 02:53:26.420668 22484 solver.cpp:319]            = 0.118644
I1118 02:53:26.420671 22484 solver.cpp:328]     Test net output #1: accuracy = 0.118644
I1118 02:53:26.420676 22484 solver.cpp:332]     Test net output #2: accuracy = 0.895833
I1118 02:53:26.420680 22484 solver.cpp:334]     Test net output #3: accuracy = 0.551585
I1118 02:53:27.066437 22484 solver.cpp:209] Iteration 900, loss = 0.222409
I1118 02:53:27.066465 22484 solver.cpp:464] Iteration 900, lr = 1e-05
I1118 02:53:29.289383 22484 solver.cpp:209] Iteration 901, loss = 0.412409
I1118 02:53:29.289412 22484 solver.cpp:464] Iteration 901, lr = 1e-05
I1118 02:53:31.520325 22484 solver.cpp:209] Iteration 902, loss = 0.208008
I1118 02:53:31.520365 22484 solver.cpp:464] Iteration 902, lr = 1e-05
I1118 02:53:33.749562 22484 solver.cpp:209] Iteration 903, loss = 0.217074
I1118 02:53:33.749591 22484 solver.cpp:464] Iteration 903, lr = 1e-05
I1118 02:53:35.978786 22484 solver.cpp:209] Iteration 904, loss = 0.148428
I1118 02:53:35.978817 22484 solver.cpp:464] Iteration 904, lr = 1e-05
I1118 02:53:38.204305 22484 solver.cpp:209] Iteration 905, loss = 0.305041
I1118 02:53:38.204334 22484 solver.cpp:464] Iteration 905, lr = 1e-05
I1118 02:53:40.423648 22484 solver.cpp:209] Iteration 906, loss = 0.429065
I1118 02:53:40.423691 22484 solver.cpp:464] Iteration 906, lr = 1e-05
I1118 02:53:42.643862 22484 solver.cpp:209] Iteration 907, loss = 0.407579
I1118 02:53:42.643915 22484 solver.cpp:464] Iteration 907, lr = 1e-05
I1118 02:53:44.877807 22484 solver.cpp:209] Iteration 908, loss = 0.443375
I1118 02:53:44.877851 22484 solver.cpp:464] Iteration 908, lr = 1e-05
I1118 02:53:47.104998 22484 solver.cpp:209] Iteration 909, loss = 0.35342
I1118 02:53:47.105028 22484 solver.cpp:464] Iteration 909, lr = 1e-05
I1118 02:53:49.338927 22484 solver.cpp:209] Iteration 910, loss = 0.105521
I1118 02:53:49.338956 22484 solver.cpp:464] Iteration 910, lr = 1e-05
I1118 02:53:51.567031 22484 solver.cpp:209] Iteration 911, loss = 0.108591
I1118 02:53:51.567073 22484 solver.cpp:464] Iteration 911, lr = 1e-05
I1118 02:53:53.788527 22484 solver.cpp:209] Iteration 912, loss = 0.2324
I1118 02:53:53.788568 22484 solver.cpp:464] Iteration 912, lr = 1e-05
I1118 02:53:56.014338 22484 solver.cpp:209] Iteration 913, loss = 0.382925
I1118 02:53:56.014367 22484 solver.cpp:464] Iteration 913, lr = 1e-05
I1118 02:53:58.247292 22484 solver.cpp:209] Iteration 914, loss = 0.212458
I1118 02:53:58.247318 22484 solver.cpp:464] Iteration 914, lr = 1e-05
I1118 02:54:00.475237 22484 solver.cpp:209] Iteration 915, loss = 0.49722
I1118 02:54:00.475266 22484 solver.cpp:464] Iteration 915, lr = 1e-05
I1118 02:54:02.706013 22484 solver.cpp:209] Iteration 916, loss = 0.0610148
I1118 02:54:02.706043 22484 solver.cpp:464] Iteration 916, lr = 1e-05
I1118 02:54:04.934909 22484 solver.cpp:209] Iteration 917, loss = 0.275807
I1118 02:54:04.934939 22484 solver.cpp:464] Iteration 917, lr = 1e-05
I1118 02:54:07.162535 22484 solver.cpp:209] Iteration 918, loss = 0.246499
I1118 02:54:07.162564 22484 solver.cpp:464] Iteration 918, lr = 1e-05
I1118 02:54:09.381450 22484 solver.cpp:209] Iteration 919, loss = 0.295961
I1118 02:54:09.381480 22484 solver.cpp:464] Iteration 919, lr = 1e-05
I1118 02:54:11.608521 22484 solver.cpp:209] Iteration 920, loss = 0.485138
I1118 02:54:11.608562 22484 solver.cpp:464] Iteration 920, lr = 1e-05
I1118 02:54:13.832272 22484 solver.cpp:209] Iteration 921, loss = 0.218993
I1118 02:54:13.832330 22484 solver.cpp:464] Iteration 921, lr = 1e-05
I1118 02:54:16.063983 22484 solver.cpp:209] Iteration 922, loss = 0.337298
I1118 02:54:16.064013 22484 solver.cpp:464] Iteration 922, lr = 1e-05
I1118 02:54:18.300097 22484 solver.cpp:209] Iteration 923, loss = 0.172053
I1118 02:54:18.300137 22484 solver.cpp:464] Iteration 923, lr = 1e-05
I1118 02:54:20.527750 22484 solver.cpp:209] Iteration 924, loss = 0.160884
I1118 02:54:20.527781 22484 solver.cpp:464] Iteration 924, lr = 1e-05
I1118 02:54:22.752163 22484 solver.cpp:209] Iteration 925, loss = 0.337253
I1118 02:54:22.752193 22484 solver.cpp:464] Iteration 925, lr = 1e-05
I1118 02:54:24.975733 22484 solver.cpp:209] Iteration 926, loss = 0.2555
I1118 02:54:24.975761 22484 solver.cpp:464] Iteration 926, lr = 1e-05
I1118 02:54:27.197121 22484 solver.cpp:209] Iteration 927, loss = 0.2147
I1118 02:54:27.197163 22484 solver.cpp:464] Iteration 927, lr = 1e-05
I1118 02:54:29.428076 22484 solver.cpp:209] Iteration 928, loss = 0.256096
I1118 02:54:29.428105 22484 solver.cpp:464] Iteration 928, lr = 1e-05
I1118 02:54:31.660459 22484 solver.cpp:209] Iteration 929, loss = 0.28452
I1118 02:54:31.660487 22484 solver.cpp:464] Iteration 929, lr = 1e-05
I1118 02:54:33.893033 22484 solver.cpp:209] Iteration 930, loss = 0.253092
I1118 02:54:33.893061 22484 solver.cpp:464] Iteration 930, lr = 1e-05
I1118 02:54:36.122834 22484 solver.cpp:209] Iteration 931, loss = 0.138677
I1118 02:54:36.122865 22484 solver.cpp:464] Iteration 931, lr = 1e-05
I1118 02:54:38.344341 22484 solver.cpp:209] Iteration 932, loss = 0.287867
I1118 02:54:38.344380 22484 solver.cpp:464] Iteration 932, lr = 1e-05
I1118 02:54:40.569978 22484 solver.cpp:209] Iteration 933, loss = 0.377109
I1118 02:54:40.570019 22484 solver.cpp:464] Iteration 933, lr = 1e-05
I1118 02:54:42.789888 22484 solver.cpp:209] Iteration 934, loss = 0.15028
I1118 02:54:42.789929 22484 solver.cpp:464] Iteration 934, lr = 1e-05
I1118 02:54:45.016948 22484 solver.cpp:209] Iteration 935, loss = 0.283331
I1118 02:54:45.017038 22484 solver.cpp:464] Iteration 935, lr = 1e-05
I1118 02:54:47.250946 22484 solver.cpp:209] Iteration 936, loss = 0.171989
I1118 02:54:47.250975 22484 solver.cpp:464] Iteration 936, lr = 1e-05
I1118 02:54:49.480509 22484 solver.cpp:209] Iteration 937, loss = 0.346922
I1118 02:54:49.480550 22484 solver.cpp:464] Iteration 937, lr = 1e-05
I1118 02:54:51.717164 22484 solver.cpp:209] Iteration 938, loss = 0.329511
I1118 02:54:51.717193 22484 solver.cpp:464] Iteration 938, lr = 1e-05
I1118 02:54:53.939673 22484 solver.cpp:209] Iteration 939, loss = 0.286596
I1118 02:54:53.939715 22484 solver.cpp:464] Iteration 939, lr = 1e-05
I1118 02:54:56.165544 22484 solver.cpp:209] Iteration 940, loss = 0.140667
I1118 02:54:56.165573 22484 solver.cpp:464] Iteration 940, lr = 1e-05
I1118 02:54:58.383131 22484 solver.cpp:209] Iteration 941, loss = 0.35365
I1118 02:54:58.383158 22484 solver.cpp:464] Iteration 941, lr = 1e-05
I1118 02:55:00.573694 22484 solver.cpp:209] Iteration 942, loss = 0.110922
I1118 02:55:00.573734 22484 solver.cpp:464] Iteration 942, lr = 1e-05
I1118 02:55:02.774591 22484 solver.cpp:209] Iteration 943, loss = 0.105112
I1118 02:55:02.774634 22484 solver.cpp:464] Iteration 943, lr = 1e-05
I1118 02:55:04.968557 22484 solver.cpp:209] Iteration 944, loss = 0.175256
I1118 02:55:04.968600 22484 solver.cpp:464] Iteration 944, lr = 1e-05
I1118 02:55:07.159945 22484 solver.cpp:209] Iteration 945, loss = 0.190173
I1118 02:55:07.159973 22484 solver.cpp:464] Iteration 945, lr = 1e-05
I1118 02:55:09.351824 22484 solver.cpp:209] Iteration 946, loss = 0.335321
I1118 02:55:09.351866 22484 solver.cpp:464] Iteration 946, lr = 1e-05
I1118 02:55:11.540349 22484 solver.cpp:209] Iteration 947, loss = 0.325959
I1118 02:55:11.540379 22484 solver.cpp:464] Iteration 947, lr = 1e-05
I1118 02:55:13.747115 22484 solver.cpp:209] Iteration 948, loss = 0.157913
I1118 02:55:13.747156 22484 solver.cpp:464] Iteration 948, lr = 1e-05
I1118 02:55:15.942160 22484 solver.cpp:209] Iteration 949, loss = 0.193492
I1118 02:55:15.942214 22484 solver.cpp:464] Iteration 949, lr = 1e-05
I1118 02:55:15.942844 22484 solver.cpp:264] Iteration 950, Testing net (#0)
I1118 02:55:29.824578 22484 solver.cpp:305] Test loss: 0.268004
I1118 02:55:29.824606 22484 solver.cpp:318] mean_score = test_score[0] { = 512} / test_score[1] { = 517 }
I1118 02:55:29.824626 22484 solver.cpp:319]            = 0.990329
I1118 02:55:29.824630 22484 solver.cpp:328]     Test net output #0: accuracy = 0.990329
I1118 02:55:29.824635 22484 solver.cpp:318] mean_score = test_score[2] { = 3} / test_score[3] { = 59 }
I1118 02:55:29.824640 22484 solver.cpp:319]            = 0.0508475
I1118 02:55:29.824643 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0508475
I1118 02:55:29.824648 22484 solver.cpp:332]     Test net output #2: accuracy = 0.894097
I1118 02:55:29.824652 22484 solver.cpp:334]     Test net output #3: accuracy = 0.520588
I1118 02:55:30.487450 22484 solver.cpp:209] Iteration 950, loss = 0.166403
I1118 02:55:30.487480 22484 solver.cpp:464] Iteration 950, lr = 1e-05
I1118 02:55:32.763083 22484 solver.cpp:209] Iteration 951, loss = 0.195586
I1118 02:55:32.763125 22484 solver.cpp:464] Iteration 951, lr = 1e-05
I1118 02:55:35.042498 22484 solver.cpp:209] Iteration 952, loss = 0.313033
I1118 02:55:35.042528 22484 solver.cpp:464] Iteration 952, lr = 1e-05
I1118 02:55:37.318259 22484 solver.cpp:209] Iteration 953, loss = 0.265081
I1118 02:55:37.318300 22484 solver.cpp:464] Iteration 953, lr = 1e-05
I1118 02:55:39.582346 22484 solver.cpp:209] Iteration 954, loss = 0.0840077
I1118 02:55:39.582387 22484 solver.cpp:464] Iteration 954, lr = 1e-05
I1118 02:55:41.830765 22484 solver.cpp:209] Iteration 955, loss = 0.703785
I1118 02:55:41.830796 22484 solver.cpp:464] Iteration 955, lr = 1e-05
I1118 02:55:44.066970 22484 solver.cpp:209] Iteration 956, loss = 0.19847
I1118 02:55:44.066999 22484 solver.cpp:464] Iteration 956, lr = 1e-05
I1118 02:55:46.292754 22484 solver.cpp:209] Iteration 957, loss = 0.256327
I1118 02:55:46.292820 22484 solver.cpp:464] Iteration 957, lr = 1e-05
I1118 02:55:48.524765 22484 solver.cpp:209] Iteration 958, loss = 0.341961
I1118 02:55:48.524796 22484 solver.cpp:464] Iteration 958, lr = 1e-05
I1118 02:55:50.747936 22484 solver.cpp:209] Iteration 959, loss = 0.327418
I1118 02:55:50.747967 22484 solver.cpp:464] Iteration 959, lr = 1e-05
I1118 02:55:52.948268 22484 solver.cpp:209] Iteration 960, loss = 0.170523
I1118 02:55:52.948298 22484 solver.cpp:464] Iteration 960, lr = 1e-05
I1118 02:55:55.139405 22484 solver.cpp:209] Iteration 961, loss = 0.279521
I1118 02:55:55.139436 22484 solver.cpp:464] Iteration 961, lr = 1e-05
I1118 02:55:57.329988 22484 solver.cpp:209] Iteration 962, loss = 0.309455
I1118 02:55:57.330016 22484 solver.cpp:464] Iteration 962, lr = 1e-05
I1118 02:55:59.533049 22484 solver.cpp:209] Iteration 963, loss = 0.461778
I1118 02:55:59.533078 22484 solver.cpp:464] Iteration 963, lr = 1e-05
I1118 02:56:01.720878 22484 solver.cpp:209] Iteration 964, loss = 0.564952
I1118 02:56:01.720919 22484 solver.cpp:464] Iteration 964, lr = 1e-05
I1118 02:56:03.923779 22484 solver.cpp:209] Iteration 965, loss = 0.194399
I1118 02:56:03.923821 22484 solver.cpp:464] Iteration 965, lr = 1e-05
I1118 02:56:06.118229 22484 solver.cpp:209] Iteration 966, loss = 0.298565
I1118 02:56:06.118259 22484 solver.cpp:464] Iteration 966, lr = 1e-05
I1118 02:56:08.306978 22484 solver.cpp:209] Iteration 967, loss = 0.181356
I1118 02:56:08.307018 22484 solver.cpp:464] Iteration 967, lr = 1e-05
I1118 02:56:10.499414 22484 solver.cpp:209] Iteration 968, loss = 0.306061
I1118 02:56:10.499456 22484 solver.cpp:464] Iteration 968, lr = 1e-05
I1118 02:56:12.689522 22484 solver.cpp:209] Iteration 969, loss = 0.144529
I1118 02:56:12.689563 22484 solver.cpp:464] Iteration 969, lr = 1e-05
I1118 02:56:14.894511 22484 solver.cpp:209] Iteration 970, loss = 0.182457
I1118 02:56:14.894541 22484 solver.cpp:464] Iteration 970, lr = 1e-05
I1118 02:56:17.095387 22484 solver.cpp:209] Iteration 971, loss = 0.182266
I1118 02:56:17.095481 22484 solver.cpp:464] Iteration 971, lr = 1e-05
I1118 02:56:19.281325 22484 solver.cpp:209] Iteration 972, loss = 0.101261
I1118 02:56:19.281368 22484 solver.cpp:464] Iteration 972, lr = 1e-05
I1118 02:56:21.488152 22484 solver.cpp:209] Iteration 973, loss = 0.24487
I1118 02:56:21.488183 22484 solver.cpp:464] Iteration 973, lr = 1e-05
I1118 02:56:23.725107 22484 solver.cpp:209] Iteration 974, loss = 0.587647
I1118 02:56:23.725148 22484 solver.cpp:464] Iteration 974, lr = 1e-05
I1118 02:56:25.969944 22484 solver.cpp:209] Iteration 975, loss = 0.292897
I1118 02:56:25.969985 22484 solver.cpp:464] Iteration 975, lr = 1e-05
I1118 02:56:28.227586 22484 solver.cpp:209] Iteration 976, loss = 0.249807
I1118 02:56:28.227613 22484 solver.cpp:464] Iteration 976, lr = 1e-05
I1118 02:56:30.474293 22484 solver.cpp:209] Iteration 977, loss = 0.193632
I1118 02:56:30.474323 22484 solver.cpp:464] Iteration 977, lr = 1e-05
I1118 02:56:32.729521 22484 solver.cpp:209] Iteration 978, loss = 0.358562
I1118 02:56:32.729552 22484 solver.cpp:464] Iteration 978, lr = 1e-05
I1118 02:56:34.969684 22484 solver.cpp:209] Iteration 979, loss = 0.105235
I1118 02:56:34.969713 22484 solver.cpp:464] Iteration 979, lr = 1e-05
I1118 02:56:37.212787 22484 solver.cpp:209] Iteration 980, loss = 0.241374
I1118 02:56:37.212829 22484 solver.cpp:464] Iteration 980, lr = 1e-05
I1118 02:56:39.457538 22484 solver.cpp:209] Iteration 981, loss = 0.285173
I1118 02:56:39.457568 22484 solver.cpp:464] Iteration 981, lr = 1e-05
I1118 02:56:41.685549 22484 solver.cpp:209] Iteration 982, loss = 0.23849
I1118 02:56:41.685590 22484 solver.cpp:464] Iteration 982, lr = 1e-05
I1118 02:56:43.918607 22484 solver.cpp:209] Iteration 983, loss = 0.330376
I1118 02:56:43.918637 22484 solver.cpp:464] Iteration 983, lr = 1e-05
I1118 02:56:46.144557 22484 solver.cpp:209] Iteration 984, loss = 0.17368
I1118 02:56:46.144587 22484 solver.cpp:464] Iteration 984, lr = 1e-05
I1118 02:56:48.377449 22484 solver.cpp:209] Iteration 985, loss = 0.268619
I1118 02:56:48.377501 22484 solver.cpp:464] Iteration 985, lr = 1e-05
I1118 02:56:50.575804 22484 solver.cpp:209] Iteration 986, loss = 0.226819
I1118 02:56:50.575846 22484 solver.cpp:464] Iteration 986, lr = 1e-05
I1118 02:56:52.764855 22484 solver.cpp:209] Iteration 987, loss = 0.534513
I1118 02:56:52.764884 22484 solver.cpp:464] Iteration 987, lr = 1e-05
I1118 02:56:54.960206 22484 solver.cpp:209] Iteration 988, loss = 0.405567
I1118 02:56:54.960235 22484 solver.cpp:464] Iteration 988, lr = 1e-05
I1118 02:56:57.171306 22484 solver.cpp:209] Iteration 989, loss = 0.691529
I1118 02:56:57.171347 22484 solver.cpp:464] Iteration 989, lr = 1e-05
I1118 02:56:59.376948 22484 solver.cpp:209] Iteration 990, loss = 0.0741791
I1118 02:56:59.376991 22484 solver.cpp:464] Iteration 990, lr = 1e-05
I1118 02:57:01.576413 22484 solver.cpp:209] Iteration 991, loss = 0.526867
I1118 02:57:01.576443 22484 solver.cpp:464] Iteration 991, lr = 1e-05
I1118 02:57:03.768681 22484 solver.cpp:209] Iteration 992, loss = 0.219119
I1118 02:57:03.768710 22484 solver.cpp:464] Iteration 992, lr = 1e-05
I1118 02:57:05.957895 22484 solver.cpp:209] Iteration 993, loss = 0.528582
I1118 02:57:05.957937 22484 solver.cpp:464] Iteration 993, lr = 1e-05
I1118 02:57:08.144992 22484 solver.cpp:209] Iteration 994, loss = 0.400782
I1118 02:57:08.145032 22484 solver.cpp:464] Iteration 994, lr = 1e-05
I1118 02:57:10.345876 22484 solver.cpp:209] Iteration 995, loss = 0.247804
I1118 02:57:10.345906 22484 solver.cpp:464] Iteration 995, lr = 1e-05
I1118 02:57:12.546221 22484 solver.cpp:209] Iteration 996, loss = 0.519755
I1118 02:57:12.546249 22484 solver.cpp:464] Iteration 996, lr = 1e-05
I1118 02:57:14.742298 22484 solver.cpp:209] Iteration 997, loss = 0.343156
I1118 02:57:14.742329 22484 solver.cpp:464] Iteration 997, lr = 1e-05
I1118 02:57:16.966898 22484 solver.cpp:209] Iteration 998, loss = 0.206929
I1118 02:57:16.966928 22484 solver.cpp:464] Iteration 998, lr = 1e-05
I1118 02:57:19.186565 22484 solver.cpp:209] Iteration 999, loss = 0.44721
I1118 02:57:19.186646 22484 solver.cpp:464] Iteration 999, lr = 1e-05
I1118 02:57:19.187243 22484 solver.cpp:264] Iteration 1000, Testing net (#0)
I1118 02:57:33.100859 22484 solver.cpp:305] Test loss: 0.264799
I1118 02:57:33.100899 22484 solver.cpp:318] mean_score = test_score[0] { = 509} / test_score[1] { = 517 }
I1118 02:57:33.100906 22484 solver.cpp:319]            = 0.984526
I1118 02:57:33.100911 22484 solver.cpp:328]     Test net output #0: accuracy = 0.984526
I1118 02:57:33.100915 22484 solver.cpp:318] mean_score = test_score[2] { = 5} / test_score[3] { = 59 }
I1118 02:57:33.100920 22484 solver.cpp:319]            = 0.0847458
I1118 02:57:33.100924 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0847458
I1118 02:57:33.100929 22484 solver.cpp:332]     Test net output #2: accuracy = 0.892361
I1118 02:57:33.100932 22484 solver.cpp:334]     Test net output #3: accuracy = 0.534636
I1118 02:57:33.761593 22484 solver.cpp:209] Iteration 1000, loss = 0.137774
I1118 02:57:33.761622 22484 solver.cpp:464] Iteration 1000, lr = 1e-05
I1118 02:57:36.035022 22484 solver.cpp:209] Iteration 1001, loss = 0.514249
I1118 02:57:36.035051 22484 solver.cpp:464] Iteration 1001, lr = 1e-05
I1118 02:57:38.303328 22484 solver.cpp:209] Iteration 1002, loss = 0.124869
I1118 02:57:38.303367 22484 solver.cpp:464] Iteration 1002, lr = 1e-05
I1118 02:57:40.577419 22484 solver.cpp:209] Iteration 1003, loss = 0.316733
I1118 02:57:40.577448 22484 solver.cpp:464] Iteration 1003, lr = 1e-05
I1118 02:57:42.861853 22484 solver.cpp:209] Iteration 1004, loss = 0.458527
I1118 02:57:42.861893 22484 solver.cpp:464] Iteration 1004, lr = 1e-05
I1118 02:57:45.122272 22484 solver.cpp:209] Iteration 1005, loss = 0.198096
I1118 02:57:45.122301 22484 solver.cpp:464] Iteration 1005, lr = 1e-05
I1118 02:57:47.356163 22484 solver.cpp:209] Iteration 1006, loss = 0.309314
I1118 02:57:47.356191 22484 solver.cpp:464] Iteration 1006, lr = 1e-05
I1118 02:57:49.580317 22484 solver.cpp:209] Iteration 1007, loss = 0.326993
I1118 02:57:49.580410 22484 solver.cpp:464] Iteration 1007, lr = 1e-05
I1118 02:57:51.773108 22484 solver.cpp:209] Iteration 1008, loss = 0.306347
I1118 02:57:51.773138 22484 solver.cpp:464] Iteration 1008, lr = 1e-05
I1118 02:57:53.962649 22484 solver.cpp:209] Iteration 1009, loss = 0.407453
I1118 02:57:53.962678 22484 solver.cpp:464] Iteration 1009, lr = 1e-05
I1118 02:57:56.157176 22484 solver.cpp:209] Iteration 1010, loss = 0.603268
I1118 02:57:56.157205 22484 solver.cpp:464] Iteration 1010, lr = 1e-05
I1118 02:57:58.367151 22484 solver.cpp:209] Iteration 1011, loss = 0.349934
I1118 02:57:58.367178 22484 solver.cpp:464] Iteration 1011, lr = 1e-05
I1118 02:58:00.559224 22484 solver.cpp:209] Iteration 1012, loss = 0.292757
I1118 02:58:00.559264 22484 solver.cpp:464] Iteration 1012, lr = 1e-05
I1118 02:58:02.749963 22484 solver.cpp:209] Iteration 1013, loss = 0.108254
I1118 02:58:02.749991 22484 solver.cpp:464] Iteration 1013, lr = 1e-05
I1118 02:58:04.937139 22484 solver.cpp:209] Iteration 1014, loss = 0.124207
I1118 02:58:04.937167 22484 solver.cpp:464] Iteration 1014, lr = 1e-05
I1118 02:58:07.132864 22484 solver.cpp:209] Iteration 1015, loss = 0.266885
I1118 02:58:07.132905 22484 solver.cpp:464] Iteration 1015, lr = 1e-05
I1118 02:58:09.327451 22484 solver.cpp:209] Iteration 1016, loss = 0.341474
I1118 02:58:09.327492 22484 solver.cpp:464] Iteration 1016, lr = 1e-05
I1118 02:58:11.522168 22484 solver.cpp:209] Iteration 1017, loss = 0.175459
I1118 02:58:11.522210 22484 solver.cpp:464] Iteration 1017, lr = 1e-05
I1118 02:58:13.732255 22484 solver.cpp:209] Iteration 1018, loss = 0.520233
I1118 02:58:13.732296 22484 solver.cpp:464] Iteration 1018, lr = 1e-05
I1118 02:58:15.958526 22484 solver.cpp:209] Iteration 1019, loss = 0.176861
I1118 02:58:15.958567 22484 solver.cpp:464] Iteration 1019, lr = 1e-05
I1118 02:58:18.188832 22484 solver.cpp:209] Iteration 1020, loss = 0.328601
I1118 02:58:18.188871 22484 solver.cpp:464] Iteration 1020, lr = 1e-05
I1118 02:58:20.413615 22484 solver.cpp:209] Iteration 1021, loss = 0.208347
I1118 02:58:20.413707 22484 solver.cpp:464] Iteration 1021, lr = 1e-05
I1118 02:58:22.637562 22484 solver.cpp:209] Iteration 1022, loss = 0.254187
I1118 02:58:22.637601 22484 solver.cpp:464] Iteration 1022, lr = 1e-05
I1118 02:58:24.864156 22484 solver.cpp:209] Iteration 1023, loss = 0.396797
I1118 02:58:24.864186 22484 solver.cpp:464] Iteration 1023, lr = 1e-05
I1118 02:58:27.094588 22484 solver.cpp:209] Iteration 1024, loss = 0.162716
I1118 02:58:27.094630 22484 solver.cpp:464] Iteration 1024, lr = 1e-05
I1118 02:58:29.329675 22484 solver.cpp:209] Iteration 1025, loss = 0.26678
I1118 02:58:29.329717 22484 solver.cpp:464] Iteration 1025, lr = 1e-05
I1118 02:58:31.559898 22484 solver.cpp:209] Iteration 1026, loss = 0.249799
I1118 02:58:31.559927 22484 solver.cpp:464] Iteration 1026, lr = 1e-05
I1118 02:58:33.781719 22484 solver.cpp:209] Iteration 1027, loss = 0.129083
I1118 02:58:33.781759 22484 solver.cpp:464] Iteration 1027, lr = 1e-05
I1118 02:58:36.005246 22484 solver.cpp:209] Iteration 1028, loss = 0.257738
I1118 02:58:36.005275 22484 solver.cpp:464] Iteration 1028, lr = 1e-05
I1118 02:58:38.229640 22484 solver.cpp:209] Iteration 1029, loss = 0.24657
I1118 02:58:38.229668 22484 solver.cpp:464] Iteration 1029, lr = 1e-05
I1118 02:58:40.468183 22484 solver.cpp:209] Iteration 1030, loss = 0.0821952
I1118 02:58:40.468224 22484 solver.cpp:464] Iteration 1030, lr = 1e-05
I1118 02:58:42.697744 22484 solver.cpp:209] Iteration 1031, loss = 0.304641
I1118 02:58:42.697773 22484 solver.cpp:464] Iteration 1031, lr = 1e-05
I1118 02:58:44.951155 22484 solver.cpp:209] Iteration 1032, loss = 0.185457
I1118 02:58:44.951185 22484 solver.cpp:464] Iteration 1032, lr = 1e-05
I1118 02:58:47.177152 22484 solver.cpp:209] Iteration 1033, loss = 0.134554
I1118 02:58:47.177192 22484 solver.cpp:464] Iteration 1033, lr = 1e-05
I1118 02:58:49.398768 22484 solver.cpp:209] Iteration 1034, loss = 0.182355
I1118 02:58:49.398798 22484 solver.cpp:464] Iteration 1034, lr = 1e-05
I1118 02:58:51.627428 22484 solver.cpp:209] Iteration 1035, loss = 0.27198
I1118 02:58:51.627481 22484 solver.cpp:464] Iteration 1035, lr = 1e-05
I1118 02:58:53.853854 22484 solver.cpp:209] Iteration 1036, loss = 0.311538
I1118 02:58:53.853883 22484 solver.cpp:464] Iteration 1036, lr = 1e-05
I1118 02:58:56.087725 22484 solver.cpp:209] Iteration 1037, loss = 0.178848
I1118 02:58:56.087766 22484 solver.cpp:464] Iteration 1037, lr = 1e-05
I1118 02:58:58.333070 22484 solver.cpp:209] Iteration 1038, loss = 0.102966
I1118 02:58:58.333112 22484 solver.cpp:464] Iteration 1038, lr = 1e-05
I1118 02:59:00.560374 22484 solver.cpp:209] Iteration 1039, loss = 0.185934
I1118 02:59:00.560400 22484 solver.cpp:464] Iteration 1039, lr = 1e-05
I1118 02:59:02.788238 22484 solver.cpp:209] Iteration 1040, loss = 0.295696
I1118 02:59:02.788278 22484 solver.cpp:464] Iteration 1040, lr = 1e-05
I1118 02:59:05.010006 22484 solver.cpp:209] Iteration 1041, loss = 0.423473
I1118 02:59:05.010047 22484 solver.cpp:464] Iteration 1041, lr = 1e-05
I1118 02:59:07.234092 22484 solver.cpp:209] Iteration 1042, loss = 0.281291
I1118 02:59:07.234122 22484 solver.cpp:464] Iteration 1042, lr = 1e-05
I1118 02:59:09.459316 22484 solver.cpp:209] Iteration 1043, loss = 0.296646
I1118 02:59:09.459343 22484 solver.cpp:464] Iteration 1043, lr = 1e-05
I1118 02:59:11.705894 22484 solver.cpp:209] Iteration 1044, loss = 0.216115
I1118 02:59:11.705924 22484 solver.cpp:464] Iteration 1044, lr = 1e-05
I1118 02:59:13.940345 22484 solver.cpp:209] Iteration 1045, loss = 0.0530674
I1118 02:59:13.940374 22484 solver.cpp:464] Iteration 1045, lr = 1e-05
I1118 02:59:16.172824 22484 solver.cpp:209] Iteration 1046, loss = 0.161346
I1118 02:59:16.172865 22484 solver.cpp:464] Iteration 1046, lr = 1e-05
I1118 02:59:18.395746 22484 solver.cpp:209] Iteration 1047, loss = 0.259955
I1118 02:59:18.395787 22484 solver.cpp:464] Iteration 1047, lr = 1e-05
I1118 02:59:20.618222 22484 solver.cpp:209] Iteration 1048, loss = 0.145693
I1118 02:59:20.618252 22484 solver.cpp:464] Iteration 1048, lr = 1e-05
I1118 02:59:22.843616 22484 solver.cpp:209] Iteration 1049, loss = 0.243961
I1118 02:59:22.843710 22484 solver.cpp:464] Iteration 1049, lr = 1e-05
I1118 02:59:22.844290 22484 solver.cpp:264] Iteration 1050, Testing net (#0)
I1118 02:59:36.734158 22484 solver.cpp:305] Test loss: 0.261764
I1118 02:59:36.734186 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 02:59:36.734194 22484 solver.cpp:319]            = 0.98646
I1118 02:59:36.734210 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 02:59:36.734215 22484 solver.cpp:318] mean_score = test_score[2] { = 6} / test_score[3] { = 59 }
I1118 02:59:36.734220 22484 solver.cpp:319]            = 0.101695
I1118 02:59:36.734223 22484 solver.cpp:328]     Test net output #1: accuracy = 0.101695
I1118 02:59:36.734227 22484 solver.cpp:332]     Test net output #2: accuracy = 0.895833
I1118 02:59:36.734231 22484 solver.cpp:334]     Test net output #3: accuracy = 0.544078
I1118 02:59:37.381127 22484 solver.cpp:209] Iteration 1050, loss = 0.238682
I1118 02:59:37.381168 22484 solver.cpp:464] Iteration 1050, lr = 1e-05
I1118 02:59:39.602509 22484 solver.cpp:209] Iteration 1051, loss = 0.214888
I1118 02:59:39.602537 22484 solver.cpp:464] Iteration 1051, lr = 1e-05
I1118 02:59:41.826992 22484 solver.cpp:209] Iteration 1052, loss = 0.134956
I1118 02:59:41.827021 22484 solver.cpp:464] Iteration 1052, lr = 1e-05
I1118 02:59:44.058696 22484 solver.cpp:209] Iteration 1053, loss = 0.0957479
I1118 02:59:44.058725 22484 solver.cpp:464] Iteration 1053, lr = 1e-05
I1118 02:59:46.293491 22484 solver.cpp:209] Iteration 1054, loss = 0.297459
I1118 02:59:46.293521 22484 solver.cpp:464] Iteration 1054, lr = 1e-05
I1118 02:59:48.517901 22484 solver.cpp:209] Iteration 1055, loss = 0.237092
I1118 02:59:48.517943 22484 solver.cpp:464] Iteration 1055, lr = 1e-05
I1118 02:59:50.741288 22484 solver.cpp:209] Iteration 1056, loss = 0.13406
I1118 02:59:50.741317 22484 solver.cpp:464] Iteration 1056, lr = 1e-05
I1118 02:59:52.957945 22484 solver.cpp:209] Iteration 1057, loss = 0.130073
I1118 02:59:52.958034 22484 solver.cpp:464] Iteration 1057, lr = 1e-05
I1118 02:59:55.184190 22484 solver.cpp:209] Iteration 1058, loss = 0.528248
I1118 02:59:55.184231 22484 solver.cpp:464] Iteration 1058, lr = 1e-05
I1118 02:59:57.420871 22484 solver.cpp:209] Iteration 1059, loss = 0.230591
I1118 02:59:57.420898 22484 solver.cpp:464] Iteration 1059, lr = 1e-05
I1118 02:59:59.647012 22484 solver.cpp:209] Iteration 1060, loss = 0.196374
I1118 02:59:59.647053 22484 solver.cpp:464] Iteration 1060, lr = 1e-05
I1118 03:00:01.881827 22484 solver.cpp:209] Iteration 1061, loss = 0.336075
I1118 03:00:01.881856 22484 solver.cpp:464] Iteration 1061, lr = 1e-05
I1118 03:00:04.104915 22484 solver.cpp:209] Iteration 1062, loss = 0.217577
I1118 03:00:04.104944 22484 solver.cpp:464] Iteration 1062, lr = 1e-05
I1118 03:00:06.329506 22484 solver.cpp:209] Iteration 1063, loss = 0.168917
I1118 03:00:06.329547 22484 solver.cpp:464] Iteration 1063, lr = 1e-05
I1118 03:00:08.555032 22484 solver.cpp:209] Iteration 1064, loss = 0.202083
I1118 03:00:08.555073 22484 solver.cpp:464] Iteration 1064, lr = 1e-05
I1118 03:00:10.774946 22484 solver.cpp:209] Iteration 1065, loss = 0.350594
I1118 03:00:10.774972 22484 solver.cpp:464] Iteration 1065, lr = 1e-05
I1118 03:00:13.011793 22484 solver.cpp:209] Iteration 1066, loss = 0.357283
I1118 03:00:13.011834 22484 solver.cpp:464] Iteration 1066, lr = 1e-05
I1118 03:00:15.243598 22484 solver.cpp:209] Iteration 1067, loss = 0.50961
I1118 03:00:15.243628 22484 solver.cpp:464] Iteration 1067, lr = 1e-05
I1118 03:00:17.427332 22484 solver.cpp:209] Iteration 1068, loss = 0.143844
I1118 03:00:17.427362 22484 solver.cpp:464] Iteration 1068, lr = 1e-05
I1118 03:00:19.597172 22484 solver.cpp:209] Iteration 1069, loss = 0.248052
I1118 03:00:19.597199 22484 solver.cpp:464] Iteration 1069, lr = 1e-05
I1118 03:00:21.760794 22484 solver.cpp:209] Iteration 1070, loss = 0.220512
I1118 03:00:21.760821 22484 solver.cpp:464] Iteration 1070, lr = 1e-05
I1118 03:00:23.927543 22484 solver.cpp:209] Iteration 1071, loss = 0.302168
I1118 03:00:23.927598 22484 solver.cpp:464] Iteration 1071, lr = 1e-05
I1118 03:00:26.123344 22484 solver.cpp:209] Iteration 1072, loss = 0.22977
I1118 03:00:26.123385 22484 solver.cpp:464] Iteration 1072, lr = 1e-05
I1118 03:00:28.367879 22484 solver.cpp:209] Iteration 1073, loss = 0.209582
I1118 03:00:28.367905 22484 solver.cpp:464] Iteration 1073, lr = 1e-05
I1118 03:00:30.601286 22484 solver.cpp:209] Iteration 1074, loss = 0.208309
I1118 03:00:30.601311 22484 solver.cpp:464] Iteration 1074, lr = 1e-05
I1118 03:00:32.826123 22484 solver.cpp:209] Iteration 1075, loss = 0.274406
I1118 03:00:32.826153 22484 solver.cpp:464] Iteration 1075, lr = 1e-05
I1118 03:00:35.049373 22484 solver.cpp:209] Iteration 1076, loss = 0.303555
I1118 03:00:35.049401 22484 solver.cpp:464] Iteration 1076, lr = 1e-05
I1118 03:00:37.275943 22484 solver.cpp:209] Iteration 1077, loss = 0.437331
I1118 03:00:37.275970 22484 solver.cpp:464] Iteration 1077, lr = 1e-05
I1118 03:00:39.499778 22484 solver.cpp:209] Iteration 1078, loss = 0.350098
I1118 03:00:39.499815 22484 solver.cpp:464] Iteration 1078, lr = 1e-05
I1118 03:00:41.734686 22484 solver.cpp:209] Iteration 1079, loss = 0.305743
I1118 03:00:41.734715 22484 solver.cpp:464] Iteration 1079, lr = 1e-05
I1118 03:00:43.971571 22484 solver.cpp:209] Iteration 1080, loss = 0.296101
I1118 03:00:43.971599 22484 solver.cpp:464] Iteration 1080, lr = 1e-05
I1118 03:00:46.205313 22484 solver.cpp:209] Iteration 1081, loss = 0.337805
I1118 03:00:46.205355 22484 solver.cpp:464] Iteration 1081, lr = 1e-05
I1118 03:00:48.429702 22484 solver.cpp:209] Iteration 1082, loss = 0.0796236
I1118 03:00:48.429744 22484 solver.cpp:464] Iteration 1082, lr = 1e-05
I1118 03:00:50.650079 22484 solver.cpp:209] Iteration 1083, loss = 0.28049
I1118 03:00:50.650117 22484 solver.cpp:464] Iteration 1083, lr = 1e-05
I1118 03:00:52.880446 22484 solver.cpp:209] Iteration 1084, loss = 0.212665
I1118 03:00:52.880475 22484 solver.cpp:464] Iteration 1084, lr = 1e-05
I1118 03:00:55.112483 22484 solver.cpp:209] Iteration 1085, loss = 0.1843
I1118 03:00:55.112537 22484 solver.cpp:464] Iteration 1085, lr = 1e-05
I1118 03:00:57.345690 22484 solver.cpp:209] Iteration 1086, loss = 0.213568
I1118 03:00:57.345718 22484 solver.cpp:464] Iteration 1086, lr = 1e-05
I1118 03:00:59.581684 22484 solver.cpp:209] Iteration 1087, loss = 0.190886
I1118 03:00:59.581724 22484 solver.cpp:464] Iteration 1087, lr = 1e-05
I1118 03:01:01.804296 22484 solver.cpp:209] Iteration 1088, loss = 0.255638
I1118 03:01:01.804323 22484 solver.cpp:464] Iteration 1088, lr = 1e-05
I1118 03:01:04.029122 22484 solver.cpp:209] Iteration 1089, loss = 0.243721
I1118 03:01:04.029162 22484 solver.cpp:464] Iteration 1089, lr = 1e-05
I1118 03:01:06.252650 22484 solver.cpp:209] Iteration 1090, loss = 0.511922
I1118 03:01:06.252691 22484 solver.cpp:464] Iteration 1090, lr = 1e-05
I1118 03:01:08.477174 22484 solver.cpp:209] Iteration 1091, loss = 0.336643
I1118 03:01:08.477202 22484 solver.cpp:464] Iteration 1091, lr = 1e-05
I1118 03:01:10.718925 22484 solver.cpp:209] Iteration 1092, loss = 0.580736
I1118 03:01:10.718952 22484 solver.cpp:464] Iteration 1092, lr = 1e-05
I1118 03:01:12.944979 22484 solver.cpp:209] Iteration 1093, loss = 0.201309
I1118 03:01:12.945020 22484 solver.cpp:464] Iteration 1093, lr = 1e-05
I1118 03:01:15.174650 22484 solver.cpp:209] Iteration 1094, loss = 0.29135
I1118 03:01:15.174679 22484 solver.cpp:464] Iteration 1094, lr = 1e-05
I1118 03:01:17.399844 22484 solver.cpp:209] Iteration 1095, loss = 0.430517
I1118 03:01:17.399873 22484 solver.cpp:464] Iteration 1095, lr = 1e-05
I1118 03:01:19.625259 22484 solver.cpp:209] Iteration 1096, loss = 0.301454
I1118 03:01:19.625288 22484 solver.cpp:464] Iteration 1096, lr = 1e-05
I1118 03:01:21.851021 22484 solver.cpp:209] Iteration 1097, loss = 0.244548
I1118 03:01:21.851050 22484 solver.cpp:464] Iteration 1097, lr = 1e-05
I1118 03:01:24.073179 22484 solver.cpp:209] Iteration 1098, loss = 0.519292
I1118 03:01:24.073209 22484 solver.cpp:464] Iteration 1098, lr = 1e-05
I1118 03:01:26.306555 22484 solver.cpp:209] Iteration 1099, loss = 0.345176
I1118 03:01:26.306669 22484 solver.cpp:464] Iteration 1099, lr = 1e-05
I1118 03:01:26.307279 22484 solver.cpp:264] Iteration 1100, Testing net (#0)
I1118 03:01:40.008328 22484 solver.cpp:305] Test loss: 0.257031
I1118 03:01:40.008358 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 03:01:40.008378 22484 solver.cpp:319]            = 0.98646
I1118 03:01:40.008383 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 03:01:40.008386 22484 solver.cpp:318] mean_score = test_score[2] { = 7} / test_score[3] { = 59 }
I1118 03:01:40.008391 22484 solver.cpp:319]            = 0.118644
I1118 03:01:40.008395 22484 solver.cpp:328]     Test net output #1: accuracy = 0.118644
I1118 03:01:40.008399 22484 solver.cpp:332]     Test net output #2: accuracy = 0.897569
I1118 03:01:40.008404 22484 solver.cpp:334]     Test net output #3: accuracy = 0.552552
I1118 03:01:40.648274 22484 solver.cpp:209] Iteration 1100, loss = 0.337653
I1118 03:01:40.648313 22484 solver.cpp:464] Iteration 1100, lr = 1e-05
I1118 03:01:42.879660 22484 solver.cpp:209] Iteration 1101, loss = 0.25641
I1118 03:01:42.879701 22484 solver.cpp:464] Iteration 1101, lr = 1e-05
I1118 03:01:45.111208 22484 solver.cpp:209] Iteration 1102, loss = 0.629034
I1118 03:01:45.111250 22484 solver.cpp:464] Iteration 1102, lr = 1e-05
I1118 03:01:47.335923 22484 solver.cpp:209] Iteration 1103, loss = 0.077608
I1118 03:01:47.335953 22484 solver.cpp:464] Iteration 1103, lr = 1e-05
I1118 03:01:49.555820 22484 solver.cpp:209] Iteration 1104, loss = 0.36487
I1118 03:01:49.555861 22484 solver.cpp:464] Iteration 1104, lr = 1e-05
I1118 03:01:51.778105 22484 solver.cpp:209] Iteration 1105, loss = 0.115846
I1118 03:01:51.778134 22484 solver.cpp:464] Iteration 1105, lr = 1e-05
I1118 03:01:54.007163 22484 solver.cpp:209] Iteration 1106, loss = 0.395965
I1118 03:01:54.007203 22484 solver.cpp:464] Iteration 1106, lr = 1e-05
I1118 03:01:56.240541 22484 solver.cpp:209] Iteration 1107, loss = 0.174348
I1118 03:01:56.240583 22484 solver.cpp:464] Iteration 1107, lr = 1e-05
I1118 03:01:58.467821 22484 solver.cpp:209] Iteration 1108, loss = 0.203631
I1118 03:01:58.467914 22484 solver.cpp:464] Iteration 1108, lr = 1e-05
I1118 03:02:00.691488 22484 solver.cpp:209] Iteration 1109, loss = 0.226721
I1118 03:02:00.691517 22484 solver.cpp:464] Iteration 1109, lr = 1e-05
I1118 03:02:02.909135 22484 solver.cpp:209] Iteration 1110, loss = 0.151745
I1118 03:02:02.909164 22484 solver.cpp:464] Iteration 1110, lr = 1e-05
I1118 03:02:05.130775 22484 solver.cpp:209] Iteration 1111, loss = 0.446906
I1118 03:02:05.130805 22484 solver.cpp:464] Iteration 1111, lr = 1e-05
I1118 03:02:07.356487 22484 solver.cpp:209] Iteration 1112, loss = 0.405891
I1118 03:02:07.356516 22484 solver.cpp:464] Iteration 1112, lr = 1e-05
I1118 03:02:09.582587 22484 solver.cpp:209] Iteration 1113, loss = 0.439031
I1118 03:02:09.582617 22484 solver.cpp:464] Iteration 1113, lr = 1e-05
I1118 03:02:11.812463 22484 solver.cpp:209] Iteration 1114, loss = 0.414528
I1118 03:02:11.812502 22484 solver.cpp:464] Iteration 1114, lr = 1e-05
I1118 03:02:14.038801 22484 solver.cpp:209] Iteration 1115, loss = 0.254324
I1118 03:02:14.038830 22484 solver.cpp:464] Iteration 1115, lr = 1e-05
I1118 03:02:16.264621 22484 solver.cpp:209] Iteration 1116, loss = 0.0985895
I1118 03:02:16.264649 22484 solver.cpp:464] Iteration 1116, lr = 1e-05
I1118 03:02:18.486524 22484 solver.cpp:209] Iteration 1117, loss = 0.105401
I1118 03:02:18.486564 22484 solver.cpp:464] Iteration 1117, lr = 1e-05
I1118 03:02:20.704624 22484 solver.cpp:209] Iteration 1118, loss = 0.409193
I1118 03:02:20.704665 22484 solver.cpp:464] Iteration 1118, lr = 1e-05
I1118 03:02:22.931627 22484 solver.cpp:209] Iteration 1119, loss = 0.257055
I1118 03:02:22.931653 22484 solver.cpp:464] Iteration 1119, lr = 1e-05
I1118 03:02:25.157965 22484 solver.cpp:209] Iteration 1120, loss = 0.293827
I1118 03:02:25.158006 22484 solver.cpp:464] Iteration 1120, lr = 1e-05
I1118 03:02:27.392416 22484 solver.cpp:209] Iteration 1121, loss = 0.442009
I1118 03:02:27.392444 22484 solver.cpp:464] Iteration 1121, lr = 1e-05
I1118 03:02:29.628399 22484 solver.cpp:209] Iteration 1122, loss = 0.210595
I1118 03:02:29.628470 22484 solver.cpp:464] Iteration 1122, lr = 1e-05
I1118 03:02:31.851042 22484 solver.cpp:209] Iteration 1123, loss = 0.224033
I1118 03:02:31.851070 22484 solver.cpp:464] Iteration 1123, lr = 1e-05
I1118 03:02:34.071298 22484 solver.cpp:209] Iteration 1124, loss = 0.22162
I1118 03:02:34.071326 22484 solver.cpp:464] Iteration 1124, lr = 1e-05
I1118 03:02:36.293092 22484 solver.cpp:209] Iteration 1125, loss = 0.19376
I1118 03:02:36.293120 22484 solver.cpp:464] Iteration 1125, lr = 1e-05
I1118 03:02:38.524466 22484 solver.cpp:209] Iteration 1126, loss = 0.406445
I1118 03:02:38.524507 22484 solver.cpp:464] Iteration 1126, lr = 1e-05
I1118 03:02:40.758303 22484 solver.cpp:209] Iteration 1127, loss = 0.0889482
I1118 03:02:40.758328 22484 solver.cpp:464] Iteration 1127, lr = 1e-05
I1118 03:02:42.981369 22484 solver.cpp:209] Iteration 1128, loss = 0.430264
I1118 03:02:42.981398 22484 solver.cpp:464] Iteration 1128, lr = 1e-05
I1118 03:02:45.210927 22484 solver.cpp:209] Iteration 1129, loss = 0.138169
I1118 03:02:45.210968 22484 solver.cpp:464] Iteration 1129, lr = 1e-05
I1118 03:02:47.436171 22484 solver.cpp:209] Iteration 1130, loss = 0.152278
I1118 03:02:47.436209 22484 solver.cpp:464] Iteration 1130, lr = 1e-05
I1118 03:02:49.663233 22484 solver.cpp:209] Iteration 1131, loss = 0.181348
I1118 03:02:49.663261 22484 solver.cpp:464] Iteration 1131, lr = 1e-05
I1118 03:02:51.886703 22484 solver.cpp:209] Iteration 1132, loss = 0.298365
I1118 03:02:51.886733 22484 solver.cpp:464] Iteration 1132, lr = 1e-05
I1118 03:02:54.104192 22484 solver.cpp:209] Iteration 1133, loss = 0.0780284
I1118 03:02:54.104220 22484 solver.cpp:464] Iteration 1133, lr = 1e-05
I1118 03:02:56.296499 22484 solver.cpp:209] Iteration 1134, loss = 0.404209
I1118 03:02:56.296540 22484 solver.cpp:464] Iteration 1134, lr = 1e-05
I1118 03:02:58.499416 22484 solver.cpp:209] Iteration 1135, loss = 0.304802
I1118 03:02:58.499456 22484 solver.cpp:464] Iteration 1135, lr = 1e-05
I1118 03:03:00.688871 22484 solver.cpp:209] Iteration 1136, loss = 0.194813
I1118 03:03:00.688963 22484 solver.cpp:464] Iteration 1136, lr = 1e-05
I1118 03:03:02.878587 22484 solver.cpp:209] Iteration 1137, loss = 0.274767
I1118 03:03:02.878617 22484 solver.cpp:464] Iteration 1137, lr = 1e-05
I1118 03:03:05.066020 22484 solver.cpp:209] Iteration 1138, loss = 0.389881
I1118 03:03:05.066050 22484 solver.cpp:464] Iteration 1138, lr = 1e-05
I1118 03:03:07.268098 22484 solver.cpp:209] Iteration 1139, loss = 0.212848
I1118 03:03:07.268139 22484 solver.cpp:464] Iteration 1139, lr = 1e-05
I1118 03:03:09.464526 22484 solver.cpp:209] Iteration 1140, loss = 0.126442
I1118 03:03:09.464555 22484 solver.cpp:464] Iteration 1140, lr = 1e-05
I1118 03:03:11.668028 22484 solver.cpp:209] Iteration 1141, loss = 0.228209
I1118 03:03:11.668056 22484 solver.cpp:464] Iteration 1141, lr = 1e-05
I1118 03:03:13.859504 22484 solver.cpp:209] Iteration 1142, loss = 0.177716
I1118 03:03:13.859534 22484 solver.cpp:464] Iteration 1142, lr = 1e-05
I1118 03:03:16.043642 22484 solver.cpp:209] Iteration 1143, loss = 0.296914
I1118 03:03:16.043670 22484 solver.cpp:464] Iteration 1143, lr = 1e-05
I1118 03:03:18.237818 22484 solver.cpp:209] Iteration 1144, loss = 0.544704
I1118 03:03:18.237857 22484 solver.cpp:464] Iteration 1144, lr = 1e-05
I1118 03:03:20.437901 22484 solver.cpp:209] Iteration 1145, loss = 0.180373
I1118 03:03:20.437929 22484 solver.cpp:464] Iteration 1145, lr = 1e-05
I1118 03:03:22.631533 22484 solver.cpp:209] Iteration 1146, loss = 0.367736
I1118 03:03:22.631562 22484 solver.cpp:464] Iteration 1146, lr = 1e-05
I1118 03:03:24.824587 22484 solver.cpp:209] Iteration 1147, loss = 0.126969
I1118 03:03:24.824616 22484 solver.cpp:464] Iteration 1147, lr = 1e-05
I1118 03:03:27.050573 22484 solver.cpp:209] Iteration 1148, loss = 0.0677935
I1118 03:03:27.050626 22484 solver.cpp:464] Iteration 1148, lr = 1e-05
I1118 03:03:29.294045 22484 solver.cpp:209] Iteration 1149, loss = 0.185304
I1118 03:03:29.294073 22484 solver.cpp:464] Iteration 1149, lr = 1e-05
I1118 03:03:29.294693 22484 solver.cpp:264] Iteration 1150, Testing net (#0)
I1118 03:03:43.296762 22484 solver.cpp:305] Test loss: 0.262634
I1118 03:03:43.296856 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 03:03:43.296865 22484 solver.cpp:319]            = 0.98646
I1118 03:03:43.296869 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 03:03:43.296875 22484 solver.cpp:318] mean_score = test_score[2] { = 3} / test_score[3] { = 59 }
I1118 03:03:43.296880 22484 solver.cpp:319]            = 0.0508475
I1118 03:03:43.296883 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0508475
I1118 03:03:43.296887 22484 solver.cpp:332]     Test net output #2: accuracy = 0.890625
I1118 03:03:43.296892 22484 solver.cpp:334]     Test net output #3: accuracy = 0.518654
I1118 03:03:43.948820 22484 solver.cpp:209] Iteration 1150, loss = 0.206575
I1118 03:03:43.948849 22484 solver.cpp:464] Iteration 1150, lr = 1e-05
I1118 03:03:46.191622 22484 solver.cpp:209] Iteration 1151, loss = 0.0864593
I1118 03:03:46.191651 22484 solver.cpp:464] Iteration 1151, lr = 1e-05
I1118 03:03:48.427616 22484 solver.cpp:209] Iteration 1152, loss = 0.332065
I1118 03:03:48.427655 22484 solver.cpp:464] Iteration 1152, lr = 1e-05
I1118 03:03:50.669160 22484 solver.cpp:209] Iteration 1153, loss = 0.221794
I1118 03:03:50.669189 22484 solver.cpp:464] Iteration 1153, lr = 1e-05
I1118 03:03:52.918350 22484 solver.cpp:209] Iteration 1154, loss = 0.179841
I1118 03:03:52.918380 22484 solver.cpp:464] Iteration 1154, lr = 1e-05
I1118 03:03:55.164434 22484 solver.cpp:209] Iteration 1155, loss = 0.099477
I1118 03:03:55.164475 22484 solver.cpp:464] Iteration 1155, lr = 1e-05
I1118 03:03:57.402500 22484 solver.cpp:209] Iteration 1156, loss = 0.116003
I1118 03:03:57.402528 22484 solver.cpp:464] Iteration 1156, lr = 1e-05
I1118 03:03:59.620889 22484 solver.cpp:209] Iteration 1157, loss = 0.386375
I1118 03:03:59.620929 22484 solver.cpp:464] Iteration 1157, lr = 1e-05
I1118 03:04:01.807354 22484 solver.cpp:209] Iteration 1158, loss = 0.314036
I1118 03:04:01.807395 22484 solver.cpp:464] Iteration 1158, lr = 1e-05
I1118 03:04:03.991578 22484 solver.cpp:209] Iteration 1159, loss = 0.0911478
I1118 03:04:03.991607 22484 solver.cpp:464] Iteration 1159, lr = 1e-05
I1118 03:04:06.186239 22484 solver.cpp:209] Iteration 1160, loss = 0.229664
I1118 03:04:06.186267 22484 solver.cpp:464] Iteration 1160, lr = 1e-05
I1118 03:04:08.387150 22484 solver.cpp:209] Iteration 1161, loss = 0.5726
I1118 03:04:08.387189 22484 solver.cpp:464] Iteration 1161, lr = 1e-05
I1118 03:04:10.581898 22484 solver.cpp:209] Iteration 1162, loss = 0.135426
I1118 03:04:10.581926 22484 solver.cpp:464] Iteration 1162, lr = 1e-05
I1118 03:04:12.772608 22484 solver.cpp:209] Iteration 1163, loss = 0.250268
I1118 03:04:12.772650 22484 solver.cpp:464] Iteration 1163, lr = 1e-05
I1118 03:04:14.956012 22484 solver.cpp:209] Iteration 1164, loss = 0.193061
I1118 03:04:14.956102 22484 solver.cpp:464] Iteration 1164, lr = 1e-05
I1118 03:04:17.147547 22484 solver.cpp:209] Iteration 1165, loss = 0.259735
I1118 03:04:17.147577 22484 solver.cpp:464] Iteration 1165, lr = 1e-05
I1118 03:04:19.364326 22484 solver.cpp:209] Iteration 1166, loss = 0.211722
I1118 03:04:19.364356 22484 solver.cpp:464] Iteration 1166, lr = 1e-05
I1118 03:04:21.556802 22484 solver.cpp:209] Iteration 1167, loss = 0.226166
I1118 03:04:21.556840 22484 solver.cpp:464] Iteration 1167, lr = 1e-05
I1118 03:04:23.759645 22484 solver.cpp:209] Iteration 1168, loss = 0.322522
I1118 03:04:23.759687 22484 solver.cpp:464] Iteration 1168, lr = 1e-05
I1118 03:04:25.952123 22484 solver.cpp:209] Iteration 1169, loss = 0.415519
I1118 03:04:25.952152 22484 solver.cpp:464] Iteration 1169, lr = 1e-05
I1118 03:04:28.149608 22484 solver.cpp:209] Iteration 1170, loss = 0.458434
I1118 03:04:28.149637 22484 solver.cpp:464] Iteration 1170, lr = 1e-05
I1118 03:04:30.342114 22484 solver.cpp:209] Iteration 1171, loss = 0.136259
I1118 03:04:30.342156 22484 solver.cpp:464] Iteration 1171, lr = 1e-05
I1118 03:04:32.549101 22484 solver.cpp:209] Iteration 1172, loss = 0.213247
I1118 03:04:32.549130 22484 solver.cpp:464] Iteration 1172, lr = 1e-05
I1118 03:04:34.778831 22484 solver.cpp:209] Iteration 1173, loss = 0.336692
I1118 03:04:34.778859 22484 solver.cpp:464] Iteration 1173, lr = 1e-05
I1118 03:04:37.015665 22484 solver.cpp:209] Iteration 1174, loss = 0.17529
I1118 03:04:37.015705 22484 solver.cpp:464] Iteration 1174, lr = 1e-05
I1118 03:04:39.249861 22484 solver.cpp:209] Iteration 1175, loss = 0.197738
I1118 03:04:39.249902 22484 solver.cpp:464] Iteration 1175, lr = 1e-05
I1118 03:04:41.496848 22484 solver.cpp:209] Iteration 1176, loss = 0.272759
I1118 03:04:41.496877 22484 solver.cpp:464] Iteration 1176, lr = 1e-05
I1118 03:04:43.739090 22484 solver.cpp:209] Iteration 1177, loss = 0.13608
I1118 03:04:43.739119 22484 solver.cpp:464] Iteration 1177, lr = 1e-05
I1118 03:04:45.979817 22484 solver.cpp:209] Iteration 1178, loss = 0.330256
I1118 03:04:45.979915 22484 solver.cpp:464] Iteration 1178, lr = 1e-05
I1118 03:04:48.230818 22484 solver.cpp:209] Iteration 1179, loss = 0.288401
I1118 03:04:48.230846 22484 solver.cpp:464] Iteration 1179, lr = 1e-05
I1118 03:04:50.480393 22484 solver.cpp:209] Iteration 1180, loss = 0.338537
I1118 03:04:50.480422 22484 solver.cpp:464] Iteration 1180, lr = 1e-05
I1118 03:04:52.749455 22484 solver.cpp:209] Iteration 1181, loss = 0.302743
I1118 03:04:52.749485 22484 solver.cpp:464] Iteration 1181, lr = 1e-05
I1118 03:04:54.993319 22484 solver.cpp:209] Iteration 1182, loss = 0.210624
I1118 03:04:54.993347 22484 solver.cpp:464] Iteration 1182, lr = 1e-05
I1118 03:04:57.216681 22484 solver.cpp:209] Iteration 1183, loss = 0.24172
I1118 03:04:57.216711 22484 solver.cpp:464] Iteration 1183, lr = 1e-05
I1118 03:04:59.409240 22484 solver.cpp:209] Iteration 1184, loss = 0.223308
I1118 03:04:59.409281 22484 solver.cpp:464] Iteration 1184, lr = 1e-05
I1118 03:05:01.603576 22484 solver.cpp:209] Iteration 1185, loss = 0.142586
I1118 03:05:01.603618 22484 solver.cpp:464] Iteration 1185, lr = 1e-05
I1118 03:05:03.808732 22484 solver.cpp:209] Iteration 1186, loss = 0.355002
I1118 03:05:03.808760 22484 solver.cpp:464] Iteration 1186, lr = 1e-05
I1118 03:05:06.005550 22484 solver.cpp:209] Iteration 1187, loss = 0.236221
I1118 03:05:06.005592 22484 solver.cpp:464] Iteration 1187, lr = 1e-05
I1118 03:05:08.205812 22484 solver.cpp:209] Iteration 1188, loss = 0.191775
I1118 03:05:08.205842 22484 solver.cpp:464] Iteration 1188, lr = 1e-05
I1118 03:05:10.393627 22484 solver.cpp:209] Iteration 1189, loss = 0.327781
I1118 03:05:10.393668 22484 solver.cpp:464] Iteration 1189, lr = 1e-05
I1118 03:05:12.590972 22484 solver.cpp:209] Iteration 1190, loss = 0.161328
I1118 03:05:12.591001 22484 solver.cpp:464] Iteration 1190, lr = 1e-05
I1118 03:05:14.780230 22484 solver.cpp:209] Iteration 1191, loss = 0.38937
I1118 03:05:14.780259 22484 solver.cpp:464] Iteration 1191, lr = 1e-05
I1118 03:05:16.979552 22484 solver.cpp:209] Iteration 1192, loss = 0.161074
I1118 03:05:16.979643 22484 solver.cpp:464] Iteration 1192, lr = 1e-05
I1118 03:05:19.179292 22484 solver.cpp:209] Iteration 1193, loss = 0.475878
I1118 03:05:19.179333 22484 solver.cpp:464] Iteration 1193, lr = 1e-05
I1118 03:05:21.370173 22484 solver.cpp:209] Iteration 1194, loss = 0.315999
I1118 03:05:21.370203 22484 solver.cpp:464] Iteration 1194, lr = 1e-05
I1118 03:05:23.573122 22484 solver.cpp:209] Iteration 1195, loss = 0.423215
I1118 03:05:23.573150 22484 solver.cpp:464] Iteration 1195, lr = 1e-05
I1118 03:05:25.796890 22484 solver.cpp:209] Iteration 1196, loss = 0.285706
I1118 03:05:25.796918 22484 solver.cpp:464] Iteration 1196, lr = 1e-05
I1118 03:05:28.023455 22484 solver.cpp:209] Iteration 1197, loss = 0.421214
I1118 03:05:28.023484 22484 solver.cpp:464] Iteration 1197, lr = 1e-05
I1118 03:05:30.261939 22484 solver.cpp:209] Iteration 1198, loss = 0.323153
I1118 03:05:30.261979 22484 solver.cpp:464] Iteration 1198, lr = 1e-05
I1118 03:05:32.490453 22484 solver.cpp:209] Iteration 1199, loss = 0.267372
I1118 03:05:32.490483 22484 solver.cpp:464] Iteration 1199, lr = 1e-05
I1118 03:05:32.491083 22484 solver.cpp:264] Iteration 1200, Testing net (#0)
I1118 03:05:46.385156 22484 solver.cpp:305] Test loss: 0.256992
I1118 03:05:46.385185 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 03:05:46.385205 22484 solver.cpp:319]            = 0.98646
I1118 03:05:46.385210 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 03:05:46.385215 22484 solver.cpp:318] mean_score = test_score[2] { = 3} / test_score[3] { = 59 }
I1118 03:05:46.385218 22484 solver.cpp:319]            = 0.0508475
I1118 03:05:46.385222 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0508475
I1118 03:05:46.385227 22484 solver.cpp:332]     Test net output #2: accuracy = 0.890625
I1118 03:05:46.385231 22484 solver.cpp:334]     Test net output #3: accuracy = 0.518654
I1118 03:05:47.034898 22484 solver.cpp:209] Iteration 1200, loss = 0.277438
I1118 03:05:47.034971 22484 solver.cpp:464] Iteration 1200, lr = 1e-05
I1118 03:05:49.265650 22484 solver.cpp:209] Iteration 1201, loss = 0.51198
I1118 03:05:49.265692 22484 solver.cpp:464] Iteration 1201, lr = 1e-05
I1118 03:05:51.495209 22484 solver.cpp:209] Iteration 1202, loss = 0.555559
I1118 03:05:51.495251 22484 solver.cpp:464] Iteration 1202, lr = 1e-05
I1118 03:05:53.720381 22484 solver.cpp:209] Iteration 1203, loss = 0.259614
I1118 03:05:53.720422 22484 solver.cpp:464] Iteration 1203, lr = 1e-05
I1118 03:05:55.948783 22484 solver.cpp:209] Iteration 1204, loss = 0.285467
I1118 03:05:55.948823 22484 solver.cpp:464] Iteration 1204, lr = 1e-05
I1118 03:05:58.176780 22484 solver.cpp:209] Iteration 1205, loss = 0.34018
I1118 03:05:58.176821 22484 solver.cpp:464] Iteration 1205, lr = 1e-05
I1118 03:06:00.408419 22484 solver.cpp:209] Iteration 1206, loss = 0.26207
I1118 03:06:00.408459 22484 solver.cpp:464] Iteration 1206, lr = 1e-05
I1118 03:06:02.635311 22484 solver.cpp:209] Iteration 1207, loss = 0.316555
I1118 03:06:02.635340 22484 solver.cpp:464] Iteration 1207, lr = 1e-05
I1118 03:06:04.866461 22484 solver.cpp:209] Iteration 1208, loss = 0.118993
I1118 03:06:04.866503 22484 solver.cpp:464] Iteration 1208, lr = 1e-05
I1118 03:06:07.101801 22484 solver.cpp:209] Iteration 1209, loss = 0.564316
I1118 03:06:07.101843 22484 solver.cpp:464] Iteration 1209, lr = 1e-05
I1118 03:06:09.328526 22484 solver.cpp:209] Iteration 1210, loss = 0.176558
I1118 03:06:09.328567 22484 solver.cpp:464] Iteration 1210, lr = 1e-05
I1118 03:06:11.555618 22484 solver.cpp:209] Iteration 1211, loss = 0.203636
I1118 03:06:11.555646 22484 solver.cpp:464] Iteration 1211, lr = 1e-05
I1118 03:06:13.776000 22484 solver.cpp:209] Iteration 1212, loss = 0.161381
I1118 03:06:13.776031 22484 solver.cpp:464] Iteration 1212, lr = 1e-05
I1118 03:06:16.004763 22484 solver.cpp:209] Iteration 1213, loss = 0.27545
I1118 03:06:16.004806 22484 solver.cpp:464] Iteration 1213, lr = 1e-05
I1118 03:06:18.235904 22484 solver.cpp:209] Iteration 1214, loss = 0.244536
I1118 03:06:18.235960 22484 solver.cpp:464] Iteration 1214, lr = 1e-05
I1118 03:06:20.466233 22484 solver.cpp:209] Iteration 1215, loss = 0.399148
I1118 03:06:20.466274 22484 solver.cpp:464] Iteration 1215, lr = 1e-05
I1118 03:06:22.696456 22484 solver.cpp:209] Iteration 1216, loss = 0.451124
I1118 03:06:22.696496 22484 solver.cpp:464] Iteration 1216, lr = 1e-05
I1118 03:06:24.915853 22484 solver.cpp:209] Iteration 1217, loss = 0.350634
I1118 03:06:24.915880 22484 solver.cpp:464] Iteration 1217, lr = 1e-05
I1118 03:06:27.141957 22484 solver.cpp:209] Iteration 1218, loss = 0.159336
I1118 03:06:27.141985 22484 solver.cpp:464] Iteration 1218, lr = 1e-05
I1118 03:06:29.376880 22484 solver.cpp:209] Iteration 1219, loss = 0.125692
I1118 03:06:29.376909 22484 solver.cpp:464] Iteration 1219, lr = 1e-05
I1118 03:06:31.610008 22484 solver.cpp:209] Iteration 1220, loss = 0.110626
I1118 03:06:31.610038 22484 solver.cpp:464] Iteration 1220, lr = 1e-05
I1118 03:06:33.841886 22484 solver.cpp:209] Iteration 1221, loss = 0.447719
I1118 03:06:33.841914 22484 solver.cpp:464] Iteration 1221, lr = 1e-05
I1118 03:06:36.072368 22484 solver.cpp:209] Iteration 1222, loss = 0.182165
I1118 03:06:36.072396 22484 solver.cpp:464] Iteration 1222, lr = 1e-05
I1118 03:06:38.299267 22484 solver.cpp:209] Iteration 1223, loss = 0.316419
I1118 03:06:38.299294 22484 solver.cpp:464] Iteration 1223, lr = 1e-05
I1118 03:06:40.528504 22484 solver.cpp:209] Iteration 1224, loss = 0.298315
I1118 03:06:40.528532 22484 solver.cpp:464] Iteration 1224, lr = 1e-05
I1118 03:06:42.759399 22484 solver.cpp:209] Iteration 1225, loss = 0.28872
I1118 03:06:42.759441 22484 solver.cpp:464] Iteration 1225, lr = 1e-05
I1118 03:06:44.989804 22484 solver.cpp:209] Iteration 1226, loss = 0.176067
I1118 03:06:44.989831 22484 solver.cpp:464] Iteration 1226, lr = 1e-05
I1118 03:06:47.217202 22484 solver.cpp:209] Iteration 1227, loss = 0.3269
I1118 03:06:47.217231 22484 solver.cpp:464] Iteration 1227, lr = 1e-05
I1118 03:06:49.445112 22484 solver.cpp:209] Iteration 1228, loss = 0.276126
I1118 03:06:49.445209 22484 solver.cpp:464] Iteration 1228, lr = 1e-05
I1118 03:06:51.649560 22484 solver.cpp:209] Iteration 1229, loss = 0.477398
I1118 03:06:51.649601 22484 solver.cpp:464] Iteration 1229, lr = 1e-05
I1118 03:06:53.848011 22484 solver.cpp:209] Iteration 1230, loss = 0.206432
I1118 03:06:53.848040 22484 solver.cpp:464] Iteration 1230, lr = 1e-05
I1118 03:06:56.041280 22484 solver.cpp:209] Iteration 1231, loss = 0.227343
I1118 03:06:56.041309 22484 solver.cpp:464] Iteration 1231, lr = 1e-05
I1118 03:06:58.241216 22484 solver.cpp:209] Iteration 1232, loss = 0.121518
I1118 03:06:58.241245 22484 solver.cpp:464] Iteration 1232, lr = 1e-05
I1118 03:07:00.435856 22484 solver.cpp:209] Iteration 1233, loss = 0.195853
I1118 03:07:00.435884 22484 solver.cpp:464] Iteration 1233, lr = 1e-05
I1118 03:07:02.633617 22484 solver.cpp:209] Iteration 1234, loss = 0.172018
I1118 03:07:02.633646 22484 solver.cpp:464] Iteration 1234, lr = 1e-05
I1118 03:07:04.830574 22484 solver.cpp:209] Iteration 1235, loss = 0.314402
I1118 03:07:04.830624 22484 solver.cpp:464] Iteration 1235, lr = 1e-05
I1118 03:07:07.026124 22484 solver.cpp:209] Iteration 1236, loss = 0.15556
I1118 03:07:07.026165 22484 solver.cpp:464] Iteration 1236, lr = 1e-05
I1118 03:07:09.213255 22484 solver.cpp:209] Iteration 1237, loss = 0.460491
I1118 03:07:09.213282 22484 solver.cpp:464] Iteration 1237, lr = 1e-05
I1118 03:07:11.406605 22484 solver.cpp:209] Iteration 1238, loss = 0.27242
I1118 03:07:11.406633 22484 solver.cpp:464] Iteration 1238, lr = 1e-05
I1118 03:07:13.621467 22484 solver.cpp:209] Iteration 1239, loss = 0.13534
I1118 03:07:13.621507 22484 solver.cpp:464] Iteration 1239, lr = 1e-05
I1118 03:07:15.813393 22484 solver.cpp:209] Iteration 1240, loss = 0.282851
I1118 03:07:15.813421 22484 solver.cpp:464] Iteration 1240, lr = 1e-05
I1118 03:07:18.010889 22484 solver.cpp:209] Iteration 1241, loss = 0.375244
I1118 03:07:18.010927 22484 solver.cpp:464] Iteration 1241, lr = 1e-05
I1118 03:07:20.206562 22484 solver.cpp:209] Iteration 1242, loss = 0.147056
I1118 03:07:20.206668 22484 solver.cpp:464] Iteration 1242, lr = 1e-05
I1118 03:07:22.434247 22484 solver.cpp:209] Iteration 1243, loss = 0.140904
I1118 03:07:22.434275 22484 solver.cpp:464] Iteration 1243, lr = 1e-05
I1118 03:07:24.663540 22484 solver.cpp:209] Iteration 1244, loss = 0.233031
I1118 03:07:24.663581 22484 solver.cpp:464] Iteration 1244, lr = 1e-05
I1118 03:07:26.893487 22484 solver.cpp:209] Iteration 1245, loss = 0.287094
I1118 03:07:26.893515 22484 solver.cpp:464] Iteration 1245, lr = 1e-05
I1118 03:07:29.132189 22484 solver.cpp:209] Iteration 1246, loss = 0.19859
I1118 03:07:29.132217 22484 solver.cpp:464] Iteration 1246, lr = 1e-05
I1118 03:07:31.356492 22484 solver.cpp:209] Iteration 1247, loss = 0.330935
I1118 03:07:31.356521 22484 solver.cpp:464] Iteration 1247, lr = 1e-05
I1118 03:07:33.587857 22484 solver.cpp:209] Iteration 1248, loss = 0.192573
I1118 03:07:33.587884 22484 solver.cpp:464] Iteration 1248, lr = 1e-05
I1118 03:07:35.811153 22484 solver.cpp:209] Iteration 1249, loss = 0.359628
I1118 03:07:35.811194 22484 solver.cpp:464] Iteration 1249, lr = 1e-05
I1118 03:07:35.811801 22484 solver.cpp:264] Iteration 1250, Testing net (#0)
I1118 03:07:49.690587 22484 solver.cpp:305] Test loss: 0.250273
I1118 03:07:49.690628 22484 solver.cpp:318] mean_score = test_score[0] { = 514} / test_score[1] { = 517 }
I1118 03:07:49.690635 22484 solver.cpp:319]            = 0.994197
I1118 03:07:49.690639 22484 solver.cpp:328]     Test net output #0: accuracy = 0.994197
I1118 03:07:49.690644 22484 solver.cpp:318] mean_score = test_score[2] { = 5} / test_score[3] { = 59 }
I1118 03:07:49.690649 22484 solver.cpp:319]            = 0.0847458
I1118 03:07:49.690652 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0847458
I1118 03:07:49.690656 22484 solver.cpp:332]     Test net output #2: accuracy = 0.901042
I1118 03:07:49.690660 22484 solver.cpp:334]     Test net output #3: accuracy = 0.539472
I1118 03:07:50.336890 22484 solver.cpp:209] Iteration 1250, loss = 0.0617805
I1118 03:07:50.336949 22484 solver.cpp:464] Iteration 1250, lr = 1e-05
I1118 03:07:52.559257 22484 solver.cpp:209] Iteration 1251, loss = 0.136968
I1118 03:07:52.559285 22484 solver.cpp:464] Iteration 1251, lr = 1e-05
I1118 03:07:54.782366 22484 solver.cpp:209] Iteration 1252, loss = 0.218411
I1118 03:07:54.782395 22484 solver.cpp:464] Iteration 1252, lr = 1e-05
I1118 03:07:57.016744 22484 solver.cpp:209] Iteration 1253, loss = 0.224481
I1118 03:07:57.016773 22484 solver.cpp:464] Iteration 1253, lr = 1e-05
I1118 03:07:59.244632 22484 solver.cpp:209] Iteration 1254, loss = 0.207748
I1118 03:07:59.244673 22484 solver.cpp:464] Iteration 1254, lr = 1e-05
I1118 03:08:01.474138 22484 solver.cpp:209] Iteration 1255, loss = 0.266467
I1118 03:08:01.474164 22484 solver.cpp:464] Iteration 1255, lr = 1e-05
I1118 03:08:03.706492 22484 solver.cpp:209] Iteration 1256, loss = 0.204744
I1118 03:08:03.706521 22484 solver.cpp:464] Iteration 1256, lr = 1e-05
I1118 03:08:05.933765 22484 solver.cpp:209] Iteration 1257, loss = 0.26028
I1118 03:08:05.933794 22484 solver.cpp:464] Iteration 1257, lr = 1e-05
I1118 03:08:08.161149 22484 solver.cpp:209] Iteration 1258, loss = 0.129788
I1118 03:08:08.161187 22484 solver.cpp:464] Iteration 1258, lr = 1e-05
I1118 03:08:10.378428 22484 solver.cpp:209] Iteration 1259, loss = 0.149598
I1118 03:08:10.378468 22484 solver.cpp:464] Iteration 1259, lr = 1e-05
I1118 03:08:12.607050 22484 solver.cpp:209] Iteration 1260, loss = 0.168477
I1118 03:08:12.607079 22484 solver.cpp:464] Iteration 1260, lr = 1e-05
I1118 03:08:14.842164 22484 solver.cpp:209] Iteration 1261, loss = 0.37097
I1118 03:08:14.842193 22484 solver.cpp:464] Iteration 1261, lr = 1e-05
I1118 03:08:17.072692 22484 solver.cpp:209] Iteration 1262, loss = 0.059385
I1118 03:08:17.072721 22484 solver.cpp:464] Iteration 1262, lr = 1e-05
I1118 03:08:19.306517 22484 solver.cpp:209] Iteration 1263, loss = 0.43181
I1118 03:08:19.306546 22484 solver.cpp:464] Iteration 1263, lr = 1e-05
I1118 03:08:21.528331 22484 solver.cpp:209] Iteration 1264, loss = 0.426645
I1118 03:08:21.528420 22484 solver.cpp:464] Iteration 1264, lr = 1e-05
I1118 03:08:23.757470 22484 solver.cpp:209] Iteration 1265, loss = 0.131486
I1118 03:08:23.757510 22484 solver.cpp:464] Iteration 1265, lr = 1e-05
I1118 03:08:25.983331 22484 solver.cpp:209] Iteration 1266, loss = 0.19398
I1118 03:08:25.983361 22484 solver.cpp:464] Iteration 1266, lr = 1e-05
I1118 03:08:28.209661 22484 solver.cpp:209] Iteration 1267, loss = 0.293795
I1118 03:08:28.209688 22484 solver.cpp:464] Iteration 1267, lr = 1e-05
I1118 03:08:30.445792 22484 solver.cpp:209] Iteration 1268, loss = 0.221366
I1118 03:08:30.445822 22484 solver.cpp:464] Iteration 1268, lr = 1e-05
I1118 03:08:32.673213 22484 solver.cpp:209] Iteration 1269, loss = 0.223996
I1118 03:08:32.673254 22484 solver.cpp:464] Iteration 1269, lr = 1e-05
I1118 03:08:34.904973 22484 solver.cpp:209] Iteration 1270, loss = 0.41203
I1118 03:08:34.905001 22484 solver.cpp:464] Iteration 1270, lr = 1e-05
I1118 03:08:37.129962 22484 solver.cpp:209] Iteration 1271, loss = 0.33584
I1118 03:08:37.130003 22484 solver.cpp:464] Iteration 1271, lr = 1e-05
I1118 03:08:39.352759 22484 solver.cpp:209] Iteration 1272, loss = 0.673406
I1118 03:08:39.352787 22484 solver.cpp:464] Iteration 1272, lr = 1e-05
I1118 03:08:41.580947 22484 solver.cpp:209] Iteration 1273, loss = 0.233882
I1118 03:08:41.580988 22484 solver.cpp:464] Iteration 1273, lr = 1e-05
I1118 03:08:43.807638 22484 solver.cpp:209] Iteration 1274, loss = 0.0980427
I1118 03:08:43.807668 22484 solver.cpp:464] Iteration 1274, lr = 1e-05
I1118 03:08:46.039110 22484 solver.cpp:209] Iteration 1275, loss = 0.265612
I1118 03:08:46.039152 22484 solver.cpp:464] Iteration 1275, lr = 1e-05
I1118 03:08:48.275285 22484 solver.cpp:209] Iteration 1276, loss = 0.339554
I1118 03:08:48.275326 22484 solver.cpp:464] Iteration 1276, lr = 1e-05
I1118 03:08:50.501608 22484 solver.cpp:209] Iteration 1277, loss = 0.159201
I1118 03:08:50.501649 22484 solver.cpp:464] Iteration 1277, lr = 1e-05
I1118 03:08:52.725826 22484 solver.cpp:209] Iteration 1278, loss = 0.306718
I1118 03:08:52.725916 22484 solver.cpp:464] Iteration 1278, lr = 1e-05
I1118 03:08:54.945124 22484 solver.cpp:209] Iteration 1279, loss = 0.238766
I1118 03:08:54.945154 22484 solver.cpp:464] Iteration 1279, lr = 1e-05
I1118 03:08:57.171555 22484 solver.cpp:209] Iteration 1280, loss = 0.076632
I1118 03:08:57.171598 22484 solver.cpp:464] Iteration 1280, lr = 1e-05
I1118 03:08:59.407752 22484 solver.cpp:209] Iteration 1281, loss = 0.273912
I1118 03:08:59.407793 22484 solver.cpp:464] Iteration 1281, lr = 1e-05
I1118 03:09:01.634249 22484 solver.cpp:209] Iteration 1282, loss = 0.481303
I1118 03:09:01.634275 22484 solver.cpp:464] Iteration 1282, lr = 1e-05
I1118 03:09:03.866349 22484 solver.cpp:209] Iteration 1283, loss = 0.240971
I1118 03:09:03.866391 22484 solver.cpp:464] Iteration 1283, lr = 1e-05
I1118 03:09:06.087084 22484 solver.cpp:209] Iteration 1284, loss = 0.346084
I1118 03:09:06.087112 22484 solver.cpp:464] Iteration 1284, lr = 1e-05
I1118 03:09:08.310555 22484 solver.cpp:209] Iteration 1285, loss = 0.232414
I1118 03:09:08.310602 22484 solver.cpp:464] Iteration 1285, lr = 1e-05
I1118 03:09:10.510702 22484 solver.cpp:209] Iteration 1286, loss = 0.293746
I1118 03:09:10.510732 22484 solver.cpp:464] Iteration 1286, lr = 1e-05
I1118 03:09:12.706482 22484 solver.cpp:209] Iteration 1287, loss = 0.176763
I1118 03:09:12.706522 22484 solver.cpp:464] Iteration 1287, lr = 1e-05
I1118 03:09:14.906846 22484 solver.cpp:209] Iteration 1288, loss = 0.165075
I1118 03:09:14.906875 22484 solver.cpp:464] Iteration 1288, lr = 1e-05
I1118 03:09:17.099879 22484 solver.cpp:209] Iteration 1289, loss = 0.393229
I1118 03:09:17.099908 22484 solver.cpp:464] Iteration 1289, lr = 1e-05
I1118 03:09:19.291683 22484 solver.cpp:209] Iteration 1290, loss = 0.222307
I1118 03:09:19.291712 22484 solver.cpp:464] Iteration 1290, lr = 1e-05
I1118 03:09:21.486384 22484 solver.cpp:209] Iteration 1291, loss = 0.215242
I1118 03:09:21.486425 22484 solver.cpp:464] Iteration 1291, lr = 1e-05
I1118 03:09:23.679098 22484 solver.cpp:209] Iteration 1292, loss = 0.15313
I1118 03:09:23.679193 22484 solver.cpp:464] Iteration 1292, lr = 1e-05
I1118 03:09:25.875515 22484 solver.cpp:209] Iteration 1293, loss = 0.225228
I1118 03:09:25.875542 22484 solver.cpp:464] Iteration 1293, lr = 1e-05
I1118 03:09:28.071295 22484 solver.cpp:209] Iteration 1294, loss = 0.364372
I1118 03:09:28.071322 22484 solver.cpp:464] Iteration 1294, lr = 1e-05
I1118 03:09:30.269860 22484 solver.cpp:209] Iteration 1295, loss = 0.305821
I1118 03:09:30.269901 22484 solver.cpp:464] Iteration 1295, lr = 1e-05
I1118 03:09:32.465481 22484 solver.cpp:209] Iteration 1296, loss = 0.38094
I1118 03:09:32.465523 22484 solver.cpp:464] Iteration 1296, lr = 1e-05
I1118 03:09:34.661383 22484 solver.cpp:209] Iteration 1297, loss = 0.586201
I1118 03:09:34.661412 22484 solver.cpp:464] Iteration 1297, lr = 1e-05
I1118 03:09:36.856247 22484 solver.cpp:209] Iteration 1298, loss = 0.243903
I1118 03:09:36.856276 22484 solver.cpp:464] Iteration 1298, lr = 1e-05
I1118 03:09:39.057384 22484 solver.cpp:209] Iteration 1299, loss = 0.297772
I1118 03:09:39.057412 22484 solver.cpp:464] Iteration 1299, lr = 1e-05
I1118 03:09:39.058001 22484 solver.cpp:264] Iteration 1300, Testing net (#0)
I1118 03:09:52.913149 22484 solver.cpp:305] Test loss: 0.258013
I1118 03:09:52.913192 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 03:09:52.913199 22484 solver.cpp:319]            = 0.98646
I1118 03:09:52.913203 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 03:09:52.913208 22484 solver.cpp:318] mean_score = test_score[2] { = 5} / test_score[3] { = 59 }
I1118 03:09:52.913213 22484 solver.cpp:319]            = 0.0847458
I1118 03:09:52.913216 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0847458
I1118 03:09:52.913220 22484 solver.cpp:332]     Test net output #2: accuracy = 0.894097
I1118 03:09:52.913224 22484 solver.cpp:334]     Test net output #3: accuracy = 0.535603
I1118 03:09:53.558998 22484 solver.cpp:209] Iteration 1300, loss = 0.365493
I1118 03:09:53.559037 22484 solver.cpp:464] Iteration 1300, lr = 1e-05
I1118 03:09:55.792462 22484 solver.cpp:209] Iteration 1301, loss = 0.398074
I1118 03:09:55.792521 22484 solver.cpp:464] Iteration 1301, lr = 1e-05
I1118 03:09:58.025075 22484 solver.cpp:209] Iteration 1302, loss = 0.347894
I1118 03:09:58.025105 22484 solver.cpp:464] Iteration 1302, lr = 1e-05
I1118 03:10:00.249114 22484 solver.cpp:209] Iteration 1303, loss = 0.270823
I1118 03:10:00.249143 22484 solver.cpp:464] Iteration 1303, lr = 1e-05
I1118 03:10:02.471456 22484 solver.cpp:209] Iteration 1304, loss = 0.457655
I1118 03:10:02.471496 22484 solver.cpp:464] Iteration 1304, lr = 1e-05
I1118 03:10:04.694978 22484 solver.cpp:209] Iteration 1305, loss = 0.435297
I1118 03:10:04.695019 22484 solver.cpp:464] Iteration 1305, lr = 1e-05
I1118 03:10:06.922725 22484 solver.cpp:209] Iteration 1306, loss = 0.37873
I1118 03:10:06.922755 22484 solver.cpp:464] Iteration 1306, lr = 1e-05
I1118 03:10:09.156206 22484 solver.cpp:209] Iteration 1307, loss = 0.32517
I1118 03:10:09.156235 22484 solver.cpp:464] Iteration 1307, lr = 1e-05
I1118 03:10:11.389572 22484 solver.cpp:209] Iteration 1308, loss = 0.268103
I1118 03:10:11.389601 22484 solver.cpp:464] Iteration 1308, lr = 1e-05
I1118 03:10:13.621675 22484 solver.cpp:209] Iteration 1309, loss = 0.227146
I1118 03:10:13.621716 22484 solver.cpp:464] Iteration 1309, lr = 1e-05
I1118 03:10:15.856434 22484 solver.cpp:209] Iteration 1310, loss = 0.307492
I1118 03:10:15.856462 22484 solver.cpp:464] Iteration 1310, lr = 1e-05
I1118 03:10:18.085613 22484 solver.cpp:209] Iteration 1311, loss = 0.232094
I1118 03:10:18.085652 22484 solver.cpp:464] Iteration 1311, lr = 1e-05
I1118 03:10:20.309162 22484 solver.cpp:209] Iteration 1312, loss = 0.384542
I1118 03:10:20.309202 22484 solver.cpp:464] Iteration 1312, lr = 1e-05
I1118 03:10:22.534495 22484 solver.cpp:209] Iteration 1313, loss = 0.243967
I1118 03:10:22.534536 22484 solver.cpp:464] Iteration 1313, lr = 1e-05
I1118 03:10:24.764860 22484 solver.cpp:209] Iteration 1314, loss = 0.366956
I1118 03:10:24.764902 22484 solver.cpp:464] Iteration 1314, lr = 1e-05
I1118 03:10:27.002799 22484 solver.cpp:209] Iteration 1315, loss = 0.108679
I1118 03:10:27.002887 22484 solver.cpp:464] Iteration 1315, lr = 1e-05
I1118 03:10:29.241125 22484 solver.cpp:209] Iteration 1316, loss = 0.341328
I1118 03:10:29.241155 22484 solver.cpp:464] Iteration 1316, lr = 1e-05
I1118 03:10:31.473831 22484 solver.cpp:209] Iteration 1317, loss = 0.281378
I1118 03:10:31.473872 22484 solver.cpp:464] Iteration 1317, lr = 1e-05
I1118 03:10:33.702649 22484 solver.cpp:209] Iteration 1318, loss = 0.340425
I1118 03:10:33.702679 22484 solver.cpp:464] Iteration 1318, lr = 1e-05
I1118 03:10:35.928818 22484 solver.cpp:209] Iteration 1319, loss = 0.46228
I1118 03:10:35.928860 22484 solver.cpp:464] Iteration 1319, lr = 1e-05
I1118 03:10:38.159132 22484 solver.cpp:209] Iteration 1320, loss = 0.4008
I1118 03:10:38.159159 22484 solver.cpp:464] Iteration 1320, lr = 1e-05
I1118 03:10:40.381814 22484 solver.cpp:209] Iteration 1321, loss = 0.0808242
I1118 03:10:40.381855 22484 solver.cpp:464] Iteration 1321, lr = 1e-05
I1118 03:10:42.620237 22484 solver.cpp:209] Iteration 1322, loss = 0.105389
I1118 03:10:42.620280 22484 solver.cpp:464] Iteration 1322, lr = 1e-05
I1118 03:10:44.859007 22484 solver.cpp:209] Iteration 1323, loss = 0.271516
I1118 03:10:44.859047 22484 solver.cpp:464] Iteration 1323, lr = 1e-05
I1118 03:10:47.088795 22484 solver.cpp:209] Iteration 1324, loss = 0.342286
I1118 03:10:47.088824 22484 solver.cpp:464] Iteration 1324, lr = 1e-05
I1118 03:10:49.311699 22484 solver.cpp:209] Iteration 1325, loss = 0.170408
I1118 03:10:49.311728 22484 solver.cpp:464] Iteration 1325, lr = 1e-05
I1118 03:10:51.529966 22484 solver.cpp:209] Iteration 1326, loss = 0.6035
I1118 03:10:51.529994 22484 solver.cpp:464] Iteration 1326, lr = 1e-05
I1118 03:10:53.756191 22484 solver.cpp:209] Iteration 1327, loss = 0.102676
I1118 03:10:53.756232 22484 solver.cpp:464] Iteration 1327, lr = 1e-05
I1118 03:10:55.988723 22484 solver.cpp:209] Iteration 1328, loss = 0.299157
I1118 03:10:55.988764 22484 solver.cpp:464] Iteration 1328, lr = 1e-05
I1118 03:10:58.230638 22484 solver.cpp:209] Iteration 1329, loss = 0.228293
I1118 03:10:58.230711 22484 solver.cpp:464] Iteration 1329, lr = 1e-05
I1118 03:11:00.459329 22484 solver.cpp:209] Iteration 1330, loss = 0.241393
I1118 03:11:00.459357 22484 solver.cpp:464] Iteration 1330, lr = 1e-05
I1118 03:11:02.680176 22484 solver.cpp:209] Iteration 1331, loss = 0.408115
I1118 03:11:02.680217 22484 solver.cpp:464] Iteration 1331, lr = 1e-05
I1118 03:11:04.867656 22484 solver.cpp:209] Iteration 1332, loss = 0.172445
I1118 03:11:04.867686 22484 solver.cpp:464] Iteration 1332, lr = 1e-05
I1118 03:11:07.056004 22484 solver.cpp:209] Iteration 1333, loss = 0.265555
I1118 03:11:07.056032 22484 solver.cpp:464] Iteration 1333, lr = 1e-05
I1118 03:11:09.252447 22484 solver.cpp:209] Iteration 1334, loss = 0.207782
I1118 03:11:09.252476 22484 solver.cpp:464] Iteration 1334, lr = 1e-05
I1118 03:11:11.448354 22484 solver.cpp:209] Iteration 1335, loss = 0.157583
I1118 03:11:11.448395 22484 solver.cpp:464] Iteration 1335, lr = 1e-05
I1118 03:11:13.641700 22484 solver.cpp:209] Iteration 1336, loss = 0.303096
I1118 03:11:13.641728 22484 solver.cpp:464] Iteration 1336, lr = 1e-05
I1118 03:11:15.828804 22484 solver.cpp:209] Iteration 1337, loss = 0.206213
I1118 03:11:15.828832 22484 solver.cpp:464] Iteration 1337, lr = 1e-05
I1118 03:11:18.019996 22484 solver.cpp:209] Iteration 1338, loss = 0.22505
I1118 03:11:18.020025 22484 solver.cpp:464] Iteration 1338, lr = 1e-05
I1118 03:11:20.213512 22484 solver.cpp:209] Iteration 1339, loss = 0.303604
I1118 03:11:20.213541 22484 solver.cpp:464] Iteration 1339, lr = 1e-05
I1118 03:11:22.413331 22484 solver.cpp:209] Iteration 1340, loss = 0.18572
I1118 03:11:22.413360 22484 solver.cpp:464] Iteration 1340, lr = 1e-05
I1118 03:11:24.605552 22484 solver.cpp:209] Iteration 1341, loss = 0.355921
I1118 03:11:24.605581 22484 solver.cpp:464] Iteration 1341, lr = 1e-05
I1118 03:11:26.799208 22484 solver.cpp:209] Iteration 1342, loss = 0.220094
I1118 03:11:26.799249 22484 solver.cpp:464] Iteration 1342, lr = 1e-05
I1118 03:11:28.989562 22484 solver.cpp:209] Iteration 1343, loss = 0.293777
I1118 03:11:28.989630 22484 solver.cpp:464] Iteration 1343, lr = 1e-05
I1118 03:11:31.180277 22484 solver.cpp:209] Iteration 1344, loss = 0.38542
I1118 03:11:31.180302 22484 solver.cpp:464] Iteration 1344, lr = 1e-05
I1118 03:11:33.373190 22484 solver.cpp:209] Iteration 1345, loss = 0.137035
I1118 03:11:33.373219 22484 solver.cpp:464] Iteration 1345, lr = 1e-05
I1118 03:11:35.571100 22484 solver.cpp:209] Iteration 1346, loss = 0.198432
I1118 03:11:35.571130 22484 solver.cpp:464] Iteration 1346, lr = 1e-05
I1118 03:11:37.779939 22484 solver.cpp:209] Iteration 1347, loss = 0.15004
I1118 03:11:37.779966 22484 solver.cpp:464] Iteration 1347, lr = 1e-05
I1118 03:11:40.010570 22484 solver.cpp:209] Iteration 1348, loss = 0.286367
I1118 03:11:40.010622 22484 solver.cpp:464] Iteration 1348, lr = 1e-05
I1118 03:11:42.239241 22484 solver.cpp:209] Iteration 1349, loss = 0.332826
I1118 03:11:42.239281 22484 solver.cpp:464] Iteration 1349, lr = 1e-05
I1118 03:11:42.239863 22484 solver.cpp:264] Iteration 1350, Testing net (#0)
I1118 03:11:56.120559 22484 solver.cpp:305] Test loss: 0.253499
I1118 03:11:56.120600 22484 solver.cpp:318] mean_score = test_score[0] { = 512} / test_score[1] { = 517 }
I1118 03:11:56.120609 22484 solver.cpp:319]            = 0.990329
I1118 03:11:56.120612 22484 solver.cpp:328]     Test net output #0: accuracy = 0.990329
I1118 03:11:56.120617 22484 solver.cpp:318] mean_score = test_score[2] { = 5} / test_score[3] { = 59 }
I1118 03:11:56.120622 22484 solver.cpp:319]            = 0.0847458
I1118 03:11:56.120626 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0847458
I1118 03:11:56.120630 22484 solver.cpp:332]     Test net output #2: accuracy = 0.897569
I1118 03:11:56.120635 22484 solver.cpp:334]     Test net output #3: accuracy = 0.537537
I1118 03:11:56.765782 22484 solver.cpp:209] Iteration 1350, loss = 0.327706
I1118 03:11:56.765810 22484 solver.cpp:464] Iteration 1350, lr = 1e-05
I1118 03:11:59.004544 22484 solver.cpp:209] Iteration 1351, loss = 0.132059
I1118 03:11:59.004639 22484 solver.cpp:464] Iteration 1351, lr = 1e-05
I1118 03:12:01.245364 22484 solver.cpp:209] Iteration 1352, loss = 0.36483
I1118 03:12:01.245393 22484 solver.cpp:464] Iteration 1352, lr = 1e-05
I1118 03:12:03.492332 22484 solver.cpp:209] Iteration 1353, loss = 0.0886774
I1118 03:12:03.492362 22484 solver.cpp:464] Iteration 1353, lr = 1e-05
I1118 03:12:05.733345 22484 solver.cpp:209] Iteration 1354, loss = 0.0960389
I1118 03:12:05.733374 22484 solver.cpp:464] Iteration 1354, lr = 1e-05
I1118 03:12:07.979024 22484 solver.cpp:209] Iteration 1355, loss = 0.19972
I1118 03:12:07.979053 22484 solver.cpp:464] Iteration 1355, lr = 1e-05
I1118 03:12:10.217895 22484 solver.cpp:209] Iteration 1356, loss = 0.204771
I1118 03:12:10.217933 22484 solver.cpp:464] Iteration 1356, lr = 1e-05
I1118 03:12:12.464046 22484 solver.cpp:209] Iteration 1357, loss = 0.28389
I1118 03:12:12.464076 22484 solver.cpp:464] Iteration 1357, lr = 1e-05
I1118 03:12:14.705879 22484 solver.cpp:209] Iteration 1358, loss = 0.214751
I1118 03:12:14.705907 22484 solver.cpp:464] Iteration 1358, lr = 1e-05
I1118 03:12:16.941792 22484 solver.cpp:209] Iteration 1359, loss = 0.266963
I1118 03:12:16.941833 22484 solver.cpp:464] Iteration 1359, lr = 1e-05
I1118 03:12:19.158850 22484 solver.cpp:209] Iteration 1360, loss = 0.148273
I1118 03:12:19.158879 22484 solver.cpp:464] Iteration 1360, lr = 1e-05
I1118 03:12:21.352494 22484 solver.cpp:209] Iteration 1361, loss = 0.139182
I1118 03:12:21.352535 22484 solver.cpp:464] Iteration 1361, lr = 1e-05
I1118 03:12:23.553411 22484 solver.cpp:209] Iteration 1362, loss = 0.236445
I1118 03:12:23.553452 22484 solver.cpp:464] Iteration 1362, lr = 1e-05
I1118 03:12:25.756458 22484 solver.cpp:209] Iteration 1363, loss = 0.238943
I1118 03:12:25.756485 22484 solver.cpp:464] Iteration 1363, lr = 1e-05
I1118 03:12:27.946744 22484 solver.cpp:209] Iteration 1364, loss = 0.241858
I1118 03:12:27.946774 22484 solver.cpp:464] Iteration 1364, lr = 1e-05
I1118 03:12:30.138381 22484 solver.cpp:209] Iteration 1365, loss = 0.0529957
I1118 03:12:30.138475 22484 solver.cpp:464] Iteration 1365, lr = 1e-05
I1118 03:12:32.324084 22484 solver.cpp:209] Iteration 1366, loss = 0.642932
I1118 03:12:32.324126 22484 solver.cpp:464] Iteration 1366, lr = 1e-05
I1118 03:12:34.522652 22484 solver.cpp:209] Iteration 1367, loss = 0.272365
I1118 03:12:34.522681 22484 solver.cpp:464] Iteration 1367, lr = 1e-05
I1118 03:12:36.723523 22484 solver.cpp:209] Iteration 1368, loss = 0.188566
I1118 03:12:36.723551 22484 solver.cpp:464] Iteration 1368, lr = 1e-05
I1118 03:12:38.919114 22484 solver.cpp:209] Iteration 1369, loss = 0.316075
I1118 03:12:38.919143 22484 solver.cpp:464] Iteration 1369, lr = 1e-05
I1118 03:12:41.114389 22484 solver.cpp:209] Iteration 1370, loss = 0.233736
I1118 03:12:41.114414 22484 solver.cpp:464] Iteration 1370, lr = 1e-05
I1118 03:12:43.297346 22484 solver.cpp:209] Iteration 1371, loss = 0.311744
I1118 03:12:43.297374 22484 solver.cpp:464] Iteration 1371, lr = 1e-05
I1118 03:12:45.486842 22484 solver.cpp:209] Iteration 1372, loss = 0.154186
I1118 03:12:45.486871 22484 solver.cpp:464] Iteration 1372, lr = 1e-05
I1118 03:12:47.690953 22484 solver.cpp:209] Iteration 1373, loss = 0.267997
I1118 03:12:47.690979 22484 solver.cpp:464] Iteration 1373, lr = 1e-05
I1118 03:12:49.883152 22484 solver.cpp:209] Iteration 1374, loss = 0.379068
I1118 03:12:49.883180 22484 solver.cpp:464] Iteration 1374, lr = 1e-05
I1118 03:12:52.077673 22484 solver.cpp:209] Iteration 1375, loss = 0.532508
I1118 03:12:52.077714 22484 solver.cpp:464] Iteration 1375, lr = 1e-05
I1118 03:12:54.268610 22484 solver.cpp:209] Iteration 1376, loss = 0.210082
I1118 03:12:54.268651 22484 solver.cpp:464] Iteration 1376, lr = 1e-05
I1118 03:12:56.458833 22484 solver.cpp:209] Iteration 1377, loss = 0.12628
I1118 03:12:56.458863 22484 solver.cpp:464] Iteration 1377, lr = 1e-05
I1118 03:12:58.674485 22484 solver.cpp:209] Iteration 1378, loss = 0.248187
I1118 03:12:58.674526 22484 solver.cpp:464] Iteration 1378, lr = 1e-05
I1118 03:13:00.902215 22484 solver.cpp:209] Iteration 1379, loss = 0.391926
I1118 03:13:00.902303 22484 solver.cpp:464] Iteration 1379, lr = 1e-05
I1118 03:13:03.135097 22484 solver.cpp:209] Iteration 1380, loss = 0.273117
I1118 03:13:03.135139 22484 solver.cpp:464] Iteration 1380, lr = 1e-05
I1118 03:13:05.360973 22484 solver.cpp:209] Iteration 1381, loss = 0.147452
I1118 03:13:05.361014 22484 solver.cpp:464] Iteration 1381, lr = 1e-05
I1118 03:13:07.585850 22484 solver.cpp:209] Iteration 1382, loss = 0.199043
I1118 03:13:07.585878 22484 solver.cpp:464] Iteration 1382, lr = 1e-05
I1118 03:13:09.812605 22484 solver.cpp:209] Iteration 1383, loss = 0.140839
I1118 03:13:09.812634 22484 solver.cpp:464] Iteration 1383, lr = 1e-05
I1118 03:13:12.033277 22484 solver.cpp:209] Iteration 1384, loss = 0.301702
I1118 03:13:12.033304 22484 solver.cpp:464] Iteration 1384, lr = 1e-05
I1118 03:13:14.262213 22484 solver.cpp:209] Iteration 1385, loss = 0.462919
I1118 03:13:14.262253 22484 solver.cpp:464] Iteration 1385, lr = 1e-05
I1118 03:13:16.491375 22484 solver.cpp:209] Iteration 1386, loss = 0.321588
I1118 03:13:16.491403 22484 solver.cpp:464] Iteration 1386, lr = 1e-05
I1118 03:13:18.724112 22484 solver.cpp:209] Iteration 1387, loss = 0.297461
I1118 03:13:18.724141 22484 solver.cpp:464] Iteration 1387, lr = 1e-05
I1118 03:13:20.957784 22484 solver.cpp:209] Iteration 1388, loss = 0.225959
I1118 03:13:20.957810 22484 solver.cpp:464] Iteration 1388, lr = 1e-05
I1118 03:13:23.181867 22484 solver.cpp:209] Iteration 1389, loss = 0.391858
I1118 03:13:23.181896 22484 solver.cpp:464] Iteration 1389, lr = 1e-05
I1118 03:13:25.405835 22484 solver.cpp:209] Iteration 1390, loss = 0.124193
I1118 03:13:25.405876 22484 solver.cpp:464] Iteration 1390, lr = 1e-05
I1118 03:13:27.628597 22484 solver.cpp:209] Iteration 1391, loss = 0.187627
I1118 03:13:27.628623 22484 solver.cpp:464] Iteration 1391, lr = 1e-05
I1118 03:13:29.860621 22484 solver.cpp:209] Iteration 1392, loss = 0.384175
I1118 03:13:29.860651 22484 solver.cpp:464] Iteration 1392, lr = 1e-05
I1118 03:13:32.097044 22484 solver.cpp:209] Iteration 1393, loss = 0.166161
I1118 03:13:32.097137 22484 solver.cpp:464] Iteration 1393, lr = 1e-05
I1118 03:13:34.322492 22484 solver.cpp:209] Iteration 1394, loss = 0.302055
I1118 03:13:34.322535 22484 solver.cpp:464] Iteration 1394, lr = 1e-05
I1118 03:13:36.551161 22484 solver.cpp:209] Iteration 1395, loss = 0.213591
I1118 03:13:36.551187 22484 solver.cpp:464] Iteration 1395, lr = 1e-05
I1118 03:13:38.772872 22484 solver.cpp:209] Iteration 1396, loss = 0.254418
I1118 03:13:38.772902 22484 solver.cpp:464] Iteration 1396, lr = 1e-05
I1118 03:13:41.000097 22484 solver.cpp:209] Iteration 1397, loss = 0.421728
I1118 03:13:41.000123 22484 solver.cpp:464] Iteration 1397, lr = 1e-05
I1118 03:13:43.194520 22484 solver.cpp:209] Iteration 1398, loss = 0.432191
I1118 03:13:43.194561 22484 solver.cpp:464] Iteration 1398, lr = 1e-05
I1118 03:13:45.387820 22484 solver.cpp:209] Iteration 1399, loss = 0.299941
I1118 03:13:45.387861 22484 solver.cpp:464] Iteration 1399, lr = 1e-05
I1118 03:13:45.388463 22484 solver.cpp:264] Iteration 1400, Testing net (#0)
I1118 03:13:59.116864 22484 solver.cpp:305] Test loss: 0.252368
I1118 03:13:59.116894 22484 solver.cpp:318] mean_score = test_score[0] { = 512} / test_score[1] { = 517 }
I1118 03:13:59.116914 22484 solver.cpp:319]            = 0.990329
I1118 03:13:59.116917 22484 solver.cpp:328]     Test net output #0: accuracy = 0.990329
I1118 03:13:59.116921 22484 solver.cpp:318] mean_score = test_score[2] { = 4} / test_score[3] { = 59 }
I1118 03:13:59.116926 22484 solver.cpp:319]            = 0.0677966
I1118 03:13:59.116930 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0677966
I1118 03:13:59.116935 22484 solver.cpp:332]     Test net output #2: accuracy = 0.895833
I1118 03:13:59.116938 22484 solver.cpp:334]     Test net output #3: accuracy = 0.529063
I1118 03:13:59.766340 22484 solver.cpp:209] Iteration 1400, loss = 0.609909
I1118 03:13:59.766368 22484 solver.cpp:464] Iteration 1400, lr = 1e-05
I1118 03:14:01.994035 22484 solver.cpp:209] Iteration 1401, loss = 0.172017
I1118 03:14:01.994063 22484 solver.cpp:464] Iteration 1401, lr = 1e-05
I1118 03:14:04.233311 22484 solver.cpp:209] Iteration 1402, loss = 0.42134
I1118 03:14:04.233372 22484 solver.cpp:464] Iteration 1402, lr = 1e-05
I1118 03:14:06.477468 22484 solver.cpp:209] Iteration 1403, loss = 0.239341
I1118 03:14:06.477498 22484 solver.cpp:464] Iteration 1403, lr = 1e-05
I1118 03:14:08.716603 22484 solver.cpp:209] Iteration 1404, loss = 0.389367
I1118 03:14:08.716644 22484 solver.cpp:464] Iteration 1404, lr = 1e-05
I1118 03:14:10.958837 22484 solver.cpp:209] Iteration 1405, loss = 0.307087
I1118 03:14:10.958863 22484 solver.cpp:464] Iteration 1405, lr = 1e-05
I1118 03:14:13.198729 22484 solver.cpp:209] Iteration 1406, loss = 0.322471
I1118 03:14:13.198758 22484 solver.cpp:464] Iteration 1406, lr = 1e-05
I1118 03:14:15.521239 22484 solver.cpp:209] Iteration 1407, loss = 0.354639
I1118 03:14:15.521270 22484 solver.cpp:464] Iteration 1407, lr = 1e-05
I1118 03:14:17.757658 22484 solver.cpp:209] Iteration 1408, loss = 0.462033
I1118 03:14:17.757699 22484 solver.cpp:464] Iteration 1408, lr = 1e-05
I1118 03:14:19.985373 22484 solver.cpp:209] Iteration 1409, loss = 0.295407
I1118 03:14:19.985414 22484 solver.cpp:464] Iteration 1409, lr = 1e-05
I1118 03:14:22.185385 22484 solver.cpp:209] Iteration 1410, loss = 0.409028
I1118 03:14:22.185427 22484 solver.cpp:464] Iteration 1410, lr = 1e-05
I1118 03:14:24.374718 22484 solver.cpp:209] Iteration 1411, loss = 0.176706
I1118 03:14:24.374747 22484 solver.cpp:464] Iteration 1411, lr = 1e-05
I1118 03:14:26.559819 22484 solver.cpp:209] Iteration 1412, loss = 0.285425
I1118 03:14:26.559859 22484 solver.cpp:464] Iteration 1412, lr = 1e-05
I1118 03:14:28.760452 22484 solver.cpp:209] Iteration 1413, loss = 0.135134
I1118 03:14:28.760493 22484 solver.cpp:464] Iteration 1413, lr = 1e-05
I1118 03:14:30.957296 22484 solver.cpp:209] Iteration 1414, loss = 0.233053
I1118 03:14:30.957321 22484 solver.cpp:464] Iteration 1414, lr = 1e-05
I1118 03:14:33.160009 22484 solver.cpp:209] Iteration 1415, loss = 0.320246
I1118 03:14:33.160050 22484 solver.cpp:464] Iteration 1415, lr = 1e-05
I1118 03:14:35.354173 22484 solver.cpp:209] Iteration 1416, loss = 0.214365
I1118 03:14:35.354235 22484 solver.cpp:464] Iteration 1416, lr = 1e-05
I1118 03:14:37.548336 22484 solver.cpp:209] Iteration 1417, loss = 0.241528
I1118 03:14:37.548364 22484 solver.cpp:464] Iteration 1417, lr = 1e-05
I1118 03:14:39.735383 22484 solver.cpp:209] Iteration 1418, loss = 0.11902
I1118 03:14:39.735424 22484 solver.cpp:464] Iteration 1418, lr = 1e-05
I1118 03:14:41.937777 22484 solver.cpp:209] Iteration 1419, loss = 0.352932
I1118 03:14:41.937816 22484 solver.cpp:464] Iteration 1419, lr = 1e-05
I1118 03:14:44.137215 22484 solver.cpp:209] Iteration 1420, loss = 0.323663
I1118 03:14:44.137256 22484 solver.cpp:464] Iteration 1420, lr = 1e-05
I1118 03:14:46.338793 22484 solver.cpp:209] Iteration 1421, loss = 0.421217
I1118 03:14:46.338824 22484 solver.cpp:464] Iteration 1421, lr = 1e-05
I1118 03:14:48.531553 22484 solver.cpp:209] Iteration 1422, loss = 0.531504
I1118 03:14:48.531582 22484 solver.cpp:464] Iteration 1422, lr = 1e-05
I1118 03:14:50.716369 22484 solver.cpp:209] Iteration 1423, loss = 0.276358
I1118 03:14:50.716398 22484 solver.cpp:464] Iteration 1423, lr = 1e-05
I1118 03:14:52.923213 22484 solver.cpp:209] Iteration 1424, loss = 0.108527
I1118 03:14:52.923254 22484 solver.cpp:464] Iteration 1424, lr = 1e-05
I1118 03:14:55.159984 22484 solver.cpp:209] Iteration 1425, loss = 0.128472
I1118 03:14:55.160013 22484 solver.cpp:464] Iteration 1425, lr = 1e-05
I1118 03:14:57.396961 22484 solver.cpp:209] Iteration 1426, loss = 0.213502
I1118 03:14:57.396989 22484 solver.cpp:464] Iteration 1426, lr = 1e-05
I1118 03:14:59.625327 22484 solver.cpp:209] Iteration 1427, loss = 0.340642
I1118 03:14:59.625370 22484 solver.cpp:464] Iteration 1427, lr = 1e-05
I1118 03:15:01.845593 22484 solver.cpp:209] Iteration 1428, loss = 0.204288
I1118 03:15:01.845621 22484 solver.cpp:464] Iteration 1428, lr = 1e-05
I1118 03:15:04.066185 22484 solver.cpp:209] Iteration 1429, loss = 0.522995
I1118 03:15:04.066226 22484 solver.cpp:464] Iteration 1429, lr = 1e-05
I1118 03:15:06.290663 22484 solver.cpp:209] Iteration 1430, loss = 0.0509432
I1118 03:15:06.290722 22484 solver.cpp:464] Iteration 1430, lr = 1e-05
I1118 03:15:08.520496 22484 solver.cpp:209] Iteration 1431, loss = 0.263372
I1118 03:15:08.520525 22484 solver.cpp:464] Iteration 1431, lr = 1e-05
I1118 03:15:10.759424 22484 solver.cpp:209] Iteration 1432, loss = 0.27135
I1118 03:15:10.759450 22484 solver.cpp:464] Iteration 1432, lr = 1e-05
I1118 03:15:12.991770 22484 solver.cpp:209] Iteration 1433, loss = 0.292476
I1118 03:15:12.991799 22484 solver.cpp:464] Iteration 1433, lr = 1e-05
I1118 03:15:15.222931 22484 solver.cpp:209] Iteration 1434, loss = 0.370816
I1118 03:15:15.222960 22484 solver.cpp:464] Iteration 1434, lr = 1e-05
I1118 03:15:17.447597 22484 solver.cpp:209] Iteration 1435, loss = 0.149292
I1118 03:15:17.447624 22484 solver.cpp:464] Iteration 1435, lr = 1e-05
I1118 03:15:19.676559 22484 solver.cpp:209] Iteration 1436, loss = 0.373122
I1118 03:15:19.676599 22484 solver.cpp:464] Iteration 1436, lr = 1e-05
I1118 03:15:21.899741 22484 solver.cpp:209] Iteration 1437, loss = 0.149561
I1118 03:15:21.899781 22484 solver.cpp:464] Iteration 1437, lr = 1e-05
I1118 03:15:24.125443 22484 solver.cpp:209] Iteration 1438, loss = 0.159357
I1118 03:15:24.125471 22484 solver.cpp:464] Iteration 1438, lr = 1e-05
I1118 03:15:26.358333 22484 solver.cpp:209] Iteration 1439, loss = 0.1516
I1118 03:15:26.358373 22484 solver.cpp:464] Iteration 1439, lr = 1e-05
I1118 03:15:28.592367 22484 solver.cpp:209] Iteration 1440, loss = 0.262227
I1118 03:15:28.592394 22484 solver.cpp:464] Iteration 1440, lr = 1e-05
I1118 03:15:30.827746 22484 solver.cpp:209] Iteration 1441, loss = 0.121763
I1118 03:15:30.827772 22484 solver.cpp:464] Iteration 1441, lr = 1e-05
I1118 03:15:33.049355 22484 solver.cpp:209] Iteration 1442, loss = 0.308167
I1118 03:15:33.049382 22484 solver.cpp:464] Iteration 1442, lr = 1e-05
I1118 03:15:35.275665 22484 solver.cpp:209] Iteration 1443, loss = 0.301598
I1118 03:15:35.275694 22484 solver.cpp:464] Iteration 1443, lr = 1e-05
I1118 03:15:37.496016 22484 solver.cpp:209] Iteration 1444, loss = 0.176798
I1118 03:15:37.496109 22484 solver.cpp:464] Iteration 1444, lr = 1e-05
I1118 03:15:39.722290 22484 solver.cpp:209] Iteration 1445, loss = 0.162663
I1118 03:15:39.722316 22484 solver.cpp:464] Iteration 1445, lr = 1e-05
I1118 03:15:41.955795 22484 solver.cpp:209] Iteration 1446, loss = 0.290504
I1118 03:15:41.955822 22484 solver.cpp:464] Iteration 1446, lr = 1e-05
I1118 03:15:44.123730 22484 solver.cpp:209] Iteration 1447, loss = 0.289083
I1118 03:15:44.123757 22484 solver.cpp:464] Iteration 1447, lr = 1e-05
I1118 03:15:46.284442 22484 solver.cpp:209] Iteration 1448, loss = 0.137302
I1118 03:15:46.284471 22484 solver.cpp:464] Iteration 1448, lr = 1e-05
I1118 03:15:48.444763 22484 solver.cpp:209] Iteration 1449, loss = 0.265022
I1118 03:15:48.444804 22484 solver.cpp:464] Iteration 1449, lr = 1e-05
I1118 03:15:48.445380 22484 solver.cpp:264] Iteration 1450, Testing net (#0)
I1118 03:16:02.176044 22484 solver.cpp:305] Test loss: 0.24421
I1118 03:16:02.176071 22484 solver.cpp:318] mean_score = test_score[0] { = 512} / test_score[1] { = 517 }
I1118 03:16:02.176091 22484 solver.cpp:319]            = 0.990329
I1118 03:16:02.176095 22484 solver.cpp:328]     Test net output #0: accuracy = 0.990329
I1118 03:16:02.176100 22484 solver.cpp:318] mean_score = test_score[2] { = 6} / test_score[3] { = 59 }
I1118 03:16:02.176105 22484 solver.cpp:319]            = 0.101695
I1118 03:16:02.176108 22484 solver.cpp:328]     Test net output #1: accuracy = 0.101695
I1118 03:16:02.176113 22484 solver.cpp:332]     Test net output #2: accuracy = 0.899306
I1118 03:16:02.176117 22484 solver.cpp:334]     Test net output #3: accuracy = 0.546012
I1118 03:16:02.826764 22484 solver.cpp:209] Iteration 1450, loss = 0.214213
I1118 03:16:02.826793 22484 solver.cpp:464] Iteration 1450, lr = 1e-05
I1118 03:16:05.069790 22484 solver.cpp:209] Iteration 1451, loss = 0.342064
I1118 03:16:05.069818 22484 solver.cpp:464] Iteration 1451, lr = 1e-05
I1118 03:16:07.330288 22484 solver.cpp:209] Iteration 1452, loss = 0.280046
I1118 03:16:07.330330 22484 solver.cpp:464] Iteration 1452, lr = 1e-05
I1118 03:16:09.576967 22484 solver.cpp:209] Iteration 1453, loss = 0.305577
I1118 03:16:09.577064 22484 solver.cpp:464] Iteration 1453, lr = 1e-05
I1118 03:16:11.823573 22484 solver.cpp:209] Iteration 1454, loss = 0.26736
I1118 03:16:11.823601 22484 solver.cpp:464] Iteration 1454, lr = 1e-05
I1118 03:16:14.072690 22484 solver.cpp:209] Iteration 1455, loss = 0.249017
I1118 03:16:14.072717 22484 solver.cpp:464] Iteration 1455, lr = 1e-05
I1118 03:16:16.319944 22484 solver.cpp:209] Iteration 1456, loss = 0.0576347
I1118 03:16:16.319985 22484 solver.cpp:464] Iteration 1456, lr = 1e-05
I1118 03:16:18.561532 22484 solver.cpp:209] Iteration 1457, loss = 0.117719
I1118 03:16:18.561573 22484 solver.cpp:464] Iteration 1457, lr = 1e-05
I1118 03:16:20.786880 22484 solver.cpp:209] Iteration 1458, loss = 0.20479
I1118 03:16:20.786906 22484 solver.cpp:464] Iteration 1458, lr = 1e-05
I1118 03:16:22.975657 22484 solver.cpp:209] Iteration 1459, loss = 0.250326
I1118 03:16:22.975685 22484 solver.cpp:464] Iteration 1459, lr = 1e-05
I1118 03:16:25.179790 22484 solver.cpp:209] Iteration 1460, loss = 0.315295
I1118 03:16:25.179817 22484 solver.cpp:464] Iteration 1460, lr = 1e-05
I1118 03:16:27.382822 22484 solver.cpp:209] Iteration 1461, loss = 0.294084
I1118 03:16:27.382849 22484 solver.cpp:464] Iteration 1461, lr = 1e-05
I1118 03:16:29.580905 22484 solver.cpp:209] Iteration 1462, loss = 0.194196
I1118 03:16:29.580945 22484 solver.cpp:464] Iteration 1462, lr = 1e-05
I1118 03:16:31.768244 22484 solver.cpp:209] Iteration 1463, loss = 0.163883
I1118 03:16:31.768271 22484 solver.cpp:464] Iteration 1463, lr = 1e-05
I1118 03:16:33.959054 22484 solver.cpp:209] Iteration 1464, loss = 0.119142
I1118 03:16:33.959084 22484 solver.cpp:464] Iteration 1464, lr = 1e-05
I1118 03:16:36.153214 22484 solver.cpp:209] Iteration 1465, loss = 0.346011
I1118 03:16:36.153255 22484 solver.cpp:464] Iteration 1465, lr = 1e-05
I1118 03:16:38.355797 22484 solver.cpp:209] Iteration 1466, loss = 0.260858
I1118 03:16:38.355823 22484 solver.cpp:464] Iteration 1466, lr = 1e-05
I1118 03:16:40.550930 22484 solver.cpp:209] Iteration 1467, loss = 0.254377
I1118 03:16:40.551028 22484 solver.cpp:464] Iteration 1467, lr = 1e-05
I1118 03:16:42.736856 22484 solver.cpp:209] Iteration 1468, loss = 0.150603
I1118 03:16:42.736886 22484 solver.cpp:464] Iteration 1468, lr = 1e-05
I1118 03:16:44.927804 22484 solver.cpp:209] Iteration 1469, loss = 0.493175
I1118 03:16:44.927832 22484 solver.cpp:464] Iteration 1469, lr = 1e-05
I1118 03:16:47.122313 22484 solver.cpp:209] Iteration 1470, loss = 0.229207
I1118 03:16:47.122340 22484 solver.cpp:464] Iteration 1470, lr = 1e-05
I1118 03:16:49.323349 22484 solver.cpp:209] Iteration 1471, loss = 0.149091
I1118 03:16:49.323390 22484 solver.cpp:464] Iteration 1471, lr = 1e-05
I1118 03:16:51.520102 22484 solver.cpp:209] Iteration 1472, loss = 0.250626
I1118 03:16:51.520131 22484 solver.cpp:464] Iteration 1472, lr = 1e-05
I1118 03:16:53.721535 22484 solver.cpp:209] Iteration 1473, loss = 0.381432
I1118 03:16:53.721562 22484 solver.cpp:464] Iteration 1473, lr = 1e-05
I1118 03:16:55.909575 22484 solver.cpp:209] Iteration 1474, loss = 0.156619
I1118 03:16:55.909615 22484 solver.cpp:464] Iteration 1474, lr = 1e-05
I1118 03:16:58.105828 22484 solver.cpp:209] Iteration 1475, loss = 0.24328
I1118 03:16:58.105856 22484 solver.cpp:464] Iteration 1475, lr = 1e-05
I1118 03:17:00.298238 22484 solver.cpp:209] Iteration 1476, loss = 0.240217
I1118 03:17:00.298264 22484 solver.cpp:464] Iteration 1476, lr = 1e-05
I1118 03:17:02.498934 22484 solver.cpp:209] Iteration 1477, loss = 0.359753
I1118 03:17:02.498973 22484 solver.cpp:464] Iteration 1477, lr = 1e-05
I1118 03:17:04.695104 22484 solver.cpp:209] Iteration 1478, loss = 0.382036
I1118 03:17:04.695133 22484 solver.cpp:464] Iteration 1478, lr = 1e-05
I1118 03:17:06.890514 22484 solver.cpp:209] Iteration 1479, loss = 0.168817
I1118 03:17:06.890552 22484 solver.cpp:464] Iteration 1479, lr = 1e-05
I1118 03:17:09.083004 22484 solver.cpp:209] Iteration 1480, loss = 0.266503
I1118 03:17:09.083046 22484 solver.cpp:464] Iteration 1480, lr = 1e-05
I1118 03:17:11.272076 22484 solver.cpp:209] Iteration 1481, loss = 0.246743
I1118 03:17:11.272147 22484 solver.cpp:464] Iteration 1481, lr = 1e-05
I1118 03:17:13.463904 22484 solver.cpp:209] Iteration 1482, loss = 0.294989
I1118 03:17:13.463945 22484 solver.cpp:464] Iteration 1482, lr = 1e-05
I1118 03:17:15.663429 22484 solver.cpp:209] Iteration 1483, loss = 0.2417
I1118 03:17:15.663457 22484 solver.cpp:464] Iteration 1483, lr = 1e-05
I1118 03:17:17.867132 22484 solver.cpp:209] Iteration 1484, loss = 0.179125
I1118 03:17:17.867161 22484 solver.cpp:464] Iteration 1484, lr = 1e-05
I1118 03:17:20.094653 22484 solver.cpp:209] Iteration 1485, loss = 0.158128
I1118 03:17:20.094682 22484 solver.cpp:464] Iteration 1485, lr = 1e-05
I1118 03:17:22.315132 22484 solver.cpp:209] Iteration 1486, loss = 0.178211
I1118 03:17:22.315161 22484 solver.cpp:464] Iteration 1486, lr = 1e-05
I1118 03:17:24.541934 22484 solver.cpp:209] Iteration 1487, loss = 0.360569
I1118 03:17:24.541975 22484 solver.cpp:464] Iteration 1487, lr = 1e-05
I1118 03:17:26.766491 22484 solver.cpp:209] Iteration 1488, loss = 0.474949
I1118 03:17:26.766533 22484 solver.cpp:464] Iteration 1488, lr = 1e-05
I1118 03:17:28.997436 22484 solver.cpp:209] Iteration 1489, loss = 0.283761
I1118 03:17:28.997464 22484 solver.cpp:464] Iteration 1489, lr = 1e-05
I1118 03:17:31.229341 22484 solver.cpp:209] Iteration 1490, loss = 0.239847
I1118 03:17:31.229369 22484 solver.cpp:464] Iteration 1490, lr = 1e-05
I1118 03:17:33.459588 22484 solver.cpp:209] Iteration 1491, loss = 0.20168
I1118 03:17:33.459617 22484 solver.cpp:464] Iteration 1491, lr = 1e-05
I1118 03:17:35.683899 22484 solver.cpp:209] Iteration 1492, loss = 0.296716
I1118 03:17:35.683928 22484 solver.cpp:464] Iteration 1492, lr = 1e-05
I1118 03:17:37.901989 22484 solver.cpp:209] Iteration 1493, loss = 0.128552
I1118 03:17:37.902019 22484 solver.cpp:464] Iteration 1493, lr = 1e-05
I1118 03:17:40.125360 22484 solver.cpp:209] Iteration 1494, loss = 0.248857
I1118 03:17:40.125402 22484 solver.cpp:464] Iteration 1494, lr = 1e-05
I1118 03:17:42.352848 22484 solver.cpp:209] Iteration 1495, loss = 0.268458
I1118 03:17:42.352948 22484 solver.cpp:464] Iteration 1495, lr = 1e-05
I1118 03:17:44.605154 22484 solver.cpp:209] Iteration 1496, loss = 0.305703
I1118 03:17:44.605183 22484 solver.cpp:464] Iteration 1496, lr = 1e-05
I1118 03:17:46.838413 22484 solver.cpp:209] Iteration 1497, loss = 0.341878
I1118 03:17:46.838443 22484 solver.cpp:464] Iteration 1497, lr = 1e-05
I1118 03:17:49.091064 22484 solver.cpp:209] Iteration 1498, loss = 0.13897
I1118 03:17:49.091094 22484 solver.cpp:464] Iteration 1498, lr = 1e-05
I1118 03:17:51.284951 22484 solver.cpp:209] Iteration 1499, loss = 0.282808
I1118 03:17:51.284992 22484 solver.cpp:464] Iteration 1499, lr = 1e-05
I1118 03:17:51.285570 22484 solver.cpp:264] Iteration 1500, Testing net (#0)
I1118 03:18:04.981192 22484 solver.cpp:305] Test loss: 0.24433
I1118 03:18:04.981235 22484 solver.cpp:318] mean_score = test_score[0] { = 512} / test_score[1] { = 517 }
I1118 03:18:04.981241 22484 solver.cpp:319]            = 0.990329
I1118 03:18:04.981246 22484 solver.cpp:328]     Test net output #0: accuracy = 0.990329
I1118 03:18:04.981251 22484 solver.cpp:318] mean_score = test_score[2] { = 5} / test_score[3] { = 59 }
I1118 03:18:04.981256 22484 solver.cpp:319]            = 0.0847458
I1118 03:18:04.981259 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0847458
I1118 03:18:04.981263 22484 solver.cpp:332]     Test net output #2: accuracy = 0.897569
I1118 03:18:04.981267 22484 solver.cpp:334]     Test net output #3: accuracy = 0.537537
I1118 03:18:05.619346 22484 solver.cpp:209] Iteration 1500, loss = 0.23413
I1118 03:18:05.619375 22484 solver.cpp:464] Iteration 1500, lr = 1e-05
I1118 03:18:07.824280 22484 solver.cpp:209] Iteration 1501, loss = 0.44325
I1118 03:18:07.824308 22484 solver.cpp:464] Iteration 1501, lr = 1e-05
I1118 03:18:10.052235 22484 solver.cpp:209] Iteration 1502, loss = 0.57542
I1118 03:18:10.052276 22484 solver.cpp:464] Iteration 1502, lr = 1e-05
I1118 03:18:12.284703 22484 solver.cpp:209] Iteration 1503, loss = 0.483319
I1118 03:18:12.284732 22484 solver.cpp:464] Iteration 1503, lr = 1e-05
I1118 03:18:14.518051 22484 solver.cpp:209] Iteration 1504, loss = 0.311818
I1118 03:18:14.518111 22484 solver.cpp:464] Iteration 1504, lr = 1e-05
I1118 03:18:16.747031 22484 solver.cpp:209] Iteration 1505, loss = 0.224917
I1118 03:18:16.747061 22484 solver.cpp:464] Iteration 1505, lr = 1e-05
I1118 03:18:18.971812 22484 solver.cpp:209] Iteration 1506, loss = 0.261118
I1118 03:18:18.971842 22484 solver.cpp:464] Iteration 1506, lr = 1e-05
I1118 03:18:21.198673 22484 solver.cpp:209] Iteration 1507, loss = 0.237289
I1118 03:18:21.198704 22484 solver.cpp:464] Iteration 1507, lr = 1e-05
I1118 03:18:23.424456 22484 solver.cpp:209] Iteration 1508, loss = 0.390554
I1118 03:18:23.424497 22484 solver.cpp:464] Iteration 1508, lr = 1e-05
I1118 03:18:25.651213 22484 solver.cpp:209] Iteration 1509, loss = 0.3428
I1118 03:18:25.651240 22484 solver.cpp:464] Iteration 1509, lr = 1e-05
I1118 03:18:27.882076 22484 solver.cpp:209] Iteration 1510, loss = 0.465915
I1118 03:18:27.882105 22484 solver.cpp:464] Iteration 1510, lr = 1e-05
I1118 03:18:30.116808 22484 solver.cpp:209] Iteration 1511, loss = 0.422109
I1118 03:18:30.116848 22484 solver.cpp:464] Iteration 1511, lr = 1e-05
I1118 03:18:32.355836 22484 solver.cpp:209] Iteration 1512, loss = 0.250439
I1118 03:18:32.355865 22484 solver.cpp:464] Iteration 1512, lr = 1e-05
I1118 03:18:34.589751 22484 solver.cpp:209] Iteration 1513, loss = 0.515906
I1118 03:18:34.589781 22484 solver.cpp:464] Iteration 1513, lr = 1e-05
I1118 03:18:36.813974 22484 solver.cpp:209] Iteration 1514, loss = 0.0946
I1118 03:18:36.814015 22484 solver.cpp:464] Iteration 1514, lr = 1e-05
I1118 03:18:39.035074 22484 solver.cpp:209] Iteration 1515, loss = 0.289331
I1118 03:18:39.035104 22484 solver.cpp:464] Iteration 1515, lr = 1e-05
I1118 03:18:41.264164 22484 solver.cpp:209] Iteration 1516, loss = 0.142651
I1118 03:18:41.264192 22484 solver.cpp:464] Iteration 1516, lr = 1e-05
I1118 03:18:43.499816 22484 solver.cpp:209] Iteration 1517, loss = 0.252865
I1118 03:18:43.499845 22484 solver.cpp:464] Iteration 1517, lr = 1e-05
I1118 03:18:45.736011 22484 solver.cpp:209] Iteration 1518, loss = 0.415473
I1118 03:18:45.736074 22484 solver.cpp:464] Iteration 1518, lr = 1e-05
I1118 03:18:47.968142 22484 solver.cpp:209] Iteration 1519, loss = 0.179119
I1118 03:18:47.968183 22484 solver.cpp:464] Iteration 1519, lr = 1e-05
I1118 03:18:50.190708 22484 solver.cpp:209] Iteration 1520, loss = 0.241838
I1118 03:18:50.190737 22484 solver.cpp:464] Iteration 1520, lr = 1e-05
I1118 03:18:52.414312 22484 solver.cpp:209] Iteration 1521, loss = 0.195607
I1118 03:18:52.414342 22484 solver.cpp:464] Iteration 1521, lr = 1e-05
I1118 03:18:54.634671 22484 solver.cpp:209] Iteration 1522, loss = 0.307543
I1118 03:18:54.634699 22484 solver.cpp:464] Iteration 1522, lr = 1e-05
I1118 03:18:56.829547 22484 solver.cpp:209] Iteration 1523, loss = 0.395515
I1118 03:18:56.829576 22484 solver.cpp:464] Iteration 1523, lr = 1e-05
I1118 03:18:59.029305 22484 solver.cpp:209] Iteration 1524, loss = 0.526455
I1118 03:18:59.029332 22484 solver.cpp:464] Iteration 1524, lr = 1e-05
I1118 03:19:01.221634 22484 solver.cpp:209] Iteration 1525, loss = 0.441479
I1118 03:19:01.221662 22484 solver.cpp:464] Iteration 1525, lr = 1e-05
I1118 03:19:03.413136 22484 solver.cpp:209] Iteration 1526, loss = 0.267123
I1118 03:19:03.413177 22484 solver.cpp:464] Iteration 1526, lr = 1e-05
I1118 03:19:05.607144 22484 solver.cpp:209] Iteration 1527, loss = 0.113888
I1118 03:19:05.607187 22484 solver.cpp:464] Iteration 1527, lr = 1e-05
I1118 03:19:07.803005 22484 solver.cpp:209] Iteration 1528, loss = 0.105053
I1118 03:19:07.803045 22484 solver.cpp:464] Iteration 1528, lr = 1e-05
I1118 03:19:09.997581 22484 solver.cpp:209] Iteration 1529, loss = 0.265954
I1118 03:19:09.997623 22484 solver.cpp:464] Iteration 1529, lr = 1e-05
I1118 03:19:12.190966 22484 solver.cpp:209] Iteration 1530, loss = 0.291626
I1118 03:19:12.190995 22484 solver.cpp:464] Iteration 1530, lr = 1e-05
I1118 03:19:14.386440 22484 solver.cpp:209] Iteration 1531, loss = 0.153093
I1118 03:19:14.386482 22484 solver.cpp:464] Iteration 1531, lr = 1e-05
I1118 03:19:16.579128 22484 solver.cpp:209] Iteration 1532, loss = 0.461987
I1118 03:19:16.579221 22484 solver.cpp:464] Iteration 1532, lr = 1e-05
I1118 03:19:18.772338 22484 solver.cpp:209] Iteration 1533, loss = 0.23651
I1118 03:19:18.772367 22484 solver.cpp:464] Iteration 1533, lr = 1e-05
I1118 03:19:20.963608 22484 solver.cpp:209] Iteration 1534, loss = 0.226849
I1118 03:19:20.963634 22484 solver.cpp:464] Iteration 1534, lr = 1e-05
I1118 03:19:23.180371 22484 solver.cpp:209] Iteration 1535, loss = 0.190139
I1118 03:19:23.180399 22484 solver.cpp:464] Iteration 1535, lr = 1e-05
I1118 03:19:25.377599 22484 solver.cpp:209] Iteration 1536, loss = 0.274339
I1118 03:19:25.377626 22484 solver.cpp:464] Iteration 1536, lr = 1e-05
I1118 03:19:27.579150 22484 solver.cpp:209] Iteration 1537, loss = 0.508113
I1118 03:19:27.579177 22484 solver.cpp:464] Iteration 1537, lr = 1e-05
I1118 03:19:29.771226 22484 solver.cpp:209] Iteration 1538, loss = 0.0772479
I1118 03:19:29.771255 22484 solver.cpp:464] Iteration 1538, lr = 1e-05
I1118 03:19:31.958796 22484 solver.cpp:209] Iteration 1539, loss = 0.427436
I1118 03:19:31.958827 22484 solver.cpp:464] Iteration 1539, lr = 1e-05
I1118 03:19:34.158823 22484 solver.cpp:209] Iteration 1540, loss = 0.107051
I1118 03:19:34.158854 22484 solver.cpp:464] Iteration 1540, lr = 1e-05
I1118 03:19:36.362747 22484 solver.cpp:209] Iteration 1541, loss = 0.138221
I1118 03:19:36.362776 22484 solver.cpp:464] Iteration 1541, lr = 1e-05
I1118 03:19:38.554692 22484 solver.cpp:209] Iteration 1542, loss = 0.301408
I1118 03:19:38.554723 22484 solver.cpp:464] Iteration 1542, lr = 1e-05
I1118 03:19:40.753255 22484 solver.cpp:209] Iteration 1543, loss = 0.242904
I1118 03:19:40.753296 22484 solver.cpp:464] Iteration 1543, lr = 1e-05
I1118 03:19:42.964882 22484 solver.cpp:209] Iteration 1544, loss = 0.10605
I1118 03:19:42.964911 22484 solver.cpp:464] Iteration 1544, lr = 1e-05
I1118 03:19:45.190930 22484 solver.cpp:209] Iteration 1545, loss = 0.487724
I1118 03:19:45.190959 22484 solver.cpp:464] Iteration 1545, lr = 1e-05
I1118 03:19:47.417264 22484 solver.cpp:209] Iteration 1546, loss = 0.224679
I1118 03:19:47.417367 22484 solver.cpp:464] Iteration 1546, lr = 1e-05
I1118 03:19:49.641968 22484 solver.cpp:209] Iteration 1547, loss = 0.149067
I1118 03:19:49.641996 22484 solver.cpp:464] Iteration 1547, lr = 1e-05
I1118 03:19:51.873317 22484 solver.cpp:209] Iteration 1548, loss = 0.25812
I1118 03:19:51.873356 22484 solver.cpp:464] Iteration 1548, lr = 1e-05
I1118 03:19:54.100613 22484 solver.cpp:209] Iteration 1549, loss = 0.372028
I1118 03:19:54.100642 22484 solver.cpp:464] Iteration 1549, lr = 1e-05
I1118 03:19:54.101275 22484 solver.cpp:264] Iteration 1550, Testing net (#0)
I1118 03:20:07.994495 22484 solver.cpp:305] Test loss: 0.250483
I1118 03:20:07.994524 22484 solver.cpp:318] mean_score = test_score[0] { = 512} / test_score[1] { = 517 }
I1118 03:20:07.994544 22484 solver.cpp:319]            = 0.990329
I1118 03:20:07.994549 22484 solver.cpp:328]     Test net output #0: accuracy = 0.990329
I1118 03:20:07.994552 22484 solver.cpp:318] mean_score = test_score[2] { = 9} / test_score[3] { = 59 }
I1118 03:20:07.994557 22484 solver.cpp:319]            = 0.152542
I1118 03:20:07.994561 22484 solver.cpp:328]     Test net output #1: accuracy = 0.152542
I1118 03:20:07.994565 22484 solver.cpp:332]     Test net output #2: accuracy = 0.904514
I1118 03:20:07.994570 22484 solver.cpp:334]     Test net output #3: accuracy = 0.571436
I1118 03:20:08.641294 22484 solver.cpp:209] Iteration 1550, loss = 0.248442
I1118 03:20:08.641335 22484 solver.cpp:464] Iteration 1550, lr = 1e-05
I1118 03:20:10.879215 22484 solver.cpp:209] Iteration 1551, loss = 0.147257
I1118 03:20:10.879241 22484 solver.cpp:464] Iteration 1551, lr = 1e-05
I1118 03:20:13.100611 22484 solver.cpp:209] Iteration 1552, loss = 0.219409
I1118 03:20:13.100641 22484 solver.cpp:464] Iteration 1552, lr = 1e-05
I1118 03:20:15.323420 22484 solver.cpp:209] Iteration 1553, loss = 0.204268
I1118 03:20:15.323448 22484 solver.cpp:464] Iteration 1553, lr = 1e-05
I1118 03:20:17.545794 22484 solver.cpp:209] Iteration 1554, loss = 0.323481
I1118 03:20:17.545853 22484 solver.cpp:464] Iteration 1554, lr = 1e-05
I1118 03:20:19.770462 22484 solver.cpp:209] Iteration 1555, loss = 0.548562
I1118 03:20:19.770490 22484 solver.cpp:464] Iteration 1555, lr = 1e-05
I1118 03:20:22.012158 22484 solver.cpp:209] Iteration 1556, loss = 0.191212
I1118 03:20:22.012199 22484 solver.cpp:464] Iteration 1556, lr = 1e-05
I1118 03:20:24.237952 22484 solver.cpp:209] Iteration 1557, loss = 0.345924
I1118 03:20:24.237982 22484 solver.cpp:464] Iteration 1557, lr = 1e-05
I1118 03:20:26.468040 22484 solver.cpp:209] Iteration 1558, loss = 0.1945
I1118 03:20:26.468082 22484 solver.cpp:464] Iteration 1558, lr = 1e-05
I1118 03:20:28.694419 22484 solver.cpp:209] Iteration 1559, loss = 0.103186
I1118 03:20:28.694447 22484 solver.cpp:464] Iteration 1559, lr = 1e-05
I1118 03:20:30.917024 22484 solver.cpp:209] Iteration 1560, loss = 0.152141
I1118 03:20:30.917050 22484 solver.cpp:464] Iteration 1560, lr = 1e-05
I1118 03:20:33.140719 22484 solver.cpp:209] Iteration 1561, loss = 0.223303
I1118 03:20:33.140748 22484 solver.cpp:464] Iteration 1561, lr = 1e-05
I1118 03:20:35.363716 22484 solver.cpp:209] Iteration 1562, loss = 0.0850876
I1118 03:20:35.363756 22484 solver.cpp:464] Iteration 1562, lr = 1e-05
I1118 03:20:37.592844 22484 solver.cpp:209] Iteration 1563, loss = 0.273841
I1118 03:20:37.592874 22484 solver.cpp:464] Iteration 1563, lr = 1e-05
I1118 03:20:39.817831 22484 solver.cpp:209] Iteration 1564, loss = 0.247031
I1118 03:20:39.817873 22484 solver.cpp:464] Iteration 1564, lr = 1e-05
I1118 03:20:42.058195 22484 solver.cpp:209] Iteration 1565, loss = 0.246384
I1118 03:20:42.058236 22484 solver.cpp:464] Iteration 1565, lr = 1e-05
I1118 03:20:44.278506 22484 solver.cpp:209] Iteration 1566, loss = 0.11502
I1118 03:20:44.278548 22484 solver.cpp:464] Iteration 1566, lr = 1e-05
I1118 03:20:46.496794 22484 solver.cpp:209] Iteration 1567, loss = 0.115021
I1118 03:20:46.496822 22484 solver.cpp:464] Iteration 1567, lr = 1e-05
I1118 03:20:48.689565 22484 solver.cpp:209] Iteration 1568, loss = 0.347023
I1118 03:20:48.689667 22484 solver.cpp:464] Iteration 1568, lr = 1e-05
I1118 03:20:50.883844 22484 solver.cpp:209] Iteration 1569, loss = 0.367769
I1118 03:20:50.883870 22484 solver.cpp:464] Iteration 1569, lr = 1e-05
I1118 03:20:53.079979 22484 solver.cpp:209] Iteration 1570, loss = 0.0920771
I1118 03:20:53.080020 22484 solver.cpp:464] Iteration 1570, lr = 1e-05
I1118 03:20:55.283673 22484 solver.cpp:209] Iteration 1571, loss = 0.244663
I1118 03:20:55.283701 22484 solver.cpp:464] Iteration 1571, lr = 1e-05
I1118 03:20:57.478309 22484 solver.cpp:209] Iteration 1572, loss = 0.476765
I1118 03:20:57.478338 22484 solver.cpp:464] Iteration 1572, lr = 1e-05
I1118 03:20:59.673580 22484 solver.cpp:209] Iteration 1573, loss = 0.252719
I1118 03:20:59.673609 22484 solver.cpp:464] Iteration 1573, lr = 1e-05
I1118 03:21:01.866561 22484 solver.cpp:209] Iteration 1574, loss = 0.324758
I1118 03:21:01.866611 22484 solver.cpp:464] Iteration 1574, lr = 1e-05
I1118 03:21:04.068084 22484 solver.cpp:209] Iteration 1575, loss = 0.229672
I1118 03:21:04.068125 22484 solver.cpp:464] Iteration 1575, lr = 1e-05
I1118 03:21:06.262393 22484 solver.cpp:209] Iteration 1576, loss = 0.228405
I1118 03:21:06.262423 22484 solver.cpp:464] Iteration 1576, lr = 1e-05
I1118 03:21:08.458369 22484 solver.cpp:209] Iteration 1577, loss = 0.171176
I1118 03:21:08.458397 22484 solver.cpp:464] Iteration 1577, lr = 1e-05
I1118 03:21:10.656163 22484 solver.cpp:209] Iteration 1578, loss = 0.183255
I1118 03:21:10.656190 22484 solver.cpp:464] Iteration 1578, lr = 1e-05
I1118 03:21:12.852638 22484 solver.cpp:209] Iteration 1579, loss = 0.299093
I1118 03:21:12.852679 22484 solver.cpp:464] Iteration 1579, lr = 1e-05
I1118 03:21:15.042675 22484 solver.cpp:209] Iteration 1580, loss = 0.327324
I1118 03:21:15.042704 22484 solver.cpp:464] Iteration 1580, lr = 1e-05
I1118 03:21:17.237203 22484 solver.cpp:209] Iteration 1581, loss = 0.414727
I1118 03:21:17.237243 22484 solver.cpp:464] Iteration 1581, lr = 1e-05
I1118 03:21:19.433976 22484 solver.cpp:209] Iteration 1582, loss = 0.136219
I1118 03:21:19.434073 22484 solver.cpp:464] Iteration 1582, lr = 1e-05
I1118 03:21:21.629789 22484 solver.cpp:209] Iteration 1583, loss = 0.230173
I1118 03:21:21.629817 22484 solver.cpp:464] Iteration 1583, lr = 1e-05
I1118 03:21:23.824815 22484 solver.cpp:209] Iteration 1584, loss = 0.260933
I1118 03:21:23.824844 22484 solver.cpp:464] Iteration 1584, lr = 1e-05
I1118 03:21:26.047016 22484 solver.cpp:209] Iteration 1585, loss = 0.1286
I1118 03:21:26.047045 22484 solver.cpp:464] Iteration 1585, lr = 1e-05
I1118 03:21:28.285712 22484 solver.cpp:209] Iteration 1586, loss = 0.185309
I1118 03:21:28.285753 22484 solver.cpp:464] Iteration 1586, lr = 1e-05
I1118 03:21:30.522971 22484 solver.cpp:209] Iteration 1587, loss = 0.243048
I1118 03:21:30.523000 22484 solver.cpp:464] Iteration 1587, lr = 1e-05
I1118 03:21:32.760030 22484 solver.cpp:209] Iteration 1588, loss = 0.104639
I1118 03:21:32.760071 22484 solver.cpp:464] Iteration 1588, lr = 1e-05
I1118 03:21:34.993911 22484 solver.cpp:209] Iteration 1589, loss = 0.36077
I1118 03:21:34.993942 22484 solver.cpp:464] Iteration 1589, lr = 1e-05
I1118 03:21:37.215402 22484 solver.cpp:209] Iteration 1590, loss = 0.19065
I1118 03:21:37.215430 22484 solver.cpp:464] Iteration 1590, lr = 1e-05
I1118 03:21:39.434368 22484 solver.cpp:209] Iteration 1591, loss = 0.466749
I1118 03:21:39.434397 22484 solver.cpp:464] Iteration 1591, lr = 1e-05
I1118 03:21:41.657317 22484 solver.cpp:209] Iteration 1592, loss = 0.341871
I1118 03:21:41.657348 22484 solver.cpp:464] Iteration 1592, lr = 1e-05
I1118 03:21:43.887394 22484 solver.cpp:209] Iteration 1593, loss = 0.170889
I1118 03:21:43.887423 22484 solver.cpp:464] Iteration 1593, lr = 1e-05
I1118 03:21:46.121819 22484 solver.cpp:209] Iteration 1594, loss = 0.21712
I1118 03:21:46.121848 22484 solver.cpp:464] Iteration 1594, lr = 1e-05
I1118 03:21:48.351198 22484 solver.cpp:209] Iteration 1595, loss = 0.324836
I1118 03:21:48.351239 22484 solver.cpp:464] Iteration 1595, lr = 1e-05
I1118 03:21:50.581789 22484 solver.cpp:209] Iteration 1596, loss = 0.140731
I1118 03:21:50.581888 22484 solver.cpp:464] Iteration 1596, lr = 1e-05
I1118 03:21:52.803812 22484 solver.cpp:209] Iteration 1597, loss = 0.527509
I1118 03:21:52.803841 22484 solver.cpp:464] Iteration 1597, lr = 1e-05
I1118 03:21:55.031618 22484 solver.cpp:209] Iteration 1598, loss = 0.139764
I1118 03:21:55.031647 22484 solver.cpp:464] Iteration 1598, lr = 1e-05
I1118 03:21:57.255453 22484 solver.cpp:209] Iteration 1599, loss = 0.155909
I1118 03:21:57.255494 22484 solver.cpp:464] Iteration 1599, lr = 1e-05
I1118 03:21:57.256078 22484 solver.cpp:264] Iteration 1600, Testing net (#0)
I1118 03:22:11.132001 22484 solver.cpp:305] Test loss: 0.240146
I1118 03:22:11.132043 22484 solver.cpp:318] mean_score = test_score[0] { = 513} / test_score[1] { = 517 }
I1118 03:22:11.132050 22484 solver.cpp:319]            = 0.992263
I1118 03:22:11.132056 22484 solver.cpp:328]     Test net output #0: accuracy = 0.992263
I1118 03:22:11.132061 22484 solver.cpp:318] mean_score = test_score[2] { = 9} / test_score[3] { = 59 }
I1118 03:22:11.132066 22484 solver.cpp:319]            = 0.152542
I1118 03:22:11.132068 22484 solver.cpp:328]     Test net output #1: accuracy = 0.152542
I1118 03:22:11.132072 22484 solver.cpp:332]     Test net output #2: accuracy = 0.90625
I1118 03:22:11.132076 22484 solver.cpp:334]     Test net output #3: accuracy = 0.572403
I1118 03:22:11.777757 22484 solver.cpp:209] Iteration 1600, loss = 0.339719
I1118 03:22:11.777786 22484 solver.cpp:464] Iteration 1600, lr = 1e-05
I1118 03:22:14.006232 22484 solver.cpp:209] Iteration 1601, loss = 0.159232
I1118 03:22:14.006273 22484 solver.cpp:464] Iteration 1601, lr = 1e-05
I1118 03:22:16.237038 22484 solver.cpp:209] Iteration 1602, loss = 0.262451
I1118 03:22:16.237077 22484 solver.cpp:464] Iteration 1602, lr = 1e-05
I1118 03:22:18.474831 22484 solver.cpp:209] Iteration 1603, loss = 0.285025
I1118 03:22:18.474860 22484 solver.cpp:464] Iteration 1603, lr = 1e-05
I1118 03:22:20.705536 22484 solver.cpp:209] Iteration 1604, loss = 0.601746
I1118 03:22:20.705610 22484 solver.cpp:464] Iteration 1604, lr = 1e-05
I1118 03:22:22.931545 22484 solver.cpp:209] Iteration 1605, loss = 0.368446
I1118 03:22:22.931574 22484 solver.cpp:464] Iteration 1605, lr = 1e-05
I1118 03:22:25.158421 22484 solver.cpp:209] Iteration 1606, loss = 0.426817
I1118 03:22:25.158449 22484 solver.cpp:464] Iteration 1606, lr = 1e-05
I1118 03:22:27.382185 22484 solver.cpp:209] Iteration 1607, loss = 0.257701
I1118 03:22:27.382213 22484 solver.cpp:464] Iteration 1607, lr = 1e-05
I1118 03:22:29.611497 22484 solver.cpp:209] Iteration 1608, loss = 0.308065
I1118 03:22:29.611526 22484 solver.cpp:464] Iteration 1608, lr = 1e-05
I1118 03:22:31.843296 22484 solver.cpp:209] Iteration 1609, loss = 0.459897
I1118 03:22:31.843325 22484 solver.cpp:464] Iteration 1609, lr = 1e-05
I1118 03:22:34.076901 22484 solver.cpp:209] Iteration 1610, loss = 0.290671
I1118 03:22:34.076930 22484 solver.cpp:464] Iteration 1610, lr = 1e-05
I1118 03:22:36.307860 22484 solver.cpp:209] Iteration 1611, loss = 0.281978
I1118 03:22:36.307900 22484 solver.cpp:464] Iteration 1611, lr = 1e-05
I1118 03:22:38.497951 22484 solver.cpp:209] Iteration 1612, loss = 0.434163
I1118 03:22:38.497992 22484 solver.cpp:464] Iteration 1612, lr = 1e-05
I1118 03:22:40.688689 22484 solver.cpp:209] Iteration 1613, loss = 0.277827
I1118 03:22:40.688717 22484 solver.cpp:464] Iteration 1613, lr = 1e-05
I1118 03:22:42.875058 22484 solver.cpp:209] Iteration 1614, loss = 0.355451
I1118 03:22:42.875087 22484 solver.cpp:464] Iteration 1614, lr = 1e-05
I1118 03:22:45.071444 22484 solver.cpp:209] Iteration 1615, loss = 0.296577
I1118 03:22:45.071472 22484 solver.cpp:464] Iteration 1615, lr = 1e-05
I1118 03:22:47.268877 22484 solver.cpp:209] Iteration 1616, loss = 0.267927
I1118 03:22:47.268918 22484 solver.cpp:464] Iteration 1616, lr = 1e-05
I1118 03:22:49.464123 22484 solver.cpp:209] Iteration 1617, loss = 0.104161
I1118 03:22:49.464153 22484 solver.cpp:464] Iteration 1617, lr = 1e-05
I1118 03:22:51.653990 22484 solver.cpp:209] Iteration 1618, loss = 0.357898
I1118 03:22:51.654062 22484 solver.cpp:464] Iteration 1618, lr = 1e-05
I1118 03:22:53.839360 22484 solver.cpp:209] Iteration 1619, loss = 0.0995215
I1118 03:22:53.839390 22484 solver.cpp:464] Iteration 1619, lr = 1e-05
I1118 03:22:56.034420 22484 solver.cpp:209] Iteration 1620, loss = 0.492959
I1118 03:22:56.034461 22484 solver.cpp:464] Iteration 1620, lr = 1e-05
I1118 03:22:58.243540 22484 solver.cpp:209] Iteration 1621, loss = 0.142168
I1118 03:22:58.243579 22484 solver.cpp:464] Iteration 1621, lr = 1e-05
I1118 03:23:00.434376 22484 solver.cpp:209] Iteration 1622, loss = 0.302962
I1118 03:23:00.434404 22484 solver.cpp:464] Iteration 1622, lr = 1e-05
I1118 03:23:02.634491 22484 solver.cpp:209] Iteration 1623, loss = 0.168831
I1118 03:23:02.634531 22484 solver.cpp:464] Iteration 1623, lr = 1e-05
I1118 03:23:04.823288 22484 solver.cpp:209] Iteration 1624, loss = 0.341263
I1118 03:23:04.823318 22484 solver.cpp:464] Iteration 1624, lr = 1e-05
I1118 03:23:07.018474 22484 solver.cpp:209] Iteration 1625, loss = 0.211638
I1118 03:23:07.018502 22484 solver.cpp:464] Iteration 1625, lr = 1e-05
I1118 03:23:09.210857 22484 solver.cpp:209] Iteration 1626, loss = 0.37848
I1118 03:23:09.210898 22484 solver.cpp:464] Iteration 1626, lr = 1e-05
I1118 03:23:11.403674 22484 solver.cpp:209] Iteration 1627, loss = 0.480745
I1118 03:23:11.403703 22484 solver.cpp:464] Iteration 1627, lr = 1e-05
I1118 03:23:13.617496 22484 solver.cpp:209] Iteration 1628, loss = 0.317562
I1118 03:23:13.617537 22484 solver.cpp:464] Iteration 1628, lr = 1e-05
I1118 03:23:15.843049 22484 solver.cpp:209] Iteration 1629, loss = 0.177437
I1118 03:23:15.843091 22484 solver.cpp:464] Iteration 1629, lr = 1e-05
I1118 03:23:18.070883 22484 solver.cpp:209] Iteration 1630, loss = 0.168331
I1118 03:23:18.070910 22484 solver.cpp:464] Iteration 1630, lr = 1e-05
I1118 03:23:20.293649 22484 solver.cpp:209] Iteration 1631, loss = 0.0875213
I1118 03:23:20.293689 22484 solver.cpp:464] Iteration 1631, lr = 1e-05
I1118 03:23:22.513170 22484 solver.cpp:209] Iteration 1632, loss = 0.355812
I1118 03:23:22.513268 22484 solver.cpp:464] Iteration 1632, lr = 1e-05
I1118 03:23:24.745122 22484 solver.cpp:209] Iteration 1633, loss = 0.219909
I1118 03:23:24.745151 22484 solver.cpp:464] Iteration 1633, lr = 1e-05
I1118 03:23:26.975721 22484 solver.cpp:209] Iteration 1634, loss = 0.220675
I1118 03:23:26.975750 22484 solver.cpp:464] Iteration 1634, lr = 1e-05
I1118 03:23:29.211380 22484 solver.cpp:209] Iteration 1635, loss = 0.350443
I1118 03:23:29.211421 22484 solver.cpp:464] Iteration 1635, lr = 1e-05
I1118 03:23:31.439653 22484 solver.cpp:209] Iteration 1636, loss = 0.252077
I1118 03:23:31.439682 22484 solver.cpp:464] Iteration 1636, lr = 1e-05
I1118 03:23:33.659987 22484 solver.cpp:209] Iteration 1637, loss = 0.219229
I1118 03:23:33.660017 22484 solver.cpp:464] Iteration 1637, lr = 1e-05
I1118 03:23:35.886111 22484 solver.cpp:209] Iteration 1638, loss = 0.354447
I1118 03:23:35.886140 22484 solver.cpp:464] Iteration 1638, lr = 1e-05
I1118 03:23:38.112483 22484 solver.cpp:209] Iteration 1639, loss = 0.302214
I1118 03:23:38.112521 22484 solver.cpp:464] Iteration 1639, lr = 1e-05
I1118 03:23:40.346161 22484 solver.cpp:209] Iteration 1640, loss = 0.363719
I1118 03:23:40.346202 22484 solver.cpp:464] Iteration 1640, lr = 1e-05
I1118 03:23:42.578848 22484 solver.cpp:209] Iteration 1641, loss = 0.18233
I1118 03:23:42.578878 22484 solver.cpp:464] Iteration 1641, lr = 1e-05
I1118 03:23:44.775187 22484 solver.cpp:209] Iteration 1642, loss = 0.354038
I1118 03:23:44.775214 22484 solver.cpp:464] Iteration 1642, lr = 1e-05
I1118 03:23:46.964287 22484 solver.cpp:209] Iteration 1643, loss = 0.140576
I1118 03:23:46.964328 22484 solver.cpp:464] Iteration 1643, lr = 1e-05
I1118 03:23:49.156605 22484 solver.cpp:209] Iteration 1644, loss = 0.234176
I1118 03:23:49.156633 22484 solver.cpp:464] Iteration 1644, lr = 1e-05
I1118 03:23:51.348389 22484 solver.cpp:209] Iteration 1645, loss = 0.204928
I1118 03:23:51.348428 22484 solver.cpp:464] Iteration 1645, lr = 1e-05
I1118 03:23:53.551082 22484 solver.cpp:209] Iteration 1646, loss = 0.316861
I1118 03:23:53.551183 22484 solver.cpp:464] Iteration 1646, lr = 1e-05
I1118 03:23:55.742249 22484 solver.cpp:209] Iteration 1647, loss = 0.119248
I1118 03:23:55.742276 22484 solver.cpp:464] Iteration 1647, lr = 1e-05
I1118 03:23:57.941781 22484 solver.cpp:209] Iteration 1648, loss = 0.444713
I1118 03:23:57.941812 22484 solver.cpp:464] Iteration 1648, lr = 1e-05
I1118 03:24:00.130614 22484 solver.cpp:209] Iteration 1649, loss = 0.24945
I1118 03:24:00.130643 22484 solver.cpp:464] Iteration 1649, lr = 1e-05
I1118 03:24:00.131219 22484 solver.cpp:264] Iteration 1650, Testing net (#0)
I1118 03:24:13.889401 22484 solver.cpp:305] Test loss: 0.242001
I1118 03:24:13.889430 22484 solver.cpp:318] mean_score = test_score[0] { = 513} / test_score[1] { = 517 }
I1118 03:24:13.889437 22484 solver.cpp:319]            = 0.992263
I1118 03:24:13.889441 22484 solver.cpp:328]     Test net output #0: accuracy = 0.992263
I1118 03:24:13.889446 22484 solver.cpp:318] mean_score = test_score[2] { = 6} / test_score[3] { = 59 }
I1118 03:24:13.889451 22484 solver.cpp:319]            = 0.101695
I1118 03:24:13.889456 22484 solver.cpp:328]     Test net output #1: accuracy = 0.101695
I1118 03:24:13.889459 22484 solver.cpp:332]     Test net output #2: accuracy = 0.901042
I1118 03:24:13.889463 22484 solver.cpp:334]     Test net output #3: accuracy = 0.546979
I1118 03:24:14.541052 22484 solver.cpp:209] Iteration 1650, loss = 0.105206
I1118 03:24:14.541092 22484 solver.cpp:464] Iteration 1650, lr = 1e-05
I1118 03:24:16.780546 22484 solver.cpp:209] Iteration 1651, loss = 0.323938
I1118 03:24:16.780575 22484 solver.cpp:464] Iteration 1651, lr = 1e-05
I1118 03:24:19.020728 22484 solver.cpp:209] Iteration 1652, loss = 0.315345
I1118 03:24:19.020756 22484 solver.cpp:464] Iteration 1652, lr = 1e-05
I1118 03:24:21.272449 22484 solver.cpp:209] Iteration 1653, loss = 0.177462
I1118 03:24:21.272491 22484 solver.cpp:464] Iteration 1653, lr = 1e-05
I1118 03:24:23.514394 22484 solver.cpp:209] Iteration 1654, loss = 0.131056
I1118 03:24:23.514436 22484 solver.cpp:464] Iteration 1654, lr = 1e-05
I1118 03:24:25.760592 22484 solver.cpp:209] Iteration 1655, loss = 0.244855
I1118 03:24:25.760690 22484 solver.cpp:464] Iteration 1655, lr = 1e-05
I1118 03:24:28.007041 22484 solver.cpp:209] Iteration 1656, loss = 0.258052
I1118 03:24:28.007081 22484 solver.cpp:464] Iteration 1656, lr = 1e-05
I1118 03:24:30.247652 22484 solver.cpp:209] Iteration 1657, loss = 0.326993
I1118 03:24:30.247680 22484 solver.cpp:464] Iteration 1657, lr = 1e-05
I1118 03:24:32.484572 22484 solver.cpp:209] Iteration 1658, loss = 0.512772
I1118 03:24:32.484601 22484 solver.cpp:464] Iteration 1658, lr = 1e-05
I1118 03:24:34.708065 22484 solver.cpp:209] Iteration 1659, loss = 0.155825
I1118 03:24:34.708096 22484 solver.cpp:464] Iteration 1659, lr = 1e-05
I1118 03:24:36.935500 22484 solver.cpp:209] Iteration 1660, loss = 0.405384
I1118 03:24:36.935528 22484 solver.cpp:464] Iteration 1660, lr = 1e-05
I1118 03:24:39.160181 22484 solver.cpp:209] Iteration 1661, loss = 0.128316
I1118 03:24:39.160220 22484 solver.cpp:464] Iteration 1661, lr = 1e-05
I1118 03:24:41.358070 22484 solver.cpp:209] Iteration 1662, loss = 0.120984
I1118 03:24:41.358100 22484 solver.cpp:464] Iteration 1662, lr = 1e-05
I1118 03:24:43.496345 22484 solver.cpp:209] Iteration 1663, loss = 0.168181
I1118 03:24:43.496386 22484 solver.cpp:464] Iteration 1663, lr = 1e-05
I1118 03:24:45.635969 22484 solver.cpp:209] Iteration 1664, loss = 0.236967
I1118 03:24:45.636010 22484 solver.cpp:464] Iteration 1664, lr = 1e-05
I1118 03:24:47.778205 22484 solver.cpp:209] Iteration 1665, loss = 0.0744212
I1118 03:24:47.778234 22484 solver.cpp:464] Iteration 1665, lr = 1e-05
I1118 03:24:49.955811 22484 solver.cpp:209] Iteration 1666, loss = 0.288698
I1118 03:24:49.955852 22484 solver.cpp:464] Iteration 1666, lr = 1e-05
I1118 03:24:52.155851 22484 solver.cpp:209] Iteration 1667, loss = 0.169484
I1118 03:24:52.155881 22484 solver.cpp:464] Iteration 1667, lr = 1e-05
I1118 03:24:54.343173 22484 solver.cpp:209] Iteration 1668, loss = 0.195018
I1118 03:24:54.343212 22484 solver.cpp:464] Iteration 1668, lr = 1e-05
I1118 03:24:56.533030 22484 solver.cpp:209] Iteration 1669, loss = 0.123524
I1118 03:24:56.533129 22484 solver.cpp:464] Iteration 1669, lr = 1e-05
I1118 03:24:58.756065 22484 solver.cpp:209] Iteration 1670, loss = 0.176643
I1118 03:24:58.756104 22484 solver.cpp:464] Iteration 1670, lr = 1e-05
I1118 03:25:00.979450 22484 solver.cpp:209] Iteration 1671, loss = 0.273503
I1118 03:25:00.979476 22484 solver.cpp:464] Iteration 1671, lr = 1e-05
I1118 03:25:03.211361 22484 solver.cpp:209] Iteration 1672, loss = 0.311128
I1118 03:25:03.211391 22484 solver.cpp:464] Iteration 1672, lr = 1e-05
I1118 03:25:05.439515 22484 solver.cpp:209] Iteration 1673, loss = 0.0625027
I1118 03:25:05.439544 22484 solver.cpp:464] Iteration 1673, lr = 1e-05
I1118 03:25:07.670904 22484 solver.cpp:209] Iteration 1674, loss = 0.260019
I1118 03:25:07.670934 22484 solver.cpp:464] Iteration 1674, lr = 1e-05
I1118 03:25:09.895385 22484 solver.cpp:209] Iteration 1675, loss = 0.394675
I1118 03:25:09.895426 22484 solver.cpp:464] Iteration 1675, lr = 1e-05
I1118 03:25:12.117713 22484 solver.cpp:209] Iteration 1676, loss = 0.153342
I1118 03:25:12.117743 22484 solver.cpp:464] Iteration 1676, lr = 1e-05
I1118 03:25:14.336149 22484 solver.cpp:209] Iteration 1677, loss = 0.261185
I1118 03:25:14.336191 22484 solver.cpp:464] Iteration 1677, lr = 1e-05
I1118 03:25:16.563216 22484 solver.cpp:209] Iteration 1678, loss = 0.293628
I1118 03:25:16.563244 22484 solver.cpp:464] Iteration 1678, lr = 1e-05
I1118 03:25:18.793581 22484 solver.cpp:209] Iteration 1679, loss = 0.194565
I1118 03:25:18.793611 22484 solver.cpp:464] Iteration 1679, lr = 1e-05
I1118 03:25:21.016937 22484 solver.cpp:209] Iteration 1680, loss = 0.214461
I1118 03:25:21.016975 22484 solver.cpp:464] Iteration 1680, lr = 1e-05
I1118 03:25:23.242503 22484 solver.cpp:209] Iteration 1681, loss = 0.267707
I1118 03:25:23.242545 22484 solver.cpp:464] Iteration 1681, lr = 1e-05
I1118 03:25:25.465523 22484 solver.cpp:209] Iteration 1682, loss = 0.168385
I1118 03:25:25.465564 22484 solver.cpp:464] Iteration 1682, lr = 1e-05
I1118 03:25:27.688985 22484 solver.cpp:209] Iteration 1683, loss = 0.448127
I1118 03:25:27.689075 22484 solver.cpp:464] Iteration 1683, lr = 1e-05
I1118 03:25:29.916679 22484 solver.cpp:209] Iteration 1684, loss = 0.303475
I1118 03:25:29.916708 22484 solver.cpp:464] Iteration 1684, lr = 1e-05
I1118 03:25:32.140527 22484 solver.cpp:209] Iteration 1685, loss = 0.157198
I1118 03:25:32.140557 22484 solver.cpp:464] Iteration 1685, lr = 1e-05
I1118 03:25:34.365236 22484 solver.cpp:209] Iteration 1686, loss = 0.279362
I1118 03:25:34.365265 22484 solver.cpp:464] Iteration 1686, lr = 1e-05
I1118 03:25:36.591001 22484 solver.cpp:209] Iteration 1687, loss = 0.304394
I1118 03:25:36.591029 22484 solver.cpp:464] Iteration 1687, lr = 1e-05
I1118 03:25:38.814978 22484 solver.cpp:209] Iteration 1688, loss = 0.143987
I1118 03:25:38.815019 22484 solver.cpp:464] Iteration 1688, lr = 1e-05
I1118 03:25:41.041113 22484 solver.cpp:209] Iteration 1689, loss = 0.318236
I1118 03:25:41.041141 22484 solver.cpp:464] Iteration 1689, lr = 1e-05
I1118 03:25:43.244189 22484 solver.cpp:209] Iteration 1690, loss = 0.200765
I1118 03:25:43.244230 22484 solver.cpp:464] Iteration 1690, lr = 1e-05
I1118 03:25:45.433346 22484 solver.cpp:209] Iteration 1691, loss = 0.129644
I1118 03:25:45.433387 22484 solver.cpp:464] Iteration 1691, lr = 1e-05
I1118 03:25:47.624706 22484 solver.cpp:209] Iteration 1692, loss = 0.261063
I1118 03:25:47.624747 22484 solver.cpp:464] Iteration 1692, lr = 1e-05
I1118 03:25:49.822193 22484 solver.cpp:209] Iteration 1693, loss = 0.338426
I1118 03:25:49.822222 22484 solver.cpp:464] Iteration 1693, lr = 1e-05
I1118 03:25:52.017788 22484 solver.cpp:209] Iteration 1694, loss = 0.229502
I1118 03:25:52.017828 22484 solver.cpp:464] Iteration 1694, lr = 1e-05
I1118 03:25:54.206954 22484 solver.cpp:209] Iteration 1695, loss = 0.407149
I1118 03:25:54.206984 22484 solver.cpp:464] Iteration 1695, lr = 1e-05
I1118 03:25:56.396432 22484 solver.cpp:209] Iteration 1696, loss = 0.145145
I1118 03:25:56.396461 22484 solver.cpp:464] Iteration 1696, lr = 1e-05
I1118 03:25:58.617121 22484 solver.cpp:209] Iteration 1697, loss = 0.21477
I1118 03:25:58.617224 22484 solver.cpp:464] Iteration 1697, lr = 1e-05
I1118 03:26:00.814692 22484 solver.cpp:209] Iteration 1698, loss = 0.192336
I1118 03:26:00.814718 22484 solver.cpp:464] Iteration 1698, lr = 1e-05
I1118 03:26:03.014616 22484 solver.cpp:209] Iteration 1699, loss = 0.128223
I1118 03:26:03.014646 22484 solver.cpp:464] Iteration 1699, lr = 1e-05
I1118 03:26:03.015249 22484 solver.cpp:264] Iteration 1700, Testing net (#0)
I1118 03:26:16.787164 22484 solver.cpp:305] Test loss: 0.238489
I1118 03:26:16.787206 22484 solver.cpp:318] mean_score = test_score[0] { = 513} / test_score[1] { = 517 }
I1118 03:26:16.787214 22484 solver.cpp:319]            = 0.992263
I1118 03:26:16.787219 22484 solver.cpp:328]     Test net output #0: accuracy = 0.992263
I1118 03:26:16.787222 22484 solver.cpp:318] mean_score = test_score[2] { = 5} / test_score[3] { = 59 }
I1118 03:26:16.787227 22484 solver.cpp:319]            = 0.0847458
I1118 03:26:16.787231 22484 solver.cpp:328]     Test net output #1: accuracy = 0.0847458
I1118 03:26:16.787235 22484 solver.cpp:332]     Test net output #2: accuracy = 0.899306
I1118 03:26:16.787240 22484 solver.cpp:334]     Test net output #3: accuracy = 0.538504
I1118 03:26:17.438789 22484 solver.cpp:209] Iteration 1700, loss = 0.377078
I1118 03:26:17.438819 22484 solver.cpp:464] Iteration 1700, lr = 1e-05
I1118 03:26:19.675261 22484 solver.cpp:209] Iteration 1701, loss = 0.211594
I1118 03:26:19.675302 22484 solver.cpp:464] Iteration 1701, lr = 1e-05
I1118 03:26:21.920945 22484 solver.cpp:209] Iteration 1702, loss = 0.304834
I1118 03:26:21.920984 22484 solver.cpp:464] Iteration 1702, lr = 1e-05
I1118 03:26:24.160951 22484 solver.cpp:209] Iteration 1703, loss = 0.148888
I1118 03:26:24.160991 22484 solver.cpp:464] Iteration 1703, lr = 1e-05
I1118 03:26:26.400853 22484 solver.cpp:209] Iteration 1704, loss = 0.203246
I1118 03:26:26.400894 22484 solver.cpp:464] Iteration 1704, lr = 1e-05
I1118 03:26:28.653908 22484 solver.cpp:209] Iteration 1705, loss = 0.273784
I1118 03:26:28.654006 22484 solver.cpp:464] Iteration 1705, lr = 1e-05
I1118 03:26:30.900826 22484 solver.cpp:209] Iteration 1706, loss = 0.221384
I1118 03:26:30.900862 22484 solver.cpp:464] Iteration 1706, lr = 1e-05
I1118 03:26:33.148370 22484 solver.cpp:209] Iteration 1707, loss = 0.389977
I1118 03:26:33.148411 22484 solver.cpp:464] Iteration 1707, lr = 1e-05
I1118 03:26:35.410778 22484 solver.cpp:209] Iteration 1708, loss = 0.4532
I1118 03:26:35.410807 22484 solver.cpp:464] Iteration 1708, lr = 1e-05
I1118 03:26:37.637841 22484 solver.cpp:209] Iteration 1709, loss = 0.496354
I1118 03:26:37.637869 22484 solver.cpp:464] Iteration 1709, lr = 1e-05
I1118 03:26:39.858474 22484 solver.cpp:209] Iteration 1710, loss = 0.310298
I1118 03:26:39.858502 22484 solver.cpp:464] Iteration 1710, lr = 1e-05
I1118 03:26:42.081944 22484 solver.cpp:209] Iteration 1711, loss = 0.285742
I1118 03:26:42.081985 22484 solver.cpp:464] Iteration 1711, lr = 1e-05
I1118 03:26:44.304496 22484 solver.cpp:209] Iteration 1712, loss = 0.352761
I1118 03:26:44.304524 22484 solver.cpp:464] Iteration 1712, lr = 1e-05
I1118 03:26:46.496156 22484 solver.cpp:209] Iteration 1713, loss = 0.270109
I1118 03:26:46.496184 22484 solver.cpp:464] Iteration 1713, lr = 1e-05
I1118 03:26:48.701128 22484 solver.cpp:209] Iteration 1714, loss = 0.324129
I1118 03:26:48.701156 22484 solver.cpp:464] Iteration 1714, lr = 1e-05
I1118 03:26:50.889219 22484 solver.cpp:209] Iteration 1715, loss = 0.317867
I1118 03:26:50.889245 22484 solver.cpp:464] Iteration 1715, lr = 1e-05
I1118 03:26:53.082782 22484 solver.cpp:209] Iteration 1716, loss = 0.352992
I1118 03:26:53.082813 22484 solver.cpp:464] Iteration 1716, lr = 1e-05
I1118 03:26:55.271536 22484 solver.cpp:209] Iteration 1717, loss = 0.27216
I1118 03:26:55.271576 22484 solver.cpp:464] Iteration 1717, lr = 1e-05
I1118 03:26:57.467167 22484 solver.cpp:209] Iteration 1718, loss = 0.234427
I1118 03:26:57.467195 22484 solver.cpp:464] Iteration 1718, lr = 1e-05
I1118 03:26:59.666993 22484 solver.cpp:209] Iteration 1719, loss = 0.362019
I1118 03:26:59.667095 22484 solver.cpp:464] Iteration 1719, lr = 1e-05
I1118 03:27:01.857260 22484 solver.cpp:209] Iteration 1720, loss = 0.205381
I1118 03:27:01.857290 22484 solver.cpp:464] Iteration 1720, lr = 1e-05
I1118 03:27:04.047400 22484 solver.cpp:209] Iteration 1721, loss = 0.213432
I1118 03:27:04.047430 22484 solver.cpp:464] Iteration 1721, lr = 1e-05
I1118 03:27:06.235540 22484 solver.cpp:209] Iteration 1722, loss = 0.193118
I1118 03:27:06.235570 22484 solver.cpp:464] Iteration 1722, lr = 1e-05
I1118 03:27:08.424940 22484 solver.cpp:209] Iteration 1723, loss = 0.492424
I1118 03:27:08.424968 22484 solver.cpp:464] Iteration 1723, lr = 1e-05
I1118 03:27:10.632220 22484 solver.cpp:209] Iteration 1724, loss = 0.199488
I1118 03:27:10.632249 22484 solver.cpp:464] Iteration 1724, lr = 1e-05
I1118 03:27:12.821782 22484 solver.cpp:209] Iteration 1725, loss = 0.236642
I1118 03:27:12.821822 22484 solver.cpp:464] Iteration 1725, lr = 1e-05
I1118 03:27:15.018549 22484 solver.cpp:209] Iteration 1726, loss = 0.209116
I1118 03:27:15.018597 22484 solver.cpp:464] Iteration 1726, lr = 1e-05
I1118 03:27:17.209843 22484 solver.cpp:209] Iteration 1727, loss = 0.246144
I1118 03:27:17.209884 22484 solver.cpp:464] Iteration 1727, lr = 1e-05
I1118 03:27:19.416857 22484 solver.cpp:209] Iteration 1728, loss = 0.243571
I1118 03:27:19.416898 22484 solver.cpp:464] Iteration 1728, lr = 1e-05
I1118 03:27:21.644508 22484 solver.cpp:209] Iteration 1729, loss = 0.428158
I1118 03:27:21.644537 22484 solver.cpp:464] Iteration 1729, lr = 1e-05
I1118 03:27:23.875946 22484 solver.cpp:209] Iteration 1730, loss = 0.38966
I1118 03:27:23.875974 22484 solver.cpp:464] Iteration 1730, lr = 1e-05
I1118 03:27:26.109534 22484 solver.cpp:209] Iteration 1731, loss = 0.521073
I1118 03:27:26.109575 22484 solver.cpp:464] Iteration 1731, lr = 1e-05
I1118 03:27:28.353749 22484 solver.cpp:209] Iteration 1732, loss = 0.110951
I1118 03:27:28.353780 22484 solver.cpp:464] Iteration 1732, lr = 1e-05
I1118 03:27:30.576331 22484 solver.cpp:209] Iteration 1733, loss = 0.106244
I1118 03:27:30.576429 22484 solver.cpp:464] Iteration 1733, lr = 1e-05
I1118 03:27:32.800340 22484 solver.cpp:209] Iteration 1734, loss = 0.127151
I1118 03:27:32.800369 22484 solver.cpp:464] Iteration 1734, lr = 1e-05
I1118 03:27:35.020895 22484 solver.cpp:209] Iteration 1735, loss = 0.476699
I1118 03:27:35.020936 22484 solver.cpp:464] Iteration 1735, lr = 1e-05
I1118 03:27:37.242630 22484 solver.cpp:209] Iteration 1736, loss = 0.210466
I1118 03:27:37.242660 22484 solver.cpp:464] Iteration 1736, lr = 1e-05
I1118 03:27:39.475952 22484 solver.cpp:209] Iteration 1737, loss = 0.406725
I1118 03:27:39.475980 22484 solver.cpp:464] Iteration 1737, lr = 1e-05
I1118 03:27:41.702517 22484 solver.cpp:209] Iteration 1738, loss = 0.190825
I1118 03:27:41.702559 22484 solver.cpp:464] Iteration 1738, lr = 1e-05
I1118 03:27:43.934088 22484 solver.cpp:209] Iteration 1739, loss = 0.305239
I1118 03:27:43.934129 22484 solver.cpp:464] Iteration 1739, lr = 1e-05
I1118 03:27:46.159731 22484 solver.cpp:209] Iteration 1740, loss = 0.228259
I1118 03:27:46.159759 22484 solver.cpp:464] Iteration 1740, lr = 1e-05
I1118 03:27:48.382802 22484 solver.cpp:209] Iteration 1741, loss = 0.256659
I1118 03:27:48.382828 22484 solver.cpp:464] Iteration 1741, lr = 1e-05
I1118 03:27:50.608150 22484 solver.cpp:209] Iteration 1742, loss = 0.283898
I1118 03:27:50.608177 22484 solver.cpp:464] Iteration 1742, lr = 1e-05
I1118 03:27:52.809988 22484 solver.cpp:209] Iteration 1743, loss = 0.291481
I1118 03:27:52.810027 22484 solver.cpp:464] Iteration 1743, lr = 1e-05
I1118 03:27:55.009033 22484 solver.cpp:209] Iteration 1744, loss = 0.172844
I1118 03:27:55.009063 22484 solver.cpp:464] Iteration 1744, lr = 1e-05
I1118 03:27:57.208936 22484 solver.cpp:209] Iteration 1745, loss = 0.194131
I1118 03:27:57.208966 22484 solver.cpp:464] Iteration 1745, lr = 1e-05
I1118 03:27:59.405531 22484 solver.cpp:209] Iteration 1746, loss = 0.139156
I1118 03:27:59.405560 22484 solver.cpp:464] Iteration 1746, lr = 1e-05
I1118 03:28:01.597743 22484 solver.cpp:209] Iteration 1747, loss = 0.204419
I1118 03:28:01.597820 22484 solver.cpp:464] Iteration 1747, lr = 1e-05
I1118 03:28:03.784318 22484 solver.cpp:209] Iteration 1748, loss = 0.221378
I1118 03:28:03.784346 22484 solver.cpp:464] Iteration 1748, lr = 1e-05
I1118 03:28:05.981951 22484 solver.cpp:209] Iteration 1749, loss = 0.280752
I1118 03:28:05.981981 22484 solver.cpp:464] Iteration 1749, lr = 1e-05
I1118 03:28:05.982568 22484 solver.cpp:264] Iteration 1750, Testing net (#0)
I1118 03:28:19.733047 22484 solver.cpp:305] Test loss: 0.249777
I1118 03:28:19.733074 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 03:28:19.733094 22484 solver.cpp:319]            = 0.98646
I1118 03:28:19.733099 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 03:28:19.733103 22484 solver.cpp:318] mean_score = test_score[2] { = 10} / test_score[3] { = 59 }
I1118 03:28:19.733108 22484 solver.cpp:319]            = 0.169492
I1118 03:28:19.733113 22484 solver.cpp:328]     Test net output #1: accuracy = 0.169492
I1118 03:28:19.733116 22484 solver.cpp:332]     Test net output #2: accuracy = 0.902778
I1118 03:28:19.733120 22484 solver.cpp:334]     Test net output #3: accuracy = 0.577976
I1118 03:28:20.381290 22484 solver.cpp:209] Iteration 1750, loss = 0.188763
I1118 03:28:20.381331 22484 solver.cpp:464] Iteration 1750, lr = 1e-05
I1118 03:28:22.612530 22484 solver.cpp:209] Iteration 1751, loss = 0.243502
I1118 03:28:22.612560 22484 solver.cpp:464] Iteration 1751, lr = 1e-05
I1118 03:28:24.843940 22484 solver.cpp:209] Iteration 1752, loss = 0.273706
I1118 03:28:24.843982 22484 solver.cpp:464] Iteration 1752, lr = 1e-05
I1118 03:28:27.065853 22484 solver.cpp:209] Iteration 1753, loss = 0.145604
I1118 03:28:27.065893 22484 solver.cpp:464] Iteration 1753, lr = 1e-05
I1118 03:28:29.291790 22484 solver.cpp:209] Iteration 1754, loss = 0.22361
I1118 03:28:29.291820 22484 solver.cpp:464] Iteration 1754, lr = 1e-05
I1118 03:28:31.520140 22484 solver.cpp:209] Iteration 1755, loss = 0.341512
I1118 03:28:31.520181 22484 solver.cpp:464] Iteration 1755, lr = 1e-05
I1118 03:28:33.746454 22484 solver.cpp:209] Iteration 1756, loss = 0.199065
I1118 03:28:33.746552 22484 solver.cpp:464] Iteration 1756, lr = 1e-05
I1118 03:28:35.980195 22484 solver.cpp:209] Iteration 1757, loss = 0.213107
I1118 03:28:35.980237 22484 solver.cpp:464] Iteration 1757, lr = 1e-05
I1118 03:28:38.202865 22484 solver.cpp:209] Iteration 1758, loss = 0.125167
I1118 03:28:38.202906 22484 solver.cpp:464] Iteration 1758, lr = 1e-05
I1118 03:28:40.434955 22484 solver.cpp:209] Iteration 1759, loss = 0.366362
I1118 03:28:40.434984 22484 solver.cpp:464] Iteration 1759, lr = 1e-05
I1118 03:28:42.661509 22484 solver.cpp:209] Iteration 1760, loss = 0.236026
I1118 03:28:42.661540 22484 solver.cpp:464] Iteration 1760, lr = 1e-05
I1118 03:28:44.881198 22484 solver.cpp:209] Iteration 1761, loss = 0.536332
I1118 03:28:44.881239 22484 solver.cpp:464] Iteration 1761, lr = 1e-05
I1118 03:28:47.107014 22484 solver.cpp:209] Iteration 1762, loss = 0.0686288
I1118 03:28:47.107056 22484 solver.cpp:464] Iteration 1762, lr = 1e-05
I1118 03:28:49.328531 22484 solver.cpp:209] Iteration 1763, loss = 0.222776
I1118 03:28:49.328560 22484 solver.cpp:464] Iteration 1763, lr = 1e-05
I1118 03:28:51.566190 22484 solver.cpp:209] Iteration 1764, loss = 0.0899893
I1118 03:28:51.566220 22484 solver.cpp:464] Iteration 1764, lr = 1e-05
I1118 03:28:53.798487 22484 solver.cpp:209] Iteration 1765, loss = 0.0909819
I1118 03:28:53.798516 22484 solver.cpp:464] Iteration 1765, lr = 1e-05
I1118 03:28:56.030525 22484 solver.cpp:209] Iteration 1766, loss = 0.188345
I1118 03:28:56.030555 22484 solver.cpp:464] Iteration 1766, lr = 1e-05
I1118 03:28:58.269098 22484 solver.cpp:209] Iteration 1767, loss = 0.223219
I1118 03:28:58.269136 22484 solver.cpp:464] Iteration 1767, lr = 1e-05
I1118 03:29:00.486840 22484 solver.cpp:209] Iteration 1768, loss = 0.213401
I1118 03:29:00.486868 22484 solver.cpp:464] Iteration 1768, lr = 1e-05
I1118 03:29:02.710952 22484 solver.cpp:209] Iteration 1769, loss = 0.182364
I1118 03:29:02.710981 22484 solver.cpp:464] Iteration 1769, lr = 1e-05
I1118 03:29:04.939983 22484 solver.cpp:209] Iteration 1770, loss = 0.190615
I1118 03:29:04.940050 22484 solver.cpp:464] Iteration 1770, lr = 1e-05
I1118 03:29:07.174679 22484 solver.cpp:209] Iteration 1771, loss = 0.198649
I1118 03:29:07.174708 22484 solver.cpp:464] Iteration 1771, lr = 1e-05
I1118 03:29:09.392318 22484 solver.cpp:209] Iteration 1772, loss = 0.117791
I1118 03:29:09.392348 22484 solver.cpp:464] Iteration 1772, lr = 1e-05
I1118 03:29:11.583721 22484 solver.cpp:209] Iteration 1773, loss = 0.201793
I1118 03:29:11.583763 22484 solver.cpp:464] Iteration 1773, lr = 1e-05
I1118 03:29:13.773386 22484 solver.cpp:209] Iteration 1774, loss = 0.241217
I1118 03:29:13.773416 22484 solver.cpp:464] Iteration 1774, lr = 1e-05
I1118 03:29:15.967779 22484 solver.cpp:209] Iteration 1775, loss = 0.261773
I1118 03:29:15.967820 22484 solver.cpp:464] Iteration 1775, lr = 1e-05
I1118 03:29:18.159580 22484 solver.cpp:209] Iteration 1776, loss = 0.0815728
I1118 03:29:18.159607 22484 solver.cpp:464] Iteration 1776, lr = 1e-05
I1118 03:29:20.358419 22484 solver.cpp:209] Iteration 1777, loss = 0.427057
I1118 03:29:20.358448 22484 solver.cpp:464] Iteration 1777, lr = 1e-05
I1118 03:29:22.553647 22484 solver.cpp:209] Iteration 1778, loss = 0.324638
I1118 03:29:22.553678 22484 solver.cpp:464] Iteration 1778, lr = 1e-05
I1118 03:29:24.747578 22484 solver.cpp:209] Iteration 1779, loss = 0.139692
I1118 03:29:24.747608 22484 solver.cpp:464] Iteration 1779, lr = 1e-05
I1118 03:29:26.936254 22484 solver.cpp:209] Iteration 1780, loss = 0.273938
I1118 03:29:26.936285 22484 solver.cpp:464] Iteration 1780, lr = 1e-05
I1118 03:29:29.126780 22484 solver.cpp:209] Iteration 1781, loss = 0.297827
I1118 03:29:29.126808 22484 solver.cpp:464] Iteration 1781, lr = 1e-05
I1118 03:29:31.323282 22484 solver.cpp:209] Iteration 1782, loss = 0.309887
I1118 03:29:31.323310 22484 solver.cpp:464] Iteration 1782, lr = 1e-05
I1118 03:29:33.514833 22484 solver.cpp:209] Iteration 1783, loss = 0.168444
I1118 03:29:33.514863 22484 solver.cpp:464] Iteration 1783, lr = 1e-05
I1118 03:29:35.715446 22484 solver.cpp:209] Iteration 1784, loss = 0.290357
I1118 03:29:35.715545 22484 solver.cpp:464] Iteration 1784, lr = 1e-05
I1118 03:29:37.909721 22484 solver.cpp:209] Iteration 1785, loss = 0.127096
I1118 03:29:37.909762 22484 solver.cpp:464] Iteration 1785, lr = 1e-05
I1118 03:29:40.098469 22484 solver.cpp:209] Iteration 1786, loss = 0.638488
I1118 03:29:40.098498 22484 solver.cpp:464] Iteration 1786, lr = 1e-05
I1118 03:29:42.290988 22484 solver.cpp:209] Iteration 1787, loss = 0.166371
I1118 03:29:42.291030 22484 solver.cpp:464] Iteration 1787, lr = 1e-05
I1118 03:29:44.485903 22484 solver.cpp:209] Iteration 1788, loss = 0.087312
I1118 03:29:44.485944 22484 solver.cpp:464] Iteration 1788, lr = 1e-05
I1118 03:29:46.684101 22484 solver.cpp:209] Iteration 1789, loss = 0.230471
I1118 03:29:46.684129 22484 solver.cpp:464] Iteration 1789, lr = 1e-05
I1118 03:29:48.887181 22484 solver.cpp:209] Iteration 1790, loss = 0.198725
I1118 03:29:48.887222 22484 solver.cpp:464] Iteration 1790, lr = 1e-05
I1118 03:29:51.072609 22484 solver.cpp:209] Iteration 1791, loss = 0.307336
I1118 03:29:51.072646 22484 solver.cpp:464] Iteration 1791, lr = 1e-05
I1118 03:29:53.263116 22484 solver.cpp:209] Iteration 1792, loss = 0.12543
I1118 03:29:53.263157 22484 solver.cpp:464] Iteration 1792, lr = 1e-05
I1118 03:29:55.460373 22484 solver.cpp:209] Iteration 1793, loss = 0.168247
I1118 03:29:55.460402 22484 solver.cpp:464] Iteration 1793, lr = 1e-05
I1118 03:29:57.700464 22484 solver.cpp:209] Iteration 1794, loss = 0.127041
I1118 03:29:57.700491 22484 solver.cpp:464] Iteration 1794, lr = 1e-05
I1118 03:29:59.927227 22484 solver.cpp:209] Iteration 1795, loss = 0.341313
I1118 03:29:59.927268 22484 solver.cpp:464] Iteration 1795, lr = 1e-05
I1118 03:30:02.153524 22484 solver.cpp:209] Iteration 1796, loss = 0.409301
I1118 03:30:02.153566 22484 solver.cpp:464] Iteration 1796, lr = 1e-05
I1118 03:30:04.377179 22484 solver.cpp:209] Iteration 1797, loss = 0.261576
I1118 03:30:04.377220 22484 solver.cpp:464] Iteration 1797, lr = 1e-05
I1118 03:30:06.599649 22484 solver.cpp:209] Iteration 1798, loss = 0.263102
I1118 03:30:06.599752 22484 solver.cpp:464] Iteration 1798, lr = 1e-05
I1118 03:30:08.824808 22484 solver.cpp:209] Iteration 1799, loss = 0.155598
I1118 03:30:08.824836 22484 solver.cpp:464] Iteration 1799, lr = 1e-05
I1118 03:30:08.825426 22484 solver.cpp:264] Iteration 1800, Testing net (#0)
I1118 03:30:22.705020 22484 solver.cpp:305] Test loss: 0.241273
I1118 03:30:22.705060 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 03:30:22.705068 22484 solver.cpp:319]            = 0.98646
I1118 03:30:22.705072 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 03:30:22.705077 22484 solver.cpp:318] mean_score = test_score[2] { = 9} / test_score[3] { = 59 }
I1118 03:30:22.705081 22484 solver.cpp:319]            = 0.152542
I1118 03:30:22.705085 22484 solver.cpp:328]     Test net output #1: accuracy = 0.152542
I1118 03:30:22.705090 22484 solver.cpp:332]     Test net output #2: accuracy = 0.901042
I1118 03:30:22.705093 22484 solver.cpp:334]     Test net output #3: accuracy = 0.569501
I1118 03:30:23.349500 22484 solver.cpp:209] Iteration 1800, loss = 0.196511
I1118 03:30:23.349540 22484 solver.cpp:464] Iteration 1800, lr = 1e-05
I1118 03:30:25.577006 22484 solver.cpp:209] Iteration 1801, loss = 0.1677
I1118 03:30:25.577033 22484 solver.cpp:464] Iteration 1801, lr = 1e-05
I1118 03:30:27.803941 22484 solver.cpp:209] Iteration 1802, loss = 0.0947836
I1118 03:30:27.803968 22484 solver.cpp:464] Iteration 1802, lr = 1e-05
I1118 03:30:30.035457 22484 solver.cpp:209] Iteration 1803, loss = 0.315627
I1118 03:30:30.035486 22484 solver.cpp:464] Iteration 1803, lr = 1e-05
I1118 03:30:32.270275 22484 solver.cpp:209] Iteration 1804, loss = 0.226009
I1118 03:30:32.270304 22484 solver.cpp:464] Iteration 1804, lr = 1e-05
I1118 03:30:34.496026 22484 solver.cpp:209] Iteration 1805, loss = 0.235077
I1118 03:30:34.496065 22484 solver.cpp:464] Iteration 1805, lr = 1e-05
I1118 03:30:36.721998 22484 solver.cpp:209] Iteration 1806, loss = 0.296318
I1118 03:30:36.722095 22484 solver.cpp:464] Iteration 1806, lr = 1e-05
I1118 03:30:38.941509 22484 solver.cpp:209] Iteration 1807, loss = 0.181526
I1118 03:30:38.941550 22484 solver.cpp:464] Iteration 1807, lr = 1e-05
I1118 03:30:41.168009 22484 solver.cpp:209] Iteration 1808, loss = 0.322031
I1118 03:30:41.168035 22484 solver.cpp:464] Iteration 1808, lr = 1e-05
I1118 03:30:43.420157 22484 solver.cpp:209] Iteration 1809, loss = 0.464034
I1118 03:30:43.420186 22484 solver.cpp:464] Iteration 1809, lr = 1e-05
I1118 03:30:45.651504 22484 solver.cpp:209] Iteration 1810, loss = 0.24545
I1118 03:30:45.651532 22484 solver.cpp:464] Iteration 1810, lr = 1e-05
I1118 03:30:47.882344 22484 solver.cpp:209] Iteration 1811, loss = 0.603972
I1118 03:30:47.882386 22484 solver.cpp:464] Iteration 1811, lr = 1e-05
I1118 03:30:50.104276 22484 solver.cpp:209] Iteration 1812, loss = 0.189722
I1118 03:30:50.104317 22484 solver.cpp:464] Iteration 1812, lr = 1e-05
I1118 03:30:52.332569 22484 solver.cpp:209] Iteration 1813, loss = 0.26372
I1118 03:30:52.332597 22484 solver.cpp:464] Iteration 1813, lr = 1e-05
I1118 03:30:54.554962 22484 solver.cpp:209] Iteration 1814, loss = 0.32933
I1118 03:30:54.554991 22484 solver.cpp:464] Iteration 1814, lr = 1e-05
I1118 03:30:56.778216 22484 solver.cpp:209] Iteration 1815, loss = 0.380137
I1118 03:30:56.778259 22484 solver.cpp:464] Iteration 1815, lr = 1e-05
I1118 03:30:59.018196 22484 solver.cpp:209] Iteration 1816, loss = 0.257723
I1118 03:30:59.018224 22484 solver.cpp:464] Iteration 1816, lr = 1e-05
I1118 03:31:01.235703 22484 solver.cpp:209] Iteration 1817, loss = 0.268495
I1118 03:31:01.235744 22484 solver.cpp:464] Iteration 1817, lr = 1e-05
I1118 03:31:03.434128 22484 solver.cpp:209] Iteration 1818, loss = 0.449898
I1118 03:31:03.434170 22484 solver.cpp:464] Iteration 1818, lr = 1e-05
I1118 03:31:05.624352 22484 solver.cpp:209] Iteration 1819, loss = 0.422423
I1118 03:31:05.624393 22484 solver.cpp:464] Iteration 1819, lr = 1e-05
I1118 03:31:07.816318 22484 solver.cpp:209] Iteration 1820, loss = 0.286315
I1118 03:31:07.816395 22484 solver.cpp:464] Iteration 1820, lr = 1e-05
I1118 03:31:10.013645 22484 solver.cpp:209] Iteration 1821, loss = 0.41131
I1118 03:31:10.013674 22484 solver.cpp:464] Iteration 1821, lr = 1e-05
I1118 03:31:12.210110 22484 solver.cpp:209] Iteration 1822, loss = 0.215968
I1118 03:31:12.210151 22484 solver.cpp:464] Iteration 1822, lr = 1e-05
I1118 03:31:14.412731 22484 solver.cpp:209] Iteration 1823, loss = 0.310593
I1118 03:31:14.412761 22484 solver.cpp:464] Iteration 1823, lr = 1e-05
I1118 03:31:16.612270 22484 solver.cpp:209] Iteration 1824, loss = 0.192626
I1118 03:31:16.612299 22484 solver.cpp:464] Iteration 1824, lr = 1e-05
I1118 03:31:18.805151 22484 solver.cpp:209] Iteration 1825, loss = 0.2117
I1118 03:31:18.805181 22484 solver.cpp:464] Iteration 1825, lr = 1e-05
I1118 03:31:20.996788 22484 solver.cpp:209] Iteration 1826, loss = 0.446564
I1118 03:31:20.996815 22484 solver.cpp:464] Iteration 1826, lr = 1e-05
I1118 03:31:23.183218 22484 solver.cpp:209] Iteration 1827, loss = 0.176566
I1118 03:31:23.183259 22484 solver.cpp:464] Iteration 1827, lr = 1e-05
I1118 03:31:25.386220 22484 solver.cpp:209] Iteration 1828, loss = 0.28309
I1118 03:31:25.386248 22484 solver.cpp:464] Iteration 1828, lr = 1e-05
I1118 03:31:27.576658 22484 solver.cpp:209] Iteration 1829, loss = 0.157603
I1118 03:31:27.576689 22484 solver.cpp:464] Iteration 1829, lr = 1e-05
I1118 03:31:29.771710 22484 solver.cpp:209] Iteration 1830, loss = 0.291448
I1118 03:31:29.771751 22484 solver.cpp:464] Iteration 1830, lr = 1e-05
I1118 03:31:31.962574 22484 solver.cpp:209] Iteration 1831, loss = 0.243219
I1118 03:31:31.962611 22484 solver.cpp:464] Iteration 1831, lr = 1e-05
I1118 03:31:34.149646 22484 solver.cpp:209] Iteration 1832, loss = 0.341208
I1118 03:31:34.149674 22484 solver.cpp:464] Iteration 1832, lr = 1e-05
I1118 03:31:36.346571 22484 solver.cpp:209] Iteration 1833, loss = 0.551192
I1118 03:31:36.346621 22484 solver.cpp:464] Iteration 1833, lr = 1e-05
I1118 03:31:38.541120 22484 solver.cpp:209] Iteration 1834, loss = 0.27953
I1118 03:31:38.541182 22484 solver.cpp:464] Iteration 1834, lr = 1e-05
I1118 03:31:40.749470 22484 solver.cpp:209] Iteration 1835, loss = 0.0828286
I1118 03:31:40.749500 22484 solver.cpp:464] Iteration 1835, lr = 1e-05
I1118 03:31:42.975846 22484 solver.cpp:209] Iteration 1836, loss = 0.117995
I1118 03:31:42.975888 22484 solver.cpp:464] Iteration 1836, lr = 1e-05
I1118 03:31:45.199157 22484 solver.cpp:209] Iteration 1837, loss = 0.212918
I1118 03:31:45.199199 22484 solver.cpp:464] Iteration 1837, lr = 1e-05
I1118 03:31:47.423653 22484 solver.cpp:209] Iteration 1838, loss = 0.351423
I1118 03:31:47.423682 22484 solver.cpp:464] Iteration 1838, lr = 1e-05
I1118 03:31:49.644456 22484 solver.cpp:209] Iteration 1839, loss = 0.222045
I1118 03:31:49.644485 22484 solver.cpp:464] Iteration 1839, lr = 1e-05
I1118 03:31:51.880081 22484 solver.cpp:209] Iteration 1840, loss = 0.505119
I1118 03:31:51.880123 22484 solver.cpp:464] Iteration 1840, lr = 1e-05
I1118 03:31:54.111812 22484 solver.cpp:209] Iteration 1841, loss = 0.0636024
I1118 03:31:54.111842 22484 solver.cpp:464] Iteration 1841, lr = 1e-05
I1118 03:31:56.342980 22484 solver.cpp:209] Iteration 1842, loss = 0.266477
I1118 03:31:56.343022 22484 solver.cpp:464] Iteration 1842, lr = 1e-05
I1118 03:31:58.571837 22484 solver.cpp:209] Iteration 1843, loss = 0.258579
I1118 03:31:58.571866 22484 solver.cpp:464] Iteration 1843, lr = 1e-05
I1118 03:32:00.788805 22484 solver.cpp:209] Iteration 1844, loss = 0.285196
I1118 03:32:00.788832 22484 solver.cpp:464] Iteration 1844, lr = 1e-05
I1118 03:32:03.012853 22484 solver.cpp:209] Iteration 1845, loss = 0.3445
I1118 03:32:03.012882 22484 solver.cpp:464] Iteration 1845, lr = 1e-05
I1118 03:32:05.236866 22484 solver.cpp:209] Iteration 1846, loss = 0.217919
I1118 03:32:05.236907 22484 solver.cpp:464] Iteration 1846, lr = 1e-05
I1118 03:32:07.469822 22484 solver.cpp:209] Iteration 1847, loss = 0.279131
I1118 03:32:07.469851 22484 solver.cpp:464] Iteration 1847, lr = 1e-05
I1118 03:32:09.703012 22484 solver.cpp:209] Iteration 1848, loss = 0.150222
I1118 03:32:09.703093 22484 solver.cpp:464] Iteration 1848, lr = 1e-05
I1118 03:32:11.953101 22484 solver.cpp:209] Iteration 1849, loss = 0.131701
I1118 03:32:11.953130 22484 solver.cpp:464] Iteration 1849, lr = 1e-05
I1118 03:32:11.953743 22484 solver.cpp:264] Iteration 1850, Testing net (#0)
I1118 03:32:25.822957 22484 solver.cpp:305] Test loss: 0.240107
I1118 03:32:25.822999 22484 solver.cpp:318] mean_score = test_score[0] { = 510} / test_score[1] { = 517 }
I1118 03:32:25.823005 22484 solver.cpp:319]            = 0.98646
I1118 03:32:25.823010 22484 solver.cpp:328]     Test net output #0: accuracy = 0.98646
I1118 03:32:25.823015 22484 solver.cpp:318] mean_score = test_score[2] { = 8} / test_score[3] { = 59 }
I1118 03:32:25.823019 22484 solver.cpp:319]            = 0.135593
I1118 03:32:25.823024 22484 solver.cpp:328]     Test net output #1: accuracy = 0.135593
I1118 03:32:25.823027 22484 solver.cpp:332]     Test net output #2: accuracy = 0.899306
I1118 03:32:25.823031 22484 solver.cpp:334]     Test net output #3: accuracy = 0.561027
I1118 03:32:26.469549 22484 solver.cpp:209] Iteration 1850, loss = 0.25292
I1118 03:32:26.469590 22484 solver.cpp:464] Iteration 1850, lr = 1e-05
I1118 03:32:28.703665 22484 solver.cpp:209] Iteration 1851, loss = 0.209669
I1118 03:32:28.703706 22484 solver.cpp:464] Iteration 1851, lr = 1e-05
I1118 03:32:30.925935 22484 solver.cpp:209] Iteration 1852, loss = 0.149838
I1118 03:32:30.925961 22484 solver.cpp:464] Iteration 1852, lr = 1e-05
I1118 03:32:33.151352 22484 solver.cpp:209] Iteration 1853, loss = 0.343846
I1118 03:32:33.151381 22484 solver.cpp:464] Iteration 1853, lr = 1e-05
I1118 03:32:35.371110 22484 solver.cpp:209] Iteration 1854, loss = 0.250789
I1118 03:32:35.371140 22484 solver.cpp:464] Iteration 1854, lr = 1e-05
I1118 03:32:37.601021 22484 solver.cpp:209] Iteration 1855, loss = 0.192765
I1118 03:32:37.601047 22484 solver.cpp:464] Iteration 1855, lr = 1e-05
I1118 03:32:39.836568 22484 solver.cpp:209] Iteration 1856, loss = 0.151537
I1118 03:32:39.836669 22484 solver.cpp:464] Iteration 1856, lr = 1e-05
I1118 03:32:42.063505 22484 solver.cpp:209] Iteration 1857, loss = 0.261249
I1118 03:32:42.063547 22484 solver.cpp:464] Iteration 1857, lr = 1e-05
I1118 03:32:44.290552 22484 solver.cpp:209] Iteration 1858, loss = 0.476122
I1118 03:32:44.290601 22484 solver.cpp:464] Iteration 1858, lr = 1e-05
I1118 03:32:46.508858 22484 solver.cpp:209] Iteration 1859, loss = 0.0992826
I1118 03:32:46.508888 22484 solver.cpp:464] Iteration 1859, lr = 1e-05
I1118 03:32:48.731916 22484 solver.cpp:209] Iteration 1860, loss = 0.170073
I1118 03:32:48.731957 22484 solver.cpp:464] Iteration 1860, lr = 1e-05
I1118 03:32:50.955819 22484 solver.cpp:209] Iteration 1861, loss = 0.226722
I1118 03:32:50.955844 22484 solver.cpp:464] Iteration 1861, lr = 1e-05
I1118 03:32:53.174465 22484 solver.cpp:209] Iteration 1862, loss = 0.332882
I1118 03:32:53.174506 22484 solver.cpp:464] Iteration 1862, lr = 1e-05
I1118 03:32:55.376081 22484 solver.cpp:209] Iteration 1863, loss = 0.337831
I1118 03:32:55.376121 22484 solver.cpp:464] Iteration 1863, lr = 1e-05
I1118 03:32:57.575495 22484 solver.cpp:209] Iteration 1864, loss = 0.343582
I1118 03:32:57.575523 22484 solver.cpp:464] Iteration 1864, lr = 1e-05
I1118 03:32:59.765717 22484 solver.cpp:209] Iteration 1865, loss = 0.107911
I1118 03:32:59.765758 22484 solver.cpp:464] Iteration 1865, lr = 1e-05
I1118 03:33:01.955160 22484 solver.cpp:209] Iteration 1866, loss = 0.356948
I1118 03:33:01.955191 22484 solver.cpp:464] Iteration 1866, lr = 1e-05
I1118 03:33:04.142390 22484 solver.cpp:209] Iteration 1867, loss = 0.0756811
I1118 03:33:04.142420 22484 solver.cpp:464] Iteration 1867, lr = 1e-05
I1118 03:33:06.346235 22484 solver.cpp:209] Iteration 1868, loss = 0.100679
I1118 03:33:06.346277 22484 solver.cpp:464] Iteration 1868, lr = 1e-05
I1118 03:33:08.538491 22484 solver.cpp:209] Iteration 1869, loss = 0.109941
I1118 03:33:08.538533 22484 solver.cpp:464] Iteration 1869, lr = 1e-05
I1118 03:33:10.734997 22484 solver.cpp:209] Iteration 1870, loss = 0.187666
I1118 03:33:10.735064 22484 solver.cpp:464] Iteration 1870, lr = 1e-05
I1118 03:33:12.925258 22484 solver.cpp:209] Iteration 1871, loss = 0.233362
I1118 03:33:12.925287 22484 solver.cpp:464] Iteration 1871, lr = 1e-05
I1118 03:33:15.114622 22484 solver.cpp:209] Iteration 1872, loss = 0.269213
I1118 03:33:15.114651 22484 solver.cpp:464] Iteration 1872, lr = 1e-05
I1118 03:33:17.315754 22484 solver.cpp:209] Iteration 1873, loss = 0.132046
I1118 03:33:17.315795 22484 solver.cpp:464] Iteration 1873, lr = 1e-05
I1118 03:33:19.511028 22484 solver.cpp:209] Iteration 1874, loss = 0.191811
I1118 03:33:19.511056 22484 solver.cpp:464] Iteration 1874, lr = 1e-05
I1118 03:33:21.707783 22484 solver.cpp:209] Iteration 1875, loss = 0.132739
I1118 03:33:21.707825 22484 solver.cpp:464] Iteration 1875, lr = 1e-05
