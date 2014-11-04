import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../python"))
from caffe.proto import caffe_pb2

def b2a(fn):
  blob = caffe_pb2.BlobProto()
  with open(fn, 'rb') as f:
    blob.ParseFromString(f.read())
  print np.array(blob.data).reshape(blob.num, blob.channels, blob.height, blob.width)
  #return [blobproto_to_array(blob) for blob in vec.blobs]

if __name__ == "__main__":
  b2a("../mean")
