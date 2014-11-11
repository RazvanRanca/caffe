import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../python"))
from caffe.proto import caffe_pb2

def b2a(fn):
  blob = caffe_pb2.BlobProto()
  with open(fn, 'rb') as f:
    blob.ParseFromString(f.read())
  arr = np.array(blob.data).reshape(blob.channels, blob.height, blob.width)
  np.save("alexMean", arr)
  #return [blobproto_to_array(blob) for blob in vec.blobs]

def oa(b,g,r): 
  dim = 256
  arr = np.array([[[b]*dim]*dim, [[g]*dim]*dim, [[r]*dim]*dim])
  np.save("razMean", arr)

if __name__ == "__main__":
  # b2a(sys.argv[1])
  oa(86.752, 101.460, 104.600)

