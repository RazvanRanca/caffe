import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../python/"))
import caffe.io

h = 256 
w = 256
mp = [103.939, 116.779, 123.68]
a = np.zeros((1,3, h,w), dtype=np.float32)
for c in range(3):
  for i in range(h):
    for j in range(w):
      a[0][c][i][j] = mp[c]

with open("meanVal.binaryproto",'w') as f:
  f.write(caffe.io.arraylist_to_blobprotovecor_str([a])[0])

with open ("/homes/ad6813/data/controlpoint_mean_256-227.binaryproto", 'r') as f:
  s = f.read()

print len(s)
print caffe.io.blobprotovector_str_to_arraylist(s)

#with open ("oxfordMean.binaryproto", 'r') as f:
#  s = f.read()

#print caffe.io.blobprotovector_str_to_arraylist(s)

