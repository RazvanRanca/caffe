#!/usr/bin/python
from PIL import Image
import sys
import os

if __name__ == "__main__":
  chans = [0,0,0]
  count = 0
  dr = sys.argv[1]
  for fl in os.listdir(dr):
    if fl.endswith(".jpg"):
      count += 1
      if count % 10 == 0:
        print count
        print map(lambda x:float(x)/count, chans)
      im = Image.open(dr + "/" + fl)
      pix = im.load()
      sx, sy = im.size
      c1, c2, c3 = 0, 0, 0
      for x in range(sx):
        for y in range(sy):
          p1, p2, p3 = pix[x,y]
          c1 += p1
          c2 += p2
          c3 += p3
      chans[0] += c1/(sx*sy)
      chans[1] += c2/(sx*sy)
      chans[2] += c3/(sx*sy)
  print map(lambda x:float(x)/count, chans)
