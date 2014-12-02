import cPickle
import numpy as np

with open("clamp_1flag.npy", 'r') as f:
  rezs = cPickle.load(f)

flag = "NoClampUsed" 
tp, fp, tn, fneg = 0, 0, 0, 0

for rez in rezs:
  fn, pos = rez.split()
  pos = int(pos)
  with open("/data/ad6813/pipe-data/Bluebox/" + fn + ".dat") as f:
    if flag in f.read():
      if pos > 0.5:
        tp += 1
      else:
        fneg += 1
    else:
      if pos > 0.5:
        fp += 1
      else:
        tn += 1
print tp, tn, fp, fneg, tp/float(tp+fneg), tn/float(tn + fp), (tp + tn) / float(100)
