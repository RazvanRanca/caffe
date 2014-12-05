import cPickle
from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
  with open(sys.argv[1], 'r') as f:
    dur = cPickle.load(f)

  relDur = filter(lambda x: x > 0 and x < 10000, dur.values())
  negDur = filter(lambda x: x < 0, dur.values())
  print len(relDur)/float(len(dur)), len(negDur)/float(len(dur))
  plt.hist(relDur, 100)
  plt.show()
