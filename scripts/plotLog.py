from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
  trainLoss = []
  trainIter = []
  testIter = []
  testLoss = []
  testAccs = {}
  nameMap = {'#0':"Class 1", '#1':"Class 2", '#2':"Overall", '#3':"Per clas"} 
  with open(sys.argv[1], 'r') as f:
    for line in f:
      try:
        start, fin = line.split(',')
      except:
        fin = line
      fin = fin.strip()
      if fin.startswith("loss"):
        trainLoss.append(float(fin.split()[-1]))
        trainIter.append(int(start.strip().split()[-1]))
      elif fin.startswith("Testing"):
        testIter.append(int(start.strip().split()[-1]))
        print line
      elif "Test loss" in fin:
        testLoss.append(float(fin.split()[-1]))
      elif "Test net output" in fin:
        name = fin.split(':')[-2].split()[-1]
        try:
          testAccs[name].append(float(fin.split()[-1]))
        except:
          testAccs[name] = [float(fin.split()[-1])]
  print testIter
  names = ['trainLoss', 'testLoss']
  axs = []
  ax, = plt.plot(trainIter, trainLoss, 'b', alpha=0.5)
  axs.append(ax)
  ax, = plt.plot(testIter, testLoss, 'r', lw=2, alpha=0.5)
  axs.append(ax)
  cols = ['g','k','c','m']
  ind = 0
  for name, accs in sorted(testAccs.items()):
    names.append(nameMap[name])
    ax, = plt.plot(testIter, accs, cols[ind], lw=2,  alpha=0.8)
    axs.append(ax)
    ind += 1
  plt.legend(axs, names, loc="best")
  plt.ylim([0,1.1])
  plt.show()
