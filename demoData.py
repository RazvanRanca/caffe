import xlrd
import numpy as np
from matplotlib import pyplot as plt

def getVals(sh, flag, mfn = None):
  col = flags[flag]
  human = [int(sh.cell_value(i, col)) for i in range(2, 849)]
  if mfn:
    mDict = {}
    with open(mfn, 'r') as f:
      for line in f:
        joint, val = line.strip().split()
        mDict[int(joint)] = float(val)
    machine = []
    for joint in joints:
      machine.append(mDict[joint])
  else:
    machine = [float(sh.cell_value(i, col+1)) for i in range(2, 849)]
  return human, machine

def readVals(hfn, mfn):
  mDict = {}
  with open(mfn, 'r') as f:
    for line in f:
      joint, val = line.strip().split()
      mDict[int(joint)] = float(val)
  machine = []
  for joint in joints:
    machine.append(mDict[joint])


def getAcc (human, machine, mc=0.5):
  considered = 0
  correct = 0
  for i in range(len(human)):
    if machine[i] > mc:
      considered += 1
      if human[i] == 1:
        correct += 1
    elif machine[i] < (1-mc):
      considered += 1
      if human[i] == 0:
        correct += 1
  if considered == 0:
    return 1.0, 0.0
  return float(correct) / considered, float(considered) / len(human)

def getAccPerMc(human, machine):
  xs = []
  ys = []
  for mc in np.arange(0.5,1,0.01):
    xs.append(mc)
    ys.append(getAcc(human, machine, mc)[0])
  ax, = plt.plot(xs, ys, lw=2)
  return ax

def showAll(funcs, xlabel="", ylabel = "", title=""):
  vals = {}
  for flag in sorted(flags.keys()):
    vals[flag] = getVals(sh,flag)
  showSome(funcs, vals, xlabel, ylabel, title)

def showSome(funcs, vals, xlabel="", ylabel = "", title=""):
  for func in funcs:
    axs = []
    labels  = []
    for flag,val in sorted(vals.items()):
      labels.append(flag)
      axs.append(func(*val))
    plt.legend(axs, labels, loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def getUsagePerMc(human, machine):
  xs = []
  ys = []
  for mc in np.arange(0.5,1,0.01):
    xs.append(mc)
    ys.append(getAcc(human, machine, mc)[1])
  ax, = plt.plot(xs, ys, lw=2)
  return ax

def getAccPerUsage(human, machine):
  xs = []
  ys = []
  for mc in np.arange(0.5,1,0.01):
    xs.append(getAcc(human, machine, mc)[1])
    ys.append(getAcc(human, machine, mc)[0])
  ax, = plt.plot(xs, ys, lw=2)
  return ax

book = xlrd.open_workbook("raw_nn_data.xlsx")
sh = book.sheet_by_index(1)
joints = []
for i in range(2, 849):
  joints.append(int(sh.cell_value(i, 0)))

flags = {'inadcl':1, 'misal':5, 'clamp':9, 'scrape':13, 'soil':17, 'unsuit':21, 'water':25}

def printComp(flag):
  print getAcc(*getVals(sh, flag))
  print getAcc(*getVals(sh, flag, flag + '_high.log'))
  print getAcc(*getVals(sh, flag, flag + '_high.log.r'))
  print getAcc(*getVals(sh, flag, flag + '_high.log.c'))
  showSome([getAccPerMc, getUsagePerMc, getAccPerUsage], {flag + '1': getVals(sh, flag), flag + '2':getVals(sh, flag, flag + '_high.log'), flag + '3':getVals(sh, flag, flag + '_high.log.r'), flag + '4':getVals(sh, flag, flag + '_high.log.c')})

def humanStats():
  for flag in sorted(flags.keys()):
    human, _ = getVals(sh, flag)
    print flag, len(filter(lambda x:x == 1, human)), len(human)

#humanStats()

def getPos(flag):
  human, machine = getVals(sh, flag)
  for i in range(len(joints)):
    if human[i] == 1:
      print joints[i], machine[i]

#getPos('water')

human, machine = getVals(sh, 'water')
print sorted(machine)

#printComp('water')
#showAll(getAccPerMc, "Minimum confidence", "Accuracy", "Accuracy of the system per imposed minimum confidence")
#showAll(getUsagePerMc, "Minimum confidence","Proportion data classified",  "Proportion of joints classified automatically per imposed minimum confidence")
#showAll(getAccPerUsage, "Proportion data classified", "Accuracy", "Trade-off between proportion of task that is automated and task accuracy")
