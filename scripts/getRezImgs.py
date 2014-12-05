import sys
import random

def getRez(md, hd, sort, mlen):
  dd = {}
  for joint, val in md.items():
    dd[joint] = abs(val - hd[joint])

  if sort == "high":
    return sorted(dd.items(), key=lambda x: x[1], reverse=True)[:mlen]
  if sort == "low":
    return sorted(dd.items(), key=lambda x: x[1])[:mlen]
  if sort == "rand":
    ddi = dd.items()
    random.shuffle(ddi)
    return ddi[:mlen]

def dispRez(md, hd, dd):
  for rez in dd:
    print rez[0], hd[rez[0]], str(md[rez[0]])[:4], rez[1]

if __name__ == "__main__":
  task = sys.argv[1]
  model = sys.argv[2]
  mDict = {}
  with open("../task/" + task + "/none/" + model + ".log", 'r') as f:
    for line in f:
      line = line.strip().split()
      mDict[int(line[0])] = float(line[1])

  hDict = {}
  with open("../data/" + task + "/val.txt", 'r') as f:
    for line in f:
      line = line.strip().split()
      hDict[int(line[0].split('/')[-1].split('.')[0])] = float(line[1])


  try:
    sort = sys.argv[3]
  except:
    sort = "high"

  try:
    mlen = int(sys.argv[4])
  except:
    mlen = len(mDict.items()) 

  dispRez(mDict, hDict, getRez(mDict, hDict, sort, mlen))
