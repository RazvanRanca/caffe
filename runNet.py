#!/usr/bin/env python
import sys
import subprocess 
import os
import shutil

def getDataInfo(folder):
  with open("data/"+folder+"/val.txt", 'r') as f:
    zeros = 0.0
    ones = 0.0
    vRed = 0
    vBlue = 0
    uniqOnes = set()
    for line in f:
      if line.strip() == "":
        continue
      if "Redbox" in line:
        vRed += 1
      elif "Bluebox" in line:
        vBlue += 1
      else:
        raise Exception("Not Bluebox/Redbox in line: " + line)
      if line.strip().endswith("0"):
        zeros += 1
      elif line.strip().endswith("1"):
        ones += 1
        uniqOnes.add(line.strip().split()[0])
      else:
        raise Exception("Unrecognized flag value in line: " + line)

  valUnder = str(1 - len(uniqOnes)/(len(uniqOnes)+zeros))[:4]
  valOver = str(1 - ones / (ones +zeros))[:4]
  
  with open("data/"+folder+"/train.txt", 'r') as f:
    zeros = 0.0
    uniqOnes = set()
    ones = 0.0
    tRed = 0
    tBlue = 0
    for line in f:
      if line.strip() == "":
        continue
      if "Redbox" in line:
        tRed += 1
      elif "Bluebox" in line:
        tBlue += 1
      else:
        raise Exception("Not Bluebox/Redbox in line: " + line)
      if line.strip().endswith("0"):
        zeros += 1
      elif line.strip().endswith("1"):
        ones += 1
        uniqOnes.add(line.strip().split()[0])
      else:
        raise Exception("Unrecognized flag value in line: " + line)
   
  trainUnder = str( 1 - len(uniqOnes)/(len(uniqOnes)+zeros))[:4]
  trainOver = str(1 - ones / (ones +zeros))[:4]
  return {"folder": folder, "trainUnder":trainUnder, "trainOver":trainOver, "valUnder":valUnder, "valOver":valOver, "tBlue":str(tBlue),  "tRed":str(tRed),"vBlue":str(vBlue), "vRed":str(vRed)}

def getFoldName(logFold, baseName="tempLog"):
  ind = 0
  curName = logFold + "/" + baseName + str(ind)
  while os.path.exists(curName):
    ind += 1
    curName = logFold + "/" + baseName + str(ind)
  return curName

def saveRelInfo(folder, solver, tradeoff, pruneSnaps):
  logFold = "task/" + folder + "/logs/"
  if not os.path.isdir(logFold):
    os.makedirs(logFold)
  foldName = getFoldName(logFold)
  assert(not os.path.isdir(foldName))
  dataDict = getDataInfo(folder)
  os.makedirs(foldName)
  shutil.copy("task/"+folder+"/" + solver, foldName)
  with open(foldName + "/" + solver, 'r') as f:
    for line in f:
      if "net:" in line:
        net = line.strip().split()[-1][1:-1].split('/')[-1]
    
  shutil.copy("task/"+folder+"/" + net, foldName)
  with open(foldName + "/dataInfo", 'w') as f:
    f.write('\n'.join(map(str, sorted(getDataInfo(folder).items() + [("tradeoff", tradeoff),("pruneSnaps", pruneSnaps)])))) 
  return foldName


def help():
  print '''usage: e.g 
  ./runNet.py task/water_high [solver = solver.prototxt] [neg/pos tradeoff = 1.0] [pruneSnaps = True]
  (takes solver.prototxt in indicated folder)
    OR
   ./runNet.py taskFile
   (taskFile can contain multiple rows of arguments resulting in consecutive training runs)'''
  sys.exit()


def runNet(args):
  try:
    solver = args[1]
  except:
    solver = "solver.prototxt"

  try:
    tradeoff = float(args[2])
  except:
    tradeoff = 1.0

  try:
    pruneSnaps = bool(args[3])
  except:
    pruneSnaps = True
  
  folder = args[0].strip('/').split('/')[-1]
  foldName = saveRelInfo(folder, solver, tradeoff, pruneSnaps)
  command = "nohup ./build/tools/caffe train -solver task/" + folder + "/" + solver + " -weights oxford/small.weights 2>&1 | tee " + foldName  + "/train.log"
  print "Running command:", command
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  posAcc = 0
  negAcc = 0
  bestScore = (0,0,0) #score, neg, pos
  bestSnap = None
  delSnap = None
  lastSnap = None
  while(True):
    if pruneSnaps and delSnap:
      print "Deleting", delSnap
      os.remove(delSnap + ".caffemodel")
      os.remove(delSnap + ".solverstate")
      delSnap = None
    retcode = process.poll() #returns None while subprocess is running
    line = process.stdout.readline()
    if "Test net output #0" in line:
      negAcc = float(line.strip().split()[-1])
    if "Test net output #1" in line:
      posAcc = float(line.strip().split()[-1])
      score = (negAcc*tradeoff + posAcc) / 2.0
      if score > bestScore[0]:
	bestScore = (score, negAcc, posAcc)
	if lastSnap != bestSnap:
	  delSnap = bestSnap
	  bestSnap = lastSnap
    if "Snapshotting solver" in line:
      snapshot = line.strip().split()[-1].split('.')[0]
      if lastSnap != bestSnap:
	delSnap = lastSnap
      lastSnap = snapshot
    print line,
    if(retcode is not None):
      print retcode
      break
  finalName = getFoldName('/'.join(foldName.strip('/').split('/')[:-1]), str(bestScore[1])[:4] + "-" + str(bestScore[2])[:4] + "-")
  if bestSnap:
    shutil.move(bestSnap + ".caffemodel", foldName)
    shutil.move(bestSnap + ".solverstate", foldName)
  if lastSnap and lastSnap != bestSnap:
    shutil.move(lastSnap + ".caffemodel", foldName)
    shutil.move(lastSnap + ".solverstate", foldName)
  os.rename(foldName, finalName)

if __name__ == "__main__":
  if len(sys.argv) == 1:
    help()
    sys.exit()

  if os.path.isfile(sys.argv[1]):
    with open(sys.argv[1], 'r') as f:
      for line in f:
        runNet(line.strip().split())
  else:
    runNet(sys.argv[1:])

