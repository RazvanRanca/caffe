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

def getLogName(logFold, baseName="tempLog"):
  ind = 0
  curName = logFold + "/" + baseName + str(ind)
  while os.path.exists(curName):
    ind += 1
    curName = logFold + "/" + baseName + str(ind)
  return curName

def saveRelInfo(folder):
  logFold = "task/" + folder + "/logs/"
  if not os.path.isdir(logFold):
    os.makedirs(logFold)
  logName = getLogName(logFold)
  assert(not os.path.isdir(logName))
  dataDict = getDataInfo(folder)
  os.makedirs(logName)
  shutil.copy("task/"+folder+"/solver.prototxt", logName)
  shutil.copy("task/"+folder+"/train_val.prototxt", logName)
  with open(logName + "/dataInfo", 'w') as f:
    f.write('\n'.join(map(str, sorted(getDataInfo(folder).items())))) 
  return logName + "/train.log"

if __name__ == "__main__":
  for d in range(1, len(sys.argv)):
    if sys.argv[d][-1] =='/':
      sys.argv[d] = sys.argv[d][:-1]
    folder = sys.argv[d].split('/')[-1]
    logName = saveRelInfo(folder)
    command = "nohup ./build/tools/caffe train -solver task/" + folder + "/solver.prototxt -weights oxford/small.weights 2>&1 | tee " + logName 
    print "Running command:", command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    while(True):
      retcode = process.poll() #returns None while subprocess is running
      line = process.stdout.readline()
      print line,
      if(retcode is not None):
        print retcode
	break
