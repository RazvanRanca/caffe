import sys
import subprocess 

def getLogName(folder):
  with open("task/"+folder+"/train_val.prototxt", 'r') as f:
    for line in f:
      name = line.strip().split()[-1][1:-1]    
      break
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
  return '-'.join([folder, name, "ur:" + trainUnder, "or:"+ trainOver, "us:" + valUnder, "os:" + valOver, "vb:"+str(tBlue),  "tb:"+str(tRed),"tr:"+str(vBlue), "vr:"+str(vRed)])


if __name__ == "__main__":
  if sys.argv[1][-1] =='/':
    sys.argv[1] = sys.argv[1][:-1]
  folder = sys.argv[1].split('/')[-1]
  logName = getLogName(folder)
  command = "nohup ./build/tools/caffe train -solver task/" + folder + "/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/" + folder + "/logs/" + logName + " &"
  print "Running command:", command
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  while(True):
    retcode = process.poll() #returns None while subprocess is running
    line = process.stdout.readline()
    print line,
    if(retcode is not None):
      break
