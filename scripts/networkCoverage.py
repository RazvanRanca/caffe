import sys
import os
import time
import cPickle

if __name__ == "__main__":
  dur = {}
  for fl in filter(lambda x:x.endswith(".met"), os.listdir(sys.argv[1])):
    with open(sys.argv[1] + "/" + fl, 'r') as f:
      for line in f:
        if line.startswith("CreatedTime"):
          c = time.strptime(line.split("=")[1].strip(), "%d/%m/%Y %H:%M:%S") 
        if line.startswith("UploadedTime"):
          u = time.strptime(line.split("=")[1].strip(), "%d/%m/%Y %H:%M:%S")
          dur[fl] = time.mktime(u) - time.mktime(c)

  with open("coverage.log", 'w') as f:
    cPickle.dump(dur, f) 
