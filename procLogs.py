import cPickle
import sys

flags = "clamp_1flag inadcl misal scrape soil_high unsuit water_high".split()
posThresh = {"clamp_1flag" : 0.5, "inadcl" : 0.5, "misal" : 0.5, "scrape" : 0.5, "soil_high" : 0.5, "unsuit" : 0.5, "water_high" : 0.5}

def imgsToJoint(preds):
  pass

if __name__ == "__main__":
  jointMap = {}
  if len(sys.argv) > 1:
    npDir = sys.argv[2]

  with open(sys.argv[1], 'r') as f:
    for line in f:
      line = line.strip().split()
      for img in line[1:]:
        jointMap[int(img)] = int(line[0])
 
  jointPos = {}
  for flag in flags:
    pos = {}
    try:
      with open(npDir + "/" + flag + ".npy", 'r') as f:
        rezs = cPickle.load(f)
        for fn, rez in rezs:
          fNo = fn.split('.')[0].split('/')[-1]
          jNo = jointMap(int(fNo))
          try:
            pos[jNo].append(rez[1])
          except:
            pos[jNo] = [rez[1]]
    except:
      pass
    flagPos[flag] = pos

  print flagPos
