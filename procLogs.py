import cPickle
import sys

flags = "clamp_1flag inadcl misal soil_high scrape unsuit water_high".split()
posThresh = {"clamp_1flag" : 0.5, "inadcl" : 0.5, "misal" : 0.5, "scrape" : 0.5, "soil_high" : 0.5, "unsuit" : 0.5, "water_high" : 0.5}

def imgsToJoint(flag, preds):
  avg = sum(preds) / float(len(preds)) 
  if avg > posThresh[flag]:
    return 1
  return 0

if __name__ == "__main__":
  if len(sys.argv) != 4:
    raise Exception("Need multJoints,log folder and output file as parameters")
  jointMap = {}
  npDir = sys.argv[2]

  with open(sys.argv[1], 'r') as f:
    for line in f:
      line = line.strip().split()
      for img in line[1:]:
        jointMap[int(img)] = int(line[0])
 
  jointPos = {}
  joints = set() 
  for flag in flags:
    pos = {}
    with open(npDir + "/" + flag + ".npy", 'r') as f:
      rezs = cPickle.load(f)
      for fn, rez in rezs:
	fNo = fn.split('.')[0].split('/')[-1]
	jNo = jointMap[int(fNo)]
        joints.add(jNo)
	try:
	  pos[jNo].append(rez[1])
	except:
	  pos[jNo] = [rez[1]]
    jointPos[flag] = pos

  with open(sys.argv[3], 'w') as f:
    out = "Joint, " + ', '.join(flags)
    print out
    f.write(out + "\n")
    for joint in sorted(list(joints)):
      out = str(joint)
      for flag in flags:
	out += ", " + str(imgsToJoint(flag, jointPos[flag][joint]))
      print out
      f.write(out + "\n")
 
