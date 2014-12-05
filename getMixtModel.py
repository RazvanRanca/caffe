def readLog(fn):
  vals = {}
  with open(fn, 'r') as f:
    for line in f:
      flag, val = line.strip().split()
      vals[int(flag)] = float(val)
  return vals

def combModels(fn1, fn2):
  vals1, vals2 = readLog(fn1), readLog(fn2)
  rez = {}
  for k in vals1.keys():
    rez[k] = (vals1[k] + vals2[k]) / 2.0
  return rez

def writeLog(vals, fn):
  with open(fn, 'w') as f:
    first = True
    for k,v in sorted(vals.items()):
      if first:
        first = False
      else:
        f.write("\n")
      f.write(str(k) + " " + str(v))

writeLog(combModels("water_high.log", "water_high.log.r"), "water_high.log.c")
writeLog(combModels("soil_high.log", "soil_high.log.r"), "soil_high.log.c")
