from matplotlib import pyplot as plt
import sys

conf = []
logs = ['clampLog'] 
for log in logs:
  with open(log, 'r') as f:
    for line in f:
      try:
        print line.strip().split()[-1]
        conf.append(float(line.strip().split()[-1][:-1]))
      except:
        pass

plt.hist(conf, bins=30)
plt.show()
