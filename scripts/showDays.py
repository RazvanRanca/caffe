from matplotlib import pyplot as plt

bCounts = {} 
with open("blueboxDays", 'r') as f:
  for line in f:
    count, day = line.strip().split()
    bCounts[day] = int(count)

rCounts = {}
with open("redboxDays", 'r') as f:
  for line in f:
    count, day = line.strip().split()
    rCounts[day] = int(count)

def dateTrans(x):
  d,m,y = map(int, x.strip().split('/'))
  return y * 365 + m * 12 + d

keys = sorted(bCounts.keys() + rCounts.keys(), key = lambda x: dateTrans(x))

bArr = []
rArr = []
tArr = []
for key in keys:
  try:
    bArr.append(bCounts[key])
  except:
    bArr.append(0)
  try:
    rArr.append(rCounts[key])
  except:
    rArr.append(0)
  tArr.append(bArr[-1] + rArr[-1])

plt.plot(bArr, 'b')
plt.plot(rArr, 'r')
plt.plot(tArr, 'g')
plt.show()
