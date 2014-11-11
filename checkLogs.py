import cPickle
import numpy as np

with open("scrape.log", 'r') as f:
  a = cPickle.load(f)


with open("scrape100.log", 'r') as f:
  b = cPickle.load(f)


with open("scrape500.log", 'r') as f:
  c = cPickle.load(f)


with open("scrapeCent.log", 'r') as f:
  d = cPickle.load(f)

print a[:10] 
print b[:10]
print c[:10] 
print d[:10]
