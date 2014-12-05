import sys

if __name__ == "__main__":
  hd = {}
  with open(sys.argv[1], 'r') as f:
    for line in f:
      print line.strip().split()  
