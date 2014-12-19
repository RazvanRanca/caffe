from Tkinter import *
from tkFileDialog import askopenfilename, askdirectory
import shutil
from PIL import Image, ImageTk
import os
import sys

curFilename = None
def showImg():
  global rowInd
  global curFilename
  global canvas
  global zoomcycle
  global zimg_id
  global orig_img
  global img
  if rowInd == len(rows):
    disp.config(text="All done")
    return
  curFilename = sys.argv[2] + "/" + rows[rowInd].strip().split()[0] + ".jpg"
  disp.config(text=' - '.join(rows[rowInd].strip().split()))
  rowInd += 1
  orig_img = Image.open(curFilename)
  w, h = orig_img.size
  if h > w:
    orig_img = orig_img.rotate(90)
    w, h = orig_img.size

  minScale = max(1, max(w/float(cw), h/float(ch)))
  print w, cw, w/cw
  print h, ch, h/ch
  print minScale
  orig_img = orig_img.resize((int(w/minScale),int(h/minScale)),Image.ANTIALIAS)
  img = ImageTk.PhotoImage(orig_img)
  canvas.create_image(0,0,image=img, anchor="nw")

  zoomcycle = 0
  zimg_id = None

  root.bind("<ButtonPress>", zoomer)
  canvas.bind("<Motion>", crop)
  frame2.update_idletasks()

def buttonClick(folder):
    shutil.copy(curFilename, folder)
    print curFilename, folder
    with open(logFile, 'a') as f:
      f.write(curFilename + " " + ' '.join(rows[rowInd-1].strip().split()) + " " + foldKey[folder.strip('/').split('/')[-1]] + "\n")
    showImg()

foldKey = {}
def makeButton(folders):
  global foldKey
  buttons = []
  for folder in folders:
    foldKey[folder] = str(len(foldKey))
    if not "/" in folder:
      folder = sys.argv[2] + "/" + folder
    if not os.path.exists(folder):
      os.makedirs(folder)

    buttons.append(Button(box2, text=folder.strip('/').split('/')[-1],  command=lambda a=folder: buttonClick(a)))
    buttons[-1].pack(anchor=N)
  with open(logFile, 'a') as f:
    f.write("#" + str(sorted(foldKey.items(), key=lambda (x,y):y)) + "\n")

zoomcycle = 0
def zoomer(event):
  global zoomcycle
  if (event.num == 1):
    zoomcycle = (zoomcycle + 1) % 5
  elif (event.num == 3):
    zoomcycle = (zoomcycle - 1) % 5
  crop(event)

def crop(event):
  global zimg_id
  global zimg
  if zimg_id: canvas.delete(zimg_id)
  if (zoomcycle) != 0:
    x,y = event.x, event.y
    if zoomcycle == 1:
      tmp = orig_img.crop((x-120,y-80,x+120,y+80))
    elif zoomcycle == 2:
      tmp = orig_img.crop((x-75,y-50,x+75,y+50))
    elif zoomcycle == 3:
      tmp = orig_img.crop((x-45,y-30,x+45,y+30))
    elif zoomcycle == 4:
      tmp = orig_img.crop((x-30,y-20,x+30,y+20))
    elif zoomcycle == 5:
      tmp = orig_img.crop((x-15,y-10,x+15,y+10))
    size = 600,400
    zimg = ImageTk.PhotoImage(tmp.resize(size))
    zimg_id = canvas.create_image(event.x,event.y,image=zimg)

if len(sys.argv) == 1: 
  print '''usage: e.g.
  python guiMisclass.py scripts/scrape_blue_rezs/listSingNeg ~/data/pipe-data/scrape_blue_rezs logFile 168
  (listSingNeg is probs for each neg image)
  (*_rezs is images)
  (logFile is the file to which the results summary is saved)
  (168 is row index in listSingNeg to start from. If non-zero will append to logFile [default=0])'''
  sys.exit()

with open(sys.argv[1], 'r') as f:
  rows = f.read().strip().split('\n')

logFile = sys.argv[3]

try:
  rowInd = int(sys.argv[4])
except:
  rowInd = 0
  print "Couldn't read rowInd, assuming 0"

if rowInd == 0 and os.path.exists(logFile):
  os.remove(logFile)

root = Tk()

frame1 = Frame(root)
frame1.pack(side=LEFT)
frame2 = Frame(root)
disp = Label(frame2, font=("Times",24))
disp.pack()
cw = 1000
ch = 600
canvas = Canvas(frame2,width=cw,height=ch)
canvas.pack()
frame2.pack(side=LEFT)

box2 = Frame(frame1, pady=10)
box2.pack(anchor = W)

makeButton(["weFuckedUp", "theyFuckedUp", "unclear", "unsuit"])
showImg()

root.mainloop()
