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
    showImg()
    shutil.copy(curFilename, folder)
    print curFilename, folder

def makeButton(folders):
  buttons = []
  for folder in folders:
    if not "/" in folder:
      folder = sys.argv[2] + "/" + folder
    if not os.path.exists(folder):
      os.makedirs(folder)

    buttons.append(Button(box2, text=folder.strip('/').split('/')[-1],  command=lambda a=folder: buttonClick(a)))
    buttons[-1].pack(anchor=N)

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

with open(sys.argv[1], 'r') as f:
  rows = f.read().strip().split('\n')
rowInd = 0
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

makeButton(["f1", "f2", "f3"])
showImg()

root.mainloop()
