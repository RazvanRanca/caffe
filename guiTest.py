from Tkinter import *
from tkFileDialog import askopenfilename, askdirectory
import subprocess
from PIL import Image, ImageTk
import os

def sel1():
   if runLoc.get() == "server":
     errLabel2.pack_forget()
   elif runType.get() == "batch":
     errLabel2.pack()
     errLabel2.config(fg="orange", text = "Warning: Batch mode slow on CPU")

def sel2():
   if runType.get() == "batch" and runLoc.get() == "local":
     errLabel2.pack()
     errLabel2.config(fg="orange", text = "Warning: Batch mode slow on CPU")
   else:
     errLabel2.pack_forget()

model = None
def sel3():
  global model
  if not os.path.isfile("oxford/" + var3.get() + ".caffemodel"):
     model = None
     errLabel3.pack()
     errLabel3.config(text = var3.get() + " classifier not currently available")
  else:
     errLabel3.pack_forget()
     model = "oxford/" + var3.get() + ".caffemodel"

filename = None
def button1Click():
  global filename
  if runType.get() == "online":
    filename = askopenfilename()
    image = Image.open(filename)
    w, h = image.size
    if h > w:
      image = image.rotate(90)
      w, h = image.size
    image = image.resize((w/2,h/2),Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    imgLabel.config(image=photo)
    disp.config(text='')
    imgLabel.image = photo # keep a reference!
    errLabel4.pack_forget()
  elif runType.get() == "batch":
    filename = askdirectory() 

def button2Click():
  print filename, model
  if filename and model:
    disp.config(text = "Processing ...")
    frame2.update_idletasks()
    
    if runLoc.get() == "local":
      cmd = "python classifyPipe.py --pretrained_model " + model + " " + filename + " temp"
    else:
      cmd = "./remoteClassify.sh"
    p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # and you can block util the cmd execute finish
    stdout, stderr = p.communicate()
    print stderr
    print stdout
    stdout = stdout.split("-=-=-START-=-=-")[-1].split("-=-=-END-=-=-")[0].strip()
    disp.config(text = stdout)
  elif filename == None:
    errLabel4.pack()
    errLabel4.config(text = "No files loaded")

root = Tk()
errs = Frame(root)
errs.pack(anchor=W)
errLabel1 = Label(errs, fg="red")
errLabel2 = Label(errs, fg="red")
errLabel3 = Label(errs, fg="red")
errLabel4 = Label(errs, fg="red")
frame1 = Frame(root)
frame1.pack(side=LEFT)
frame2 = Frame(root)
frame2.pack(side=LEFT)
disp = Label(frame2)
disp.pack()
imgLabel = Label(frame2)
imgLabel.pack()

box1 = Frame(frame1, bd=3, relief=RIDGE, pady=5)
box1.pack( anchor=W)
runLoc = StringVar()

R1 = Radiobutton(box1, text="Run remotely", variable=runLoc, value="server", command=sel1)
R1.pack( anchor = W )
R1.invoke()

R2 = Radiobutton(box1, text="Run locally", variable=runLoc, value="local", command=sel1)
R2.pack( anchor = W )

box2 = Frame(frame1, bd=3, relief=RIDGE, pady=5)
box2.pack(anchor = W)
runType = StringVar()

R3 = Radiobutton(box2, text="Run online", variable=runType, value="online", command=sel2)
R3.pack( anchor = W )
R3.invoke()

R4 = Radiobutton(box2, text="Run batch", variable=runType, value="batch", command=sel2)
R4.pack( anchor = W )

box3 = Frame(frame1, bd=3, relief=RIDGE, pady=5)
box3.pack(anchor = W)
var3 = StringVar()

R5 = Radiobutton(box3, text="Test no clamp", variable=var3, value="clamp", command=sel3)
R5.pack( anchor = W )
R5.invoke()

R6 = Radiobutton(box3, text="Test no scrape", variable=var3, value="scrape", command=sel3)
R6.pack( anchor = W )

R7 = Radiobutton(box3, text="Test water contamination", variable=var3, value="water", command=sel3)
R7.pack( anchor = W )

R8 = Radiobutton(box3, text="Test soil contamination", variable=var3, value="soil", command=sel3)
R8.pack( anchor = W )

R9 = Radiobutton(box3, text="Test inadequate clamp", variable=var3, value="inadcl", command=sel3)
R9.pack( anchor = W )

R10 = Radiobutton(box3, text="Test joint misaligned", variable=var3, value="misal", command=sel3)
R10.pack( anchor = W )

R11 = Radiobutton(box3, text="Test unsuitable photo", variable=var3, value="unsuit", command=sel3)
R11.pack( anchor = W )

box4 = Frame(frame1, pady=10)
box4.pack(anchor = W)
button1 = Button(box4, text="Select file(s)",  command=button1Click)
button1.pack(anchor=W, side=LEFT)
button2 = Button(box4, text="Run classifier",  command=button2Click)
button2.pack(anchor=W, side=LEFT)

root.mainloop()
