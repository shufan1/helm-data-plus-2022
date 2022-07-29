#from tkinter import X
from PIL import Image
import os
import re
from glob import glob
# import tkinter,tkinter.filedialog
# root=tkinter.Tk()

#dirname=tkinter.filedialog.askdirectory(parent=root, initialdir="/",title="Please select a workspace with cropped images")
def reconstruct(dirname):
    directory_list = glob(os.path.join(dirname,"*.png"))
    r= re.compile(".*[0-9]+_[0-9]+.*")
    directory_list  = list(filter(r.search, directory_list ))

    xCoordList=list()
    yCoordList=list()
    dictionary = dict()
    for f in directory_list:
        splitString=os.path.basename(f).split("_")
        xCoord,yCoord="",""
        matchX = re.match(r"([a-z]+)([0-9]+)",splitString[0])
        if matchX:
                items = matchX.groups()
                locationName=items[0]
                xCoord=items[1]
        matchY = re.match(r"([0-9]+).([a-z]+)",splitString[1])
        if matchY:
                items = matchY.groups()
                yCoord=items[0]
        if(xCoord not in dictionary.keys()):
            dictionary[xCoord]=dict()
        # save image in corresponding x y coordinate
        dictionary[xCoord][yCoord]=Image.open(os.path.join(dirname,f))
        # store unique X y coord combo
        if int(xCoord) not in xCoordList:
            xCoordList.append(int(xCoord))
        if int(yCoord) not in yCoordList:
            yCoordList.append(int(yCoord))

    xCoordList.sort()
    yCoordList.sort()

    maxWidth=0
    maxHeight=0

    for i in yCoordList:
        maxWidth=maxWidth+dictionary["0"][str(i)].width
    for i in xCoordList:
        maxHeight=maxHeight+dictionary[str(i)]["0"].height
    lastHeight=0
    lastUsedWidth=0

    outputFinal=Image.new('L',(maxWidth,maxHeight))
    for xCoord in xCoordList:
        output=Image.new('L',(maxWidth,dictionary[str(xCoord)]["0"].height))
        for yCoord in yCoordList:
            #print(str(lastUsedWidth)+"-"+str(lastHeight))
            output.paste(dictionary[str(xCoord)][str(yCoord)],(lastUsedWidth,0))
            lastUsedWidth+=dictionary[str(xCoord)][str(yCoord)].width

        outputFinal.paste(output,(0,lastHeight))
        lastHeight+=dictionary[str(xCoord)]["0"].height
        lastUsedWidth=0
    
    return outputFinal

