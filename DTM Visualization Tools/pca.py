import arcpy
from arcpy import env
import tkinter, tkinter.filedialog
import os
from os.path import exists
root = tkinter.Tk()

#Users select folder where PCA file will be stored
dirname = tkinter.filedialog.askdirectory(parent=root, initialdir="/",title="Please Select a Workspace")
env.workspace=dirname
env.overwriteOutput = True

#User selects the rasters they want to use for PCA and the xml files are stored in a list
lst=[]
rasters = tkinter.filedialog.askopenfilenames(parent=root, initialdir=dirname, title='Choose the Hillshade Files for PCA', filetypes=(("XML File","*.xml"), ("all files","*.*")))
lst = list(rasters)
lst2 = [x.replace('.aux.xml','') for x in lst]

print(lst2)
inrasters="';'".join(lst2)
inrasters = "'"+inrasters+"'"
print(inrasters)

root.destroy()

#Create a window
root=tkinter.Tk()
root.geometry('400x100')

principalCompsInput = tkinter.Text(root,height = 2,width = 2)
principalCompsInput.grid(row=0, column=1)
prinCompLabel = tkinter.Label(root, text = "Number of Principal Components:")
prinCompLabel.grid(row=0, column=0)

#initialize principal components
principalComps = 0

#function that runs PCA for desired files
def prinComps(number):
    output = makeOutFile
    out_multiband_raster = arcpy.sa.PrincipalComponents(inrasters, number, None); out_multiband_raster.save(output)

def makeOutFile():
    outFile= dirname + "/PrinComps" + str(principalComps)
    i=1
    checker=True
    while(checker):
        if(os.path.isfile(dirname+outFile)!=True):
            return outFile + "_file" + str(i)
            break
        i+=1

def takeInput():
    principalComps=principalCompsInput.get("1.0",'end-1c')
    principalComps = int(principalComps) 
    if(principalComps>len(lst)):
        principalComps=len(lst)
    root.destroy()
    prinComps(principalComps)

#Button is created that collects values and runs PCA when clicked
takeInputButton = tkinter.Button(root,text = "OK", command = takeInput)
takeInputButton.grid(row=4,column=2)

root.mainloop()