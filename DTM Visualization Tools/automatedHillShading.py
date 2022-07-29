import arcpy
from arcpy import env
import tkinter, tkinter.filedialog
import os
root = tkinter.Tk()

#Select the directory where you want the outuput folder to be located
dirname = tkinter.filedialog.askdirectory(parent=root, initialdir="/",title="Please select a workspace")

env.workspace=dirname
env.overwriteOutput = True

path=dirname.split("/")
#Get the name of the location, usually the rast part of the directory path
locationName=path[len(path)-1]

#Select the input
inRaster= tkinter.filedialog.askopenfilename(parent=root, initialdir=dirname,title="Please select input raster",filetypes=(("TIF File","*.tif"), ("all files","*.*")))
root.destroy()

#launch an input window
root=tkinter.Tk()
root.geometry('600x200')

#THE BELOW CODE FROM LINES 25-58 BUILDS THE INPUT FIELDS FOR EACH VARIABLE
minAzimuthInput = tkinter.Text(root,height = 2,width = 5)
minAzimuthInput.grid(row=0, column=1)
minAzimuthLabel = tkinter.Label(root, text = "Min Azimuth:")
minAzimuthLabel.grid(row=0, column=0)


maxAzimuthInput = tkinter.Text(root,height = 2,width = 5)
maxAzimuthInput.grid(row=0, column=3)
maxAzimuthLabel = tkinter.Label(root, text = "Max Azimuth:")
maxAzimuthLabel.grid(row=0, column=2)

azimuthIncrementInput = tkinter.Text(root,height = 2,width = 5)
azimuthIncrementInput.grid(row=0, column=5)
azimuthIncrementLabel = tkinter.Label(root, text = "Azimuth Increment:")
azimuthIncrementLabel.grid(row=0, column=4)


minAltitudeInput = tkinter.Text(root,height = 2,width = 5)
minAltitudeInput.grid(row=1, column=1)
minAltitudeLabel = tkinter.Label(root, text = "Min Altitude:")
minAltitudeLabel.grid(row=1, column=0)


maxAltitudeInput = tkinter.Text(root,height = 2,width = 5)
maxAltitudeInput.grid(row=1, column=3)
maxAltitudeLabel = tkinter.Label(root, text = "Max Altitude:")
maxAltitudeLabel.grid(row=1, column=2)

altitudeIncrementInput = tkinter.Text(root,height = 2,width = 5)
altitudeIncrementInput.grid(row=1, column=5)
altitudeIncrementLabel = tkinter.Label(root, text = "Altitude Increment:")
altitudeIncrementLabel.grid(row=1, column=4)

locationInput = tkinter.Text(root,height = 2,width =7)
locationInput.grid(row=2, column=2)
locationInput.insert(1.0,locationName)
locationLabel = tkinter.Label(root, text = "Location Name:")
locationLabel.grid(row=2, column=1)


#INITIALIZING RELEVANT VARIBALES FOR AZIMUTH AND ALTITUDE SETTINGS
minAzimuth=0
maxAzimuth=360
azimuthIncrement=45

minAltitude=5
maxAltitude=45
altitudeIncrement=5




# A Method for creating a Folder where all the individual subfolders for each pair of settings will be located, has logic
# built in to avoid name duplication of the folders
def makeSaveDirectory(i):
    outRaster="Output"
    try:
        if i==0:
         os.mkdir(os.path.join(dirname,outRaster))
        else: 
            os.mkdir(os.path.join(dirname,outRaster+"("+str(i)+")")) 
            outRaster=outRaster+"("+str(i)+")"
        return outRaster
    except:
        return makeSaveDirectory(i+1)

globalOutRaster=makeSaveDirectory(0)

#CREATE A SMALL WINDOW WHICH displays the progress
def displayProgress(progressRoot,totalAzimuthOptions, azimuthIndex, totalAltitudeOptions, index):
    
    
    progressLabel = tkinter.Label(progressRoot, text = "Generating:" + str((totalAltitudeOptions*(azimuthIndex-1))+index) + "/"+str(totalAltitudeOptions*totalAzimuthOptions))
    progressLabel.pack()
    print("Generating:" + str((totalAltitudeOptions*(azimuthIndex-1))+index) + "/"+str(totalAltitudeOptions*totalAzimuthOptions))
    
    


# A loop which goes through all the altitude values in the given range with the given inceremnt. It runs the hillshade method for each pair.
def cycleAltitude(azimuth,minAltitude,maxAltitude,altitudeIncrement,totalAzimuthOptions,azimuthIndex,location):
    totalAltitudeOptions=int(maxAltitude/altitudeIncrement)
    
    print("totalAltitudeOptions"+str(totalAltitudeOptions))
    index=1
    for altitudeVariable in range(minAltitude,maxAltitude+1,altitudeIncrement):
        progressRoot=tkinter.Tk()
        progressRoot.geometry('200x50')
        displayProgress(progressRoot,totalAzimuthOptions, azimuthIndex, totalAltitudeOptions, index)
        progressRoot.update_idletasks()
        progressRoot.update()
        arcpy.ddd.HillShade(inRaster, str(globalOutRaster)+"/"+location+str(azimuth)+"_"+str(altitudeVariable), azimuth, altitudeVariable, "SHADOWS", 1)
        progressRoot.destroy()
        index=index+1

#A loop which goes through all the azimuth values in the given range with the given inceremnt. 
# Matches each value with all possible altitude values.
def cycleAzimuth(minAzimuth,maxAzimuth,azimuthIncrement,minAltitude,maxAltitude,altitudeIncrement,location):
    totalAzimuthOptions=int(maxAzimuth/azimuthIncrement)
    index=1
    for azimuthVariable in range(minAzimuth,maxAzimuth,azimuthIncrement):
        cycleAltitude(azimuthVariable,minAltitude,maxAltitude,altitudeIncrement,totalAzimuthOptions,index,location)
        index=index+1




#When the "Save Input" button is pressed, the values in each cell are collected and the loops are run to generate all output rasters.
def takeInput():
    minAzimuth=minAzimuthInput.get("1.0",'end-1c')
    maxAzimuth=maxAzimuthInput.get("1.0",'end-1c')
    azimuthIncrement=azimuthIncrementInput.get("1.0",'end-1c')

    minAltitude=minAltitudeInput.get("1.0",'end-1c')
    maxAltitude=maxAltitudeInput.get("1.0",'end-1c')
    altitudeIncrement=altitudeIncrementInput.get("1.0",'end-1c')

    location=locationInput.get("1.0",'end-1c')
    location=location[:7]
    print(location)

    root.destroy()
    
    cycleAzimuth(int(minAzimuth),int(maxAzimuth),int(azimuthIncrement),int(minAltitude),int(maxAltitude),int(altitudeIncrement),location)

takeInputButton = tkinter.Button(root,
                        text = "Save Input", 
                        command = takeInput)
takeInputButton.grid(row=2,column=3)





root.mainloop()