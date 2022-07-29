import arcpy
from arcpy import env
import tkinter, tkinter.filedialog
import os
import arcgis
import re
root = tkinter.Tk()
#Select the directory where you want the outuput folder to be located
dirname = tkinter.filedialog.askdirectory(parent=root, initialdir="/",title="Please select a workspace")

env.workspace=dirname
env.overwriteOutput = True

root.destroy()
# A Method for creating a Folder where all the individual subfolders for each pair of settings will be located, has logic
# built in to avoid name duplication of the folders
def makeSaveDirectory(i):
    outRaster="OutputImages"
    try:
        if i==0:
         os.mkdir(os.path.join(dirname,outRaster))
        else: 
            os.mkdir(os.path.join(dirname,outRaster+"("+str(i)+")")) 
            outRaster=outRaster+"("+str(i)+")"
        return outRaster
    except:
        return makeSaveDirectory(i+1)

globalOutFolder=makeSaveDirectory(0)

directory_list = list()
for r, dirs, files in os.walk(dirname, topdown=False):
    for name in dirs:
        #Filter the filenames according to current naming convensinos(Fore20_20) for generated raster files, make sure there are no duplicates
        if re.match("[a-z]*[0-9]*_[0-9]*",name) and not (name in directory_list):
            currentRasterPath=os.path.join(r, name)
            directory_list.append(currentRasterPath)
            testRaster=arcpy.Raster(currentRasterPath)
            arcpy.CopyRaster_management(testRaster,r""+"/"+globalOutFolder+"/"+name+".jpg","DEFAULTS","0","9","","","8_BIT_UNSIGNED")



