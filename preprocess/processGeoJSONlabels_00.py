import argparse
import arcpy
import os
import re
from glob import glob
import sys
from rewriteClassIdx import checkclass_all,checkclass_single
"""
Preprocess data from Azavea. 
To run in command line, type (INCLUDING the quotation marks):
    "%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" 00_processGeoJsonLabels.py 
    followed by arguments.
        e.g.: "%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" processGeoJsonLabels_00.py -b
        e.g.: "%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" processGeoJsonLabels_00.py -m ardmayle_march14_dtm.tif -hsd Ardmayle225_15.tif -g Ardmayle.geojson 
                
If preprocess by batch, put dtm,hillshade and geojson in ~/data_folder/data/*, output will be generated in data_folder/data/*
if not preprocess by batch, output label is saved in the same directory as -d
"""


siteRegex = re.compile("Kilbixy|Fore|Glenogra|Ardmayle|Kilmacahill|Ballynahnich|Knockainey|Knockainy", re.IGNORECASE)
# get predefined IRENET95_Irish_Transverse_Mercator projection system
IRNET95_prj = arcpy.SpatialReference(2157)
# set environment coordsystem  = IRNET projection coordinate system
arcpy.env.outputCoordinateSystem = IRNET95_prj 
# set overwrite to True
arcpy.env.overwriteOutput=True 



def convertGeojsonTOfeatures( output_root, geojsonInput,siteName):
    # create Features from GeoJSON
    outputFeatures = os.path.join(output_root, f"{siteName}ShpFeatures")

    # create shape files from GEOJSON, output a .shp files an its complimentary files
    # the ouput shape features data is in IRNET95 Tranverse Mercater system already
    # because ae set output coord at the beginning
    print("\nConverting geojson to features using IRNET95 Irish Transverse Mercator projected coordinate system ...")
    outputFeatures_shp = arcpy.conversion.JSONToFeatures(
        geojsonInput, outputFeatures, "POLYGON").getOutput(0)
    print("Output .shp file: ",outputFeatures_shp)

    return outputFeatures_shp


def featureTOraster_tif (output_root,features_shp,siteName,field):
    featureRaster_tif  = os.path.join(output_root, f"{siteName}_label.tif")

    print(f"\nConverting shape file to raster using {field} ...")
    arcpy.FeatureToRaster_conversion(
        in_features=features_shp, 
        field = field,
        out_raster = featureRaster_tif,
        )
    print("Output raster .tif file: ",featureRaster_tif )

    return featureRaster_tif


def preprocess_single(output_root,inputDTM, geojsonInput_name,field):
    siteName  = re.findall(siteRegex,os.path.basename(inputDTM))[0]
    print("Preprocessing ",siteName)
    print("input DTM: ",inputDTM)

    # set arcgis workspace env before converting features to raster
    # use the DTM raster extent, cell size as the processing extent
    # use the DTM cell_size
    # this makes the output label label map to the dtm .tif pixel by pixel 
    print("Setting cell size ...")
    arcpy.env.cellSize  = inputDTM
    print("Setting output extent ..")
    arcpy.env.extent  = inputDTM
    outputFeatures_shp = convertGeojsonTOfeatures(output_root,geojsonInput_name,siteName)
    featureRaster_tif = featureTOraster_tif (output_root,outputFeatures_shp,siteName,field)
    print(f"{siteName} completed.\n\n")
    return featureRaster_tif


if __name__ ==  "__main__":
    
    desc = """
            Preprocess data from Azavea. 
            To run in command line, type (INCLUDING the quotation marks):
            "%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" 00_processGeoJsonLabels.py 
            followed by arguments.
                e.g.: "%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" processGeoJsonLabels_00.py -b
                e.g.: "%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" processGeoJsonLabels_00.py -m ardmayle_march14_dtm.tif -hsd Ardmayle225_15.tif -g Ardmayle.geojson 
                
            If preprocess by batch, put dtm,hillshade and geojson in ~/data_folder/data/*, output will be generated in data_folder/data/*
            if not preprocess by batch, output label is saved in the same directory as -d
            """
    root_path = os.path.abspath(os.path.join(os.getcwd() ,"../../"))
    default_datafolder= os.path.join(root_path,"ArcGis/Projects/HELM/data/Azavea/data")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-b','--batch', default = False, action='store_true',
                        help ="if batch processing or processing one file and one geojson. \nIf providing -b, no -m and -g arguments should be provided.")
    #parser.add_argument('-p','--project_root',nargs='?',default= default_project_root,
    #                    help="full path to the ArcGis project workspace path if not loading from ArcGIS workspace")
    parser.add_argument('-d','--data_folder',nargs="?", default= default_datafolder,
                        help="full path to Azavea folder")
    parser.add_argument('-m','--dtm',nargs="?",default ="Kilbixy_dtm.tif",
                         help = "relative path from data folder to .tif dtm of which annotations were made")
    parser.add_argument('-hsd','--hillshade',nargs="?",default ="kilbixy0_15.tif",
                         help = "relative path from data folder to .tif hillshade of which annotations were made")
    parser.add_argument('-g','--geojson',nargs="?",default = "kilbixy.geojson", 
                        help = "relative path from data folder to .geojson file storing the annotations")
    parser.add_argument('-f','--field', nargs="?",default='default', 
                        help="which field from geojson file to use to create label raster")
    try:
        args, unkown = parser.parse_known_args()
    except SystemExit:
        sys.exit("Please provide full path to data folder")
        

    batch = args.batch
  
    data_folder = args.data_folder
    print("Input data root dir:", data_folder)
    field = args.field 
    
    output_label_root = os.path.join(data_folder, "labels_shpfiles")
    os.makedirs(output_label_root, exist_ok=True) 
    print("Extracted label(s) saved in:", output_label_root, "..")

    if not batch:
        inputDTM = os.path.join(data_folder, args.dtm)
        print("input DTM", inputDTM)
        geojsonInput = os.path.join(data_folder, args.geojson)
        print("input GeoJSon", geojsonInput)
        # reproject geojson and rasterize it
        featureRaster_tif = preprocess_single(output_label_root, inputDTM, geojsonInput,field)


        hillshade = os.path.join(data_folder, args.hillshade)
        label = featureRaster_tif 
        outputlabel_folder = data_folder
        os.makedirs(outputlabel_folder,exist_ok=True)
        print("\norgainze files, map class number to 0 1 2 3...")
        print("labels are saved in ",outputlabel_folder)

        class_list = ['background','historical_walls','modern_features']
        # vsiualize, ask the user to reorder class so background = 0, historical = 1, and modern = 2
        checkclass_single(hillshade,label,outputlabel_folder,class_list)


    else:
        input_root = data_folder
        print("Processing by batch ...")
        # find all dtm tif and geojson files
        siteDTMs = glob(os.path.join(input_root,'*dtm.tif'))
        geojsonFiles_unordered  = glob(os.path.join(input_root,'*.geojson'))
        print(f"{len(siteDTMs)} are found.")


        for i in range(len(siteDTMs)):
            siteName  = re.findall(siteRegex,os.path.basename(siteDTMs[i]))[0]
            # find corresponding geojson file given a siteName
            r = re.compile(rf".*{siteName}.*",re.IGNORECASE)
            geojson_matched = list(filter(r.match, geojsonFiles_unordered)) [0]

            inputDTM = siteDTMs[i]
            geojsonInput= geojson_matched 
            featureRaster_tif = preprocess_single(output_label_root, inputDTM, geojsonInput,field)

        # orgainze files, map class number to 0 1 2 3...
        img_folder = input_root
        raw_label_folder = output_label_root
        final_output_folder = input_root
        os.makedirs(final_output_folder,exist_ok=True)
        print("\norgainze files, map class number to 0 1 2 3...")
        print("output saved in ",final_output_folder)

        class_list = ['background','historical_walls','modern_features']

        checkclass_all(img_folder,raw_label_folder,final_output_folder,class_list)
        # this functions find hillshade and its corresponding label tif, 
        # and ask the user to reorder class so background = 0, historical = 1, and modern = 2

           


"""
def reproject_inputMap(input_root, output_root, inputMap_name,siteName):
    inputfileName  = re.findall(r".*(?=\.tif)",os.path.basename(inputMap_name))[0]
    reprojectedMap_name = f"{inputfileName}IRNET95.tif"
    inputMap_tif = os.path.join(input_root, inputMap_name)
    reprojectedMap_tif = os.path.join(output_root, reprojectedMap_name)
    # get input tif cell size
    cell_sizeX = float(arcpy.GetRasterProperties_management(inputMap_tif,"CELLSIZEX").getOutput(0))
    cell_sizeY = float(arcpy.GetRasterProperties_management(inputMap_tif,"CELLSIZEY").getOutput(0))
    print("\ninput cell size: ", cell_sizeX,cell_sizeY)
    # set cellsize: keep this consistent for all output
    arcpy.env.cellSize = inputMap_tif

    # project raster using IRENET95_Irish_Transverse_Mercator system
    print("Reprojecting input raster to IRNET95 Irish Transverse Mercator projection system ...")
    arcpy.ProjectRaster_management(inputMap_tif, reprojectedMap_tif,
                                out_coor_system = IRNET95_prj,
                                resampling_type = "BILINEAR")
    print("Output .tif file: ",reprojectedMap_tif)


    cell_sizeX = float(arcpy.GetRasterProperties_management(reprojectedMap_tif,"CELLSIZEX").getOutput(0))
    cell_sizeY = float(arcpy.GetRasterProperties_management(reprojectedMap_tif,"CELLSIZEY").getOutput(0))
    print("check cell size:", cell_sizeX,cell_sizeY)

    return reprojectedMap_tif

def stretch_map(input_root,output_root,inputMap_name,siteName):
    inputfileName  = re.findall(r".+IRNET95(?=\.tif)",os.path.basename(inputMap_name))[0]
    stretchMap_name = f"{inputfileName}_stretch.tif"
    inputMap_tif = os.path.join(input_root, inputMap_name)
    stretchMap_tif = os.path.join(output_root, stretchMap_name)

"""
