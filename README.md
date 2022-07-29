# HELM Data+ Project
### Overview
The HELM project aims detect archaeological features of medieval Irish settlements using aerial imagery within the modern Irish landscape. The project can be broken down into two main stages: using visualization techniques to enhance present features and developing a method for automatic feature detection.

Using aerial imagery via drones is cost-effective and efficient compared to traditional archaeological fieldwork. We used images of locations across Ireland, reconstructed as digital terrain models (DTMs), and visualized them to locate low-lying medieval features in the landscape such as burgage plots and ridge and furrows. We subsequently implemented and trained an image segmentation model to attempt automatic feature detection. For more information, a project poster is linked here: [Project Poster](https://drive.google.com/file/d/1ENWW2p9eLCgx6ollhio1ItrFacDXWnOA/view?usp=sharing)

### Folders/Files
* The `DTM Visualization Tools` folder contains 3 files that can be run in ArcGIS software with ArcPy to perform visualization techniques on DTMs. The `automatedHillShading.py` file utilizes a small GUI to create a set of custom hillshades, while the `TPI.py` and `DEV.py` files create single rasters. The last file `autoConvertRastersToJPG.py` converts the hillshades into JPEG images that can be used elsewhere for quick viewing or labeling purposes.

* The `lidar` folder contains a script that can be used in GRASS GIS software to convert raw lidar data into a DTM.

* The `preprocess` folder contains files that prepare visualizations (hillshades) and labels for use in the image segmentation model. There are **two scripts** that need to be run in order to preprocess the data: `processGeoJSONlabels_00.py` and `preparedata_01.py`. These files take the exported data from the Azavea Groundwork labeling software, convert and project them, split the images, and augment them to increase the training data size.`rewriteClassIdx.py` is a helper function for the processGEOJSONlabels file. The `kilmac_label` folder contains a special file for the Kilmacahill data. The `transformation_plots` folder contains an example image of split, flipped, and rotated images/labels.

* The `model` folder contains the U-Net image segmentation model in two different forms and a file for running it with a sample dataset. `TestingSegmentationModels.ipynb` runs the model with a sample car image dataset. `trainUnet.ipynb` is a python notebook that can be uploaded into *Google Colab* and run, while `trainUnetCLT.py` is a python script that can be run via command line.

* The `utils` folder contains helper functions for the image segmentation model that do not need to be individually run. They are called on in the python command line version of the model.

### Setting up and Running Machine Learning Feature Detection
#### 1. Processing Azavea labels
Annotations made in Azavea are exported as GEOJSON files. `process/processGeoJSONlabels_00.py· converts vector= annotations into pixelated raster images in ArcGIS using `arcpy`. Thus, it requires ArcGIS Pro installation. Label extraction was done in remote desktop with ArcGIS Pro installation in this project. This command line tool can process label for each site, one by one, or by batch for multiple sites at once. Before running the command line tool, put the raw DTM, selected hillshade (both in .tif formate) and the corresponding GEOJSON file for each site inside one directory. To process the annotation of label of one site, site Ardmayle for example, specify:
```
"%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" processGeoJsonLabels_00.py -d <full_path_to_the_data_folder> -m ardmayle_march14_dtm.tif -hsd Ardmayle225_15.tif -g Ardmayle.geojson               
```
The string before `processGeoJsonLabels_00.py` invoke python interpreter with arcpy installation. Output raster label will be saved under the same directory as the input files.

To run process labels by batch, add a "-b" flag, and there is no need to specify the DTM, hillshade visualization and label files:
```
"%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy" processGeoJsonLabels_00.py -b -d <full_path_to_the_data_folder>
```
All output raster labels will be saved under the same directory as the input files.
`-d full_path_to_the_data_folder` is optional. Editing the default value for this argument argument inside `processGeoJsonLabels_00.py` if necessary. 

Now, the annotations have been processed. To run model training in local machine or Colab, compress the data folder containing DTM, hillshade visualization, GEOJSON label and converted rasterized label files. The upload the folder to Box or any other way.


#### 2. Setting up data after labels are processed
1.Download `00_Data_Preprocessed` from `Box>>Azavea`. <br>
2.Unzip the folder and put it under this git repository.<br>
3.Step 1-2 set up the raw data folder. At this step, your git repository should look like this:
<details>
        <summary>Click to expand!</summary>

        helm-data-plus-2022
        ├── 00_Data_Preprocessed
        │   ├── Ardmayle.geojson
        │   ├── Ardmayle225_15.tif
        │   ├── Ardmayle_label.tif
        │   ├── ardmayle_march14_dtm.tif
        │   ├── Knockainey Town Mapping_dtm.tif
        │   ├── knockainey.geojson
        │   ├── knockainey135_15.tif
        │   └── knockainey_label.tif

</details>

#### 3. Splitting and augmentation.
There are two options at this step. The big DTM, hillshade and label images of each site is splitted in to subimage with overlap, then flip and rotation are applied to increase the size of data for training later. The first option is to do spliting and augmetation from command line, and write out the subimages and their augmented versions as png files. To do this, navigate into the `preprocess` folder and run `preparedata_01.py` by:<br>

```python preparedata_01.py -c 512 -o 64 -in {absolute path to input data} -o {absolute path to output folder}```

If no specify path to input data and output folder, the input folder is `00_Data_Preprocessed`. A new folder `01_Data_Prepared` is created, and the png files will be saved under this folder. The directory would look like this:
<details>
        <summary>Click to expand!</summary>

        helm-data-plus-2022
        ├── 00_Data_Preprocessed
        │   └── ...
        ├── 01_Data_Splitted
        │   ├── ardmayle
        │   │   ├── dtms
        │   │   │   ├── ardmayle0_0.png
        │   │   │   └── ...
        │   │   ├── labels
        │   │   │   ├── ardmayle0_0.png
        │   │   │   └── ...
        │   │   └── hillshades
        │   │       ├── ...
        │   │       └── ...
        │   ├── kibixy
        │   │   ├── dtms/...
        │   │   ├── labels/...
        │   │   └── hillshades/...
        │   └──knockainey/...
        ├── 02_Data_Augmented
        │   ├── ardmayle
        │   │   ├── dtms
        │   │   │   ├── ardmayle0_0.png
        │   │   │   ├── ardmayle0_0hoflip.png
        │   │   │   └── ...
        │   │   ├── labels
        │   │   │   ├── ardmayle0_0.png
        │   │   │   ├── ardmayle0_0hoflip.png
        │   │   │   └── ...
        │   │   └── hillshades
        │   │       ├── ...
        │   │       └── ...
        │   ├── kibixy
        │   │   ├── dtms/...
        │   │   ├── labels/...
        │   │   └── hillshades/...
        │   └──knockainey/...
        ├── utlis/...
        │
        ...

</details>

#### 4.1 Training in Google Colab notebook
1.Open up model/trainUnet.ipynb in Google Colab. <br>
2.Compress 00_Data_Preprocessed, utils and prerprocess folder as zipe files. Upload all three ziped folder into your Google Colab runtime. <br>

#### 4.2 Training in local/ssh remote machine using Command Line Tool
1. Scp 00_Data_Preprocessed, preprocess utils to the same directory as the model folder if needed <br>
2. Run `model/trainUnetCLT.py`. For example:
```
        python trainUnetCLT.py -d 00_Data_Preprocessed -train ardmayle kilbixy -test knockainey -use_h -prop 0.8 -s 512 -overlap 64 -lr 0.0007 -m 0.8 -e 5 -b 10 -f -o train_ard_kilb
```
The meaning of each arguenets and flag are explained below:
```
usage: trainUnetCLT.py [-h] [-d [DATA_ROOT]] [-train TRAIN [TRAIN ...]] [-test TEST [TEST ...]] [-prop [PROP]] [-s [SIZE]]
                       [--overlap [OVERLAP]] [-use_h] [-lr [LR]] [-m [MOMENTUM]] [-e [EPOCH]] [-b [BATCH_SIZE]] [-f] [-l] [-p [PATH]]
                       [-o [MODEL_NAME]]

Train Unet on HLEM data

optional arguments:
  -h, --help            show this help message and exit
  -d [DATA_ROOT], --data_root [DATA_ROOT]
                        input data root
  -train TRAIN [TRAIN ...]
                        training set site name(s) list
  -test TEST [TEST ...]
                        test set site name(s) list
  -prop [PROP]          training val split, train proportion 0-1
  -s [SIZE], --size [SIZE]
                        img size
  --overlap [OVERLAP]   number of pixels on right and bottom to overlap when splitting
  -use_h, --hillshade   use hillshade as input
  -lr [LR], --lr [LR]   learning rate
  -m [MOMENTUM], --momentum [MOMENTUM]
                        momentum
  -e [EPOCH], --epoch [EPOCH]
                        epochs
  -b [BATCH_SIZE], --batch_size [BATCH_SIZE]
                        batch size
  -f, --freeze          freeze encoder weights
  -l, --load            load pretrained weights
  -p [PATH], --path [PATH]
                        path to pretrianed weights that will be loaded before training
  -o [MODEL_NAME], --model_name [MODEL_NAME]
                        customized name for the saved model file

```

    






