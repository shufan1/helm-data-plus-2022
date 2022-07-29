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






