
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv2

# cutsom rgb values for each feature
fw_default_c = (29, 130, 161) # RGB color for brugage plot
mf_default_c = (199, 56, 84)
bg_default_c = (255,255,255)

def make_rgba_label (label_array,alpha=1,
                    fw_color = fw_default_c,
                    mf_color= mf_default_c,
                    bg_color = bg_default_c):
    # by default return not transparent label
    
    color_list = [bg_color, fw_color,mf_color]
    unique_class = np.unique(label_array)
    label_rgb = 255 * np.ones((*label_array.shape, 3), dtype=np.uint8)
    label_alpha = int(alpha*255) # 0 is totally transparent, 
    for i in range(len(unique_class)):
        class_i_label = label_array== unique_class[i]
        label_rgb[class_i_label, :] = color_list[i]
    
    bg_label= label_array==0
    if label_alpha!=0:
        # create alpha channel, background is totally transparent
        alpha_label = np.full(label_array.shape,label_alpha )
        alpha_label[bg_label] = 0
    else:
        alpha_label = np.full(label_array.shape,255)
    
    # First convert rgb label to rgba with alpha channel
    label_rgba = cv2.cvtColor(label_rgb, cv2.COLOR_RGB2RGBA)
    # Then assign the label to the last channel of the image
    label_rgba[:, :, 3] = alpha_label
    label_rgba_img = Image.fromarray(label_rgba)

    return label_rgba_img

def overlay_map_label(hsd,label,label_alpha,
                    fw_color = fw_default_c,
                    mf_color= mf_default_c,
                    bg_color = bg_default_c):
    if hsd.mode != 'RGB':
        img_rgb = hsd.convert('RGB')

    label_array = np.asarray(label) if type(label) != "numpy.ndarray" else label
    label_rgba_img = make_rgba_label(label_array,alpha=label_alpha,
                                    fw_color  = fw_color,mf_color=mf_color,
                                    bg_color=bg_color) 
    img_rgb.paste(label_rgba_img,mask =label_rgba_img)

    return label_rgba_img, img_rgb


