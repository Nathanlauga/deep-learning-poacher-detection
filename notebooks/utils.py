import cv2
import os
from os.path import isfile, join
from tqdm import tqdm
import gc

import matplotlib.pyplot as plt
import numpy as np

def extract_frames(file, out_dir='frames', save=False, skip=30):
    """Extract frames from a video file.

    Parameters
    ----------
    file:str
        Video file with path
    out_dir: str (default 'frames')
        Path where to extract the frames
    save: bool (default False)
        Whether you want to save the image or not
    skip: int (default 30):
        Number of images to skip between each save
    """
    video = cv2.VideoCapture(file)
    fname = file.split('/')[-1]
    count = 0

    success, image = video.read()
    while success:
        if save & (count % skip == 0):
            countFormated = "{0:0=5d}".format(count)
            cv2.imwrite("%s/%s_frame_%s.jpg" % (out_dir, fname, countFormated), image)
            
        success, image = video.read()
        count += 1

    print('Frames saved to `%s/` directory.' % (out_dir))

    video.release()

def convert_frames_to_video(pathIn,pathOut,fps,size):
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in tqdm(range(len(files))):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width,height)
        #inserting the frames into an image array
        out.write(img)

        # del img
        gc.collect()


    out.release()
    
def move_file(file, new_file):
    """Moves a file to a new directory

    Parameters
    ----------
    file: str
        Current path for the file to move
    new_file: str
        New path for the file
    """
    os.rename(file, new_file)


def move_frames(dir_name, out_dir, valid_ext=['png', 'jpg', 'jpeg']):
    """For the given path, get the List of 
    all files in the directory tree

    Parameters
    ----------
    dir_name: str
        Directory where to get frames
    out_dir: str
        Directory where to save all frames
    valid_ext: list
        list of valid extensions to move files 

    Returns
    -------
    list:
        File list inside directory
    """
    # create a list of file and sub directories
    # names in the given directory
    files_list = os.listdir(dir_name)
    files = list()

    # Iterate over all the entries
    for entry in files_list:
        # Create full path
        path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(path):
            files_tmp = move_frames(path, out_dir)

            if len(files_tmp) > 0:
                if files_tmp[0].split('.')[-1] in valid_ext:
                    if path.split('/')[-1] == files_tmp[0].split('/')[-2]:
                        for file in files_tmp:
                            new_file = out_dir+'/' + \
                                ('_'.join(file.split('/')[-2:]))
                            move_file(file, new_file)
        else:
            files.append(path)

    return files


def regroup_frames_same_dir(parent_dir, out_dir='/data/train'):
    """Regroup all frames in the same directory.

    Given a parent directory, find all picture file 
    (extension jpg, png or jpeg) and move them to out_dir

    Parameters
    ----------
    parent_dir: str
        Parent directory where all the directories with 
        drone videos are
    out_dir: str
        Directory where to save all frames
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print("Directory", out_dir, "created.")
    else:
        print("Directory ", out_dir, " already exists.")

    move_frames(parent_dir, out_dir)

    files_list = os.listdir(out_dir)
    print('There are %i frames moved to `%s` directory.' %
          (len(files_list), out_dir))


def get_files_by_ext(dir_name, ext=['txt'], recursive=False):
    """Retrieves all files from a directory
    using extension list

    Parameters
    ----------
    dir_name: str
        Path of the directory
    ext: list
        list of valid extensions
    recursive: boolean
        True : Read images in sub-folders of dir_name
        False : Read images only in the folder dir_name
    
    Returns
    -------
    list:
        files list
    """
    files_list = os.listdir(dir_name)
    files = list()
    for entry in files_list:
        path = os.path.join(dir_name, entry)
        if os.path.isdir(path):
            if recursive:
                files += get_files_by_ext(path, ext=ext, recursive=True)
        elif path.split('.')[-1] in ext:
            files.append(path)

    return files

def image_to_matrix(path, resize_shape=(416,416)):
    """
    Convert an image (using path) to a numpy array 
    
    Parameters
    ----------
    path: str or list
        Path(s) to the image(s) to convert
    resize_shape: tuple (default (416,416))
        Size for the matrix (default for Yolov3 model)
    
    Returns
    -------
    np.array
        matrix with shape (3,h,w)
    """
    scale = (1 / 255)
    
    if type(path) == list:
        # images = [cv2.resize(cv2.imread(p), resize_shape) for p in path]
        images = [cv2.imread(p) for p in path]
        blob = cv2.dnn.blobFromImages(images, scale, resize_shape, 
                                     (0,0,0), True, crop=False)
        
    else:
        image = cv2.imread(path)
        # image = cv2.resize(image, resize_shape)
        blob = cv2.dnn.blobFromImage(image, scale, resize_shape, 
                                     (0,0,0), True, crop=False)
    return blob    


def plot_image(image):
    """Plots the image.
    
    Parameters
    ----------
    image: np.array
        Matrix of the image with (3,h,w) shape
    """
    if len(image.shape) == 4:
        image = image.reshape(image.shape[1:])
    
    image = image.transpose((1,2,0))
    
    fig = plt.figure(figsize=(7,7))
    plt.imshow(image)
    plt.show()

    
def detect_object(outs, list_images, Width, Height, nb_out_layer):    
    i = 0
    dict_obj_detected = {}

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
#         #Dimension1 = Number of Images
#         #Dimension2 = X_out_grid * Y_out_grid * nb_out_layer
#         #Dimension3 = 5 + nb_classes
#         out = out.reshape(out.shape[0],\
#                           out.shape[1]*out.shape[2]*nb_out_layer,\
#                           int(out.shape[3]/nb_out_layer)
#                          )
        
        for image in out:
            image_name = list_images[i]
            if not image_name in dict_obj_detected:
                dict_obj_detected[image_name] = {}
                dict_obj_detected[image_name]["class_ids"] = list()
                dict_obj_detected[image_name]["confidences"] = list()
                dict_obj_detected[image_name]["boxes"] = list()
            for detection in image:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    dict_obj_detected[image_name]["class_ids"].append(class_id)
                    dict_obj_detected[image_name]["confidences"].append(float(confidence))
                    dict_obj_detected[image_name]["boxes"].append([x, y, w, h])
            i += 1
        i = 0 

    return dict_obj_detected

def detect_danger(dict_obj_detected, idx_class=0):
    
    dict_image_danger = {}
    
    for image_name, row in dict_obj_detected.items():
        if len(row["class_ids"]) != 0 and idx_class in row["class_ids"]: 
            dict_image_danger[image_name] = True
        else:
            dict_image_danger[image_name] = False
    
    return dict_image_danger

## Function used to draw bouding boxes onto an image, and save this image. 
## Before calling this function, you need to get the dict_obj_detected, while calling the function detect_object()

def get_bounding_box(image_path, image_items, classes, COLORS, conf_threshold, nms_threshold):
#   apply non-max suppression
    indices = cv2.dnn.NMSBoxes(image_items["boxes"], image_items["confidences"], conf_threshold, nms_threshold)
    image = cv2.imread(image_path)

#   go through the detections remaining
#   after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = image_items["boxes"][i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, image_items["class_ids"][i], image_items["confidences"][i], \
                                    round(x), round(y), round(x+w), round(y+h), \
                                    classes, COLORS
                               )
    # save output image to disk
    cv2.imwrite(os.path.dirname(image_path) + "/output_with_bounding_box/" + os.path.basename(image_path), image)

    # release resources
    # cv2.destroyAllWindows()

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
def get_video_frame_number(image_name):
    frame_name = image_name.split("/")[3]
    video_number = frame_name.split(".")[0] + "." + \
                    frame_name.split(".")[1] + "." +\
                    frame_name.split(".")[2]
    frame_number = frame_name.split(".")[3].split("_")[2]
    frame_number = frame_number.lstrip("0")
    if frame_number == '':
        frame_number = 0 
    return video_number, int(frame_number)