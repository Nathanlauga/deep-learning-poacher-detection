import cv2
import os


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
            cv2.imwrite("%s/%s_frame%d.jpg" % (out_dir, fname, count), image)

        success, image = video.read()
        count += 1

    print('Frames saved to `%s/` directory.' % (out_dir))

    video.release()


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
