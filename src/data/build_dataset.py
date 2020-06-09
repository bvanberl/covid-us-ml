import pandas as pd
import yaml
import os
import cv2
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.data.filter_beam import filter_beam, find_contour_area, contour_image
from src.data.preprocess import to_greyscale

def mp4_to_images(mp4_path):
    '''
    Converts an mp4 video to a series of images and saves the images in the same directory.
    Calculates the area of the largest contour for every image and relays the image with 
    the largest area to mask_all_images. 
    :param mp4_path: File name of the mp4 file to convert to series of images.
    :return: Returns the mask used, else mp4_path and number of frames if failed to find contour.
    '''
    vc = cv2.VideoCapture(mp4_path)
    vid_dir, mp4_filename = os.path.split(mp4_path)      # Get folder and filename of mp4 file respectively
    mp4_filename = mp4_filename.split('.')[0]       # Strip file extension

    idx = 0
    max_area = 0
    max_area_id = 0
    while (True):
        ret, frame = vc.read()
        if not ret:
            break   # End of frames reached
        img_path = vid_dir + '/' + mp4_filename + '_' + str(idx) + '.jpg'
        area = find_contour_area(frame)
        if area > max_area: # Record which contour has the maximum area
            max_area = area
            max_area_id = idx
        cv2.imwrite(img_path, frame) # Save all the images out
        idx += 1

    mask = mask_all_images(vid_dir + '/' + mp4_filename + '_', max_area_id, idx) # Find and return mask.

    if isinstance(mask, int): # mask_all_images returns 0 if the algorithm failed
        return (vid_dir + '/' + mp4_filename + '_', idx) # Return a tuple of the video on which the algorithm failed
    else:
        return mask # Return the mask.

def mask_all_images(img_dir, max_area_id, idx, temp_mask=0):
    '''
    Finds the mask of the image with the largest area, and applies that mask
    to every frame in the video.
    :param img_dir: the path to the images
    :param max_area_id: the id of the frame with the largest contour area
    :param idx: the number of frames in the video
    :return: a number indicating whether the process succeeded or failed
    '''
    if isinstance(temp_mask, int): # If there is no temp_mask to use
        frame = cv2.imread(img_dir + str(max_area_id)+'.jpg') 
        mask = filter_beam(frame) # Calculate the mask using the frame with the largest contour
    else:
        mask = temp_mask

    if isinstance(mask, int): # If filter_beam fails, it returns 0
        return 0
    else: # Apply the mask onto all images in the video.
        for i in range(idx):
            frame = cv2.imread(img_dir + str(i) + '.jpg') 
            frame = cv2.bitwise_and(frame, mask) # Use that mask on every frame in the video
            grey_frame = to_greyscale(frame)
            cv2.imwrite(img_dir + str(i) + '.jpg', grey_frame)  # Overwrite every image
        return mask

def contour_all_images(mp4_path): 
    '''
    For all frames in the video, create a mask to blacken all regions not bound
    by the largest contour found within the image.
    :param mp4_path: a tuple containing the path to the frames in index 0, and
                     the number of frames in index 1
    '''
    for i in range(mp4_path[1]): 
        image = cv2.imread(mp4_path[0] + str(i) + '.jpg')
        image = contour_image(image) # Find the largest contour in each image and use it as the mask.
        grey_frame = to_greyscale(image)
        cv2.imwrite(mp4_path[0] + str(i) + '.jpg', grey_frame)

def build_encounter_dataframe(cfg):
    '''
    Build a dataframe of filenames and labels. If mp4
    :param cfg: Project config dictionary
    :return: DataFrame of file paths of examples and their corresponding class labels
    '''

    # Get paths of raw datasets to be included
    ncovid_data_path = cfg['PATHS']['NCOVID_DATA']
    covid_data_path = cfg['PATHS']['COVID_DATA']
    smooth_data_path = cfg['PATHS']['SMOOTH_DATA']

    class_dict = {cfg['DATA']['CLASSES'][i]: i for i in range(len(cfg['DATA']['CLASSES']))}  # Map class name to number

    # Label all encounters for COVID-19 class
    covid_encounter_dirs = []
    for encounter_dir in os.listdir(covid_data_path):
        encounter_dir = os.path.join(covid_data_path, encounter_dir).replace("\\","/")
        if (os.path.isdir(encounter_dir)):
            covid_encounter_dirs.append(encounter_dir)
    covid_encounter_df = pd.DataFrame({'encounter': covid_encounter_dirs, 'label': class_dict['COVID']})

    # Label all encounters for COVID-19 class
    ncovid_encounter_dirs = []
    for encounter_dir in os.listdir(ncovid_data_path):
        encounter_dir = os.path.join(ncovid_data_path, encounter_dir).replace("\\","/")
        if (os.path.isdir(encounter_dir)):
            ncovid_encounter_dirs.append(encounter_dir)
    ncovid_encounter_df = pd.DataFrame({'encounter': ncovid_encounter_dirs, 'label': class_dict['NCOVID']})

    # Label all encounters for COVID-19 class
    smooth_encounter_dirs = []
    for encounter_dir in os.listdir(smooth_data_path):
        encounter_dir = os.path.join(smooth_data_path, encounter_dir).replace("\\","/")
        if (os.path.isdir(encounter_dir)):
            smooth_encounter_dirs.append(encounter_dir)
    smooth_encounter_df = pd.DataFrame({'encounter': smooth_encounter_dirs, 'label': class_dict['SMOOTH']})

    # Combine all encounters data
    encounter_df = pd.concat([ncovid_encounter_df, covid_encounter_df, smooth_encounter_df], axis=0)
    return encounter_df


def build_file_dataframe(cfg, encounter_df, img_overwrite=False):
    '''
    Build a Pandas Dataframe linking US image file names and classes. If the encounter directories only contain mp4
    video files, save each frame as a jpg file in the same directory.
    :param cfg: Project config
    :param encounter_df: A Pandas Dataframe linking encounter directory paths and labels
    :return: A Pandas Dataframe linking file names for images from these encounters with labels
    '''

    label_dict = {i: cfg['DATA']['CLASSES'][i] for i in range(len(cfg['DATA']['CLASSES']))}  # Map class number to name

    # Iterate over all encounters and build a Pandas dataframe linking image file names and classes.
    filenames = []
    labels = []
    counter = 0
    for index, row in tqdm(encounter_df.iterrows()):
        if counter % 10 == 0:
            print(str(counter) + ' / ' + str(encounter_df.shape[0]) + ' encounters')    # Keep track of progress
        counter += 1
        if (os.path.isdir(row['encounter'])):
            if (not glob.glob(row['encounter'] + '/*.jpg')) or img_overwrite:
                needs_reprocessing = []
                temp_mask = 0
                for mp4_file in glob.glob(row['encounter'] + '/*.mp4'):
                    path = mp4_to_images(mp4_file) # Convert mp4 encounter file to image files
                    if isinstance(path, tuple): # mp4_to_images failed for this path
                        needs_reprocessing.append(path)
                    elif isinstance(temp_mask, int):
                        temp_mask = path # saves the first mask found
                if isinstance(temp_mask, int): # If images need reprocessing and there is no temp_mask from another image in the same folder
                    for mp4_file in needs_reprocessing: 
                        contour_all_images(mp4_file) # Use the largest contour in each image as the mask
                else:
                    for mp4_file in needs_reprocessing: # Use the temp_mask from another video to mask any failed videos.
                        mask_all_images(mp4_file[0], 0, mp4_file[1], temp_mask)     

            encounter_filenames = [f.replace(cfg['PATHS']['RAW_DATA'], '').replace("\\","/")
                                   for f in glob.glob(row['encounter'] + "/*.jpg")]
            filenames.extend(encounter_filenames)
            labels.extend([row['label']] * len(encounter_filenames))

    file_df = pd.DataFrame({'filename': filenames, 'label': labels})
    file_df['label_str'] = file_df['label'].map(label_dict)             # Add column for string representation of label
    return file_df



def build_dataset(cfg=None, img_overwrite=False):
    '''
    Build and partition dataset. Assemble all image file paths, assign labels. Then partition into training, validation
    test sets.
    :param cfg: Optional parameter to set your own config object.
    :param img_overwrite: Boolean indicating whether to rewrite generated images.
    '''

    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))  # Load config data

    # Build Pandas dataframe of filenames and corresponding labels.
    print("Assigning labels to encounters.")
    encounter_df = build_encounter_dataframe(cfg)

    # Randomly split encounters dataframe into train, val and test sets
    print("Partitioning training and test sets.")
    test1_split = cfg['DATA']['TEST1_SPLIT']
    test2_split = cfg['DATA']['TEST2_SPLIT']
    encounter_df, encounter_df_test2 = train_test_split(encounter_df, test_size=test2_split,
                                                        random_state=42, stratify=encounter_df['label'])
    relative_test1_split = test1_split / (1 - test2_split)  # Calculate fraction of train_df to be used for validation
    encounter_df_trainval, encounter_df_test1 = train_test_split(encounter_df, test_size=relative_test1_split,
                                                                 random_state=42, stratify=encounter_df['label'])

    # Build Pandas dataframe to link all image file names and labels.
    print("Building image dataset")
    test1_file_df = build_file_dataframe(cfg, encounter_df_test1, img_overwrite=img_overwrite)
    test2_file_df = build_file_dataframe(cfg, encounter_df_test2, img_overwrite=img_overwrite)
    train_file_df = build_file_dataframe(cfg, encounter_df_trainval, img_overwrite=img_overwrite)

    # Save training, test1 and test2 sets
    if not os.path.exists(cfg['PATHS']['PROCESSED_DATA']):
        os.makedirs(cfg['PATHS']['PROCESSED_DATA'])
    train_file_df.to_csv(cfg['PATHS']['TRAIN_SET'])
    test1_file_df.to_csv(cfg['PATHS']['TEST1_SET'])     # For assessing individual models
    test2_file_df.to_csv(cfg['PATHS']['TEST2_SET'])     # To be left out until the final model
    encounter_df_trainval.to_csv(cfg['PATHS']['ENCOUNTERS_TRAINVAL'])   # Train and val to be partitioned upon training
    encounter_df_test1.to_csv(cfg['PATHS']['ENCOUNTERS_TEST1'])
    encounter_df_test2.to_csv(cfg['PATHS']['ENCOUNTERS_TEST2'])
    return

if __name__ == '__main__':
    build_dataset(img_overwrite=False)