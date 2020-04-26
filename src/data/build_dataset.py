import pandas as pd
import yaml
import os
import cv2
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.data.filter_beam import filter_beam
from src.data.preprocess import to_greyscale

def mp4_to_images(mp4_path):
    '''
    Converts an mp4 video to a series of images and save the images in the same directory. Applies beam filtering
    algorithm to mask out everything but the US beam in the images.
    :param mp4_path: File name of the mp4 file to convert to series of images.
    '''

    vc = cv2.VideoCapture(mp4_path)
    vid_dir, mp4_filename = os.path.split(mp4_path)      # Get folder and filename of mp4 file respectively
    mp4_filename = mp4_filename.split('.')[0]       # Strip file extension

    idx = 0
    while (True):
        ret, frame = vc.read()
        if not ret:
            break   # End of frames reached
        img_path = vid_dir + '/' + mp4_filename + '_' + str(idx) + '.jpg'
        frame = filter_beam(frame, triangles_mask=True)  # Mask out everything but US beam
        grey_frame = to_greyscale(frame)
        cv2.imwrite(img_path, grey_frame)
        idx += 1

def build_encounter_dataframe(cfg):
    '''
    Build a dataframe of filenames and labels. If mp4
    :param cfg: Project config dictionary
    :return: DataFrame of file paths of examples and their corresponding class labels
    '''

    # Get paths of raw datasets to be included
    ncovid_data_path = cfg['PATHS']['NCOVID_DATA']
    covid_data_path = cfg['PATHS']['COVID_DATA']

    class_dict = {cfg['DATA']['CLASSES'][i]: i for i in range(len(cfg['DATA']['CLASSES']))}  # Map class name to number

    # Label all non-COVID-19 encounters
    ncovid_encounter_dirs = []
    for encounter_dir in os.listdir(ncovid_data_path):
        encounter_dir = os.path.join(ncovid_data_path, encounter_dir).replace("\\","/")
        if (os.path.isdir(encounter_dir)):
            ncovid_encounter_dirs.append(encounter_dir)
    ncovid_encounter_df = pd.DataFrame({'encounter': ncovid_encounter_dirs, 'label': class_dict['NCOVID']})

    # Label all COVID-19 encounters
    covid_encounter_dirs = []
    for encounter_dir in os.listdir(covid_data_path):
        encounter_dir = os.path.join(covid_data_path, encounter_dir).replace("\\","/")
        if (os.path.isdir(encounter_dir)):
            covid_encounter_dirs.append(encounter_dir)
    covid_encounter_df = pd.DataFrame({'encounter': covid_encounter_dirs, 'label': class_dict['COVID']})

    encounter_df = pd.concat([ncovid_encounter_df, covid_encounter_df], axis=0)  # Combine both datasets
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
                for mp4_file in glob.glob(row['encounter'] + '/*.mp4'):
                    mp4_to_images(mp4_file)     # Convert mp4 encounter file to image files
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
    print("Partitioning training, validation and test sets.")
    val_split = cfg['DATA']['VAL_SPLIT']
    test_split = cfg['DATA']['TEST_SPLIT']
    encounter_df_train, encounter_df_test = train_test_split(encounter_df, test_size=test_split,
                                                             stratify=encounter_df['label'])
    relative_val_split = val_split / (1 - test_split)  # Calculate fraction of train_df to be used for validation
    encounter_df_train, encounter_df_val = train_test_split(encounter_df_train, test_size=relative_val_split,
                                                      stratify=encounter_df_train['label'])

    # Build Pandas dataframes to link image file names and labels.
    print("Building training set")
    file_df_train = build_file_dataframe(cfg, encounter_df_train, img_overwrite=img_overwrite)
    print("Building validation set")
    file_df_val = build_file_dataframe(cfg, encounter_df_val, img_overwrite=img_overwrite)
    print("Building test set")
    file_df_test = build_file_dataframe(cfg, encounter_df_test, img_overwrite=img_overwrite)

    # Save training, validation and test sets
    if not os.path.exists(cfg['PATHS']['PROCESSED_DATA']):
        os.makedirs(cfg['PATHS']['PROCESSED_DATA'])
    file_df_train.to_csv(cfg['PATHS']['TRAIN_SET'])
    file_df_val.to_csv(cfg['PATHS']['VAL_SET'])
    file_df_test.to_csv(cfg['PATHS']['TEST_SET'])
    encounter_df_train.to_csv(cfg['PATHS']['TRAIN_ENCOUNTERS'])
    encounter_df_val.to_csv(cfg['PATHS']['VAL_ENCOUNTERS'])
    encounter_df_test.to_csv(cfg['PATHS']['TEST_ENCOUNTERS'])
    return

if __name__ == '__main__':
    build_dataset(img_overwrite=False)