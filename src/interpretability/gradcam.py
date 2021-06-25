import yaml
import os
import datetime
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from src.predict import predict_instance, predict_set
from src.visualization.visualize import visualize_heatmap
from src.predict import get_preprocessing_function

def setup_gradcam():
    '''
    Load relevant variables to apply Grad-CAM
    :return: dict containing important information and objects for Grad-CAM visualizations
    '''

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    setup_dict = {}

    setup_dict['MODEL'] = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    setup_dict['IMG_PATH'] = cfg['PATHS']['IMAGES']
    setup_dict['RAW_DATA_PATH'] = cfg['PATHS']['RAW_DATA']
    setup_dict['OTTAWA_SET'] = pd.read_csv(cfg['PATHS']['OTTAWA_SET'])
    setup_dict['IMG_DIM'] = cfg['DATA']['IMG_DIM']
    setup_dict['CLASSES'] = cfg['DATA']['CLASSES']

    # Get name of final convolutional layer
    layer_name = ''
    for layer in setup_dict['MODEL'].layers:
        if any('Conv2D' in l for l in layer._keras_api_names):
            layer_name = layer.name
    setup_dict['LAYER_NAME'] = layer_name
    return setup_dict


def apply_gradcam(cfg, setup_dict, dataset, hm_intensity=0.5, dir_path=None):
    '''
    For each image in the dataset provided, make a prediction and overlay a heatmap depicting the gradient of the
    predicted class with respect to the feature maps of the final convolutional layer of the model.
    :param setup_dict: dict containing important information and objects for Grad-CAM
    :param dataset: Pandas Dataframe of examples, linking image filenames to labels
    :param hm_intensity: Overall visual intensity of the heatmap to be overlaid onto the original image
    :param dir_path: Path to directory to save Grad-CAM heatmap visualizations
    '''

    # Create ImageDataGenerator for test set
    if cfg['TRAIN']['MODEL_DEF'] in ['custom_resnet', 'custom_ffcnn']:
        test_img_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    else:
        test_img_gen = ImageDataGenerator(preprocessing_function=get_preprocessing_function(cfg['TRAIN']['MODEL_DEF']))
    test_generator = test_img_gen.flow_from_dataframe(dataframe=dataset,
                                                      directory=cfg['PATHS']['RAW_DATA'],
                                                      x_col="filename", y_col='label_str',
                                                      target_size=tuple(cfg['DATA']['IMG_DIM']), batch_size=1,
                                                      class_mode='categorical', validate_filenames=False, shuffle=False)

    preds, probs = predict_set(cfg, setup_dict['MODEL'], get_preprocessing_function(cfg['TRAIN']['MODEL_DEF']), dataset)

    for idx in tqdm(range(probs.shape[0])):

        # Get idx'th preprocessed image in the  dataset
        x, y = test_generator.next()

        # Get the corresponding original image (no preprocessing)
        orig_img = cv2.imread(setup_dict['RAW_DATA_PATH'] + dataset['filename'].iloc[idx])
        new_dim = tuple(setup_dict['IMG_DIM'])
        orig_img = cv2.resize(orig_img, new_dim, interpolation=cv2.INTER_NEAREST)     # Resize image

        # Obtain gradient of output with respect to last convolutional layer weights
        with tf.GradientTape() as tape:
            last_conv_layer = setup_dict['MODEL'].get_layer(setup_dict['LAYER_NAME'])
            iterate = Model([setup_dict['MODEL'].inputs], [setup_dict['MODEL'].output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(x)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

        # Upsample and overlay heatmap onto original image
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0.0)    # Equivalent of passing through ReLU
        heatmap /= np.max(heatmap)
        heatmap = heatmap.squeeze(axis=0)
        heatmap = cv2.resize(heatmap, tuple(setup_dict['IMG_DIM']))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_img = cv2.addWeighted(heatmap, hm_intensity, orig_img, 1.0 - hm_intensity, 0)

        # Visualize the Grad-CAM heatmap and optionally save it to disk
        img_filename = dataset['filename'].iloc[idx]
        label = dataset['label'].iloc[idx]
        _ = visualize_heatmap(orig_img, heatmap_img, img_filename, label, probs[idx], setup_dict['CLASSES'],
                                  dir_path=dir_path)
    return heatmap


def apply_gradcam_to_encounter(setup_dict, encounter_path):
    '''
    Apply Grad-CAM to each image in an encounter and save all results in a new directory.
    :param setup_dict: dict containing important information and objects for Grad-CAM
    :param encounter_path: Absolute path to an encounter folder
    '''

    # Get indices in test set of images comprising the encounter
    enc_name = encounter_path.split('/')[-1].split(' ')[0]
    enc_files_df = setup_dict['OTTAWA_SET'][setup_dict['OTTAWA_SET']['filename'].str.contains(enc_name)]
    assert enc_files_df.shape[0] > 0, 'Cannot find any images in test set with this encounter name.'

    # We will save all heatmaps in the same folder
    heatmap_dir = setup_dict['IMG_PATH'] + enc_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    os.mkdir(heatmap_dir)

    # Apply Grad-CAM to each image in the encounter
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    _ = apply_gradcam(cfg, setup_dict, enc_files_df, hm_intensity=0.5, dir_path=heatmap_dir)
    return


def run_gradcam_on_img(setup_dict, idx, hm_intensity=0.5, dir_path=None):
    '''
    Apply Grad-CAM to an individual image at a specified index in the test set.
    :param setup_dict: dict containing important information and objects for Grad-CAM
    :param idx: index of image in the test set
    :param hm_intensity: Overall visual intensity of the heatmap to be overlaid onto the original image
    :param dir_path: Path to directory to save Grad-CAM heatmap visualizations
    :return: The heatmap produced by Grad-CAM
    '''
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    files_df = pd.read_csv(cfg['PATHS']['OTTAWA_SET'])
    row = files_df.iloc[[idx]]
    heatmap = apply_gradcam(cfg, setup_dict, row, hm_intensity=hm_intensity, dir_path=dir_path)
    return heatmap


if __name__ == '__main__':
    setup_dict = setup_gradcam()
    heatmap = run_gradcam_on_img(setup_dict, 11700, hm_intensity=0.5, dir_path=setup_dict['IMG_PATH'])    # Generate heatmap for image
    #apply_gradcam_to_encounter(setup_dict, '/home/ampc/raw/COVID/10.15.39 hrs __0002107')

