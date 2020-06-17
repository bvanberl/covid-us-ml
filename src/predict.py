import yaml, os, dill, json
import numpy as np
import pandas as pd
from sklearn.metrics import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2_preprocess
from src.visualization.visualize import *

def get_preprocessing_function(model_type):
    '''
    Get the preprocessing function according to the type of TensorFlow model
    :param model_type: The pretrained model
    :return: A reference to the appropriate preprocessing function
    '''
    if model_type == 'resnet50v2':
        return resnet_preprocess
    elif model_type == 'resnet101v2':
        return resnet_preprocess
    elif model_type == 'inceptionv3':
        return inceptionv3_preprocess
    elif model_type == 'vgg16':
        return vgg16_preprocess
    elif model_type == 'mobilenetv2':
        return mobilenetv2_preprocess
    elif model_type == 'inceptionresnetv2':
        return inceptionresnetv2_preprocess
    elif model_type == 'xception':
        return xception_preprocess
    else:
        return None


def predict_instance(x, model):
    '''
    Runs model prediction on 1 or more input images.
    :param x: Image(s) to predict
    :param model: A Keras model
    :return: A numpy array comprising a list of class probabilities for each prediction
    '''
    y = model.predict(x)  # Run prediction on the perturbations
    return y


def predict_set(cfg, model, preprocessing_func, dataset):
    '''
    Given a dataset, make predictions for each constituent example.
    :param cfg: project config
    :param model: A trained Keras model
    :param preprocessing_func: Preprocessing function to apply before sending image to model
    :param dataset: Pandas Dataframe of examples, linking image filenames to labels
    :return: List of predicted classes, array of classwise prediction probabilities
    '''

    if cfg['TRAIN']['MODEL_DEF'] in ['custom_resnet', 'custom_ffcnn']:
        test_img_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    else:
        test_img_gen = ImageDataGenerator(preprocessing_function=preprocessing_func)

    test_generator = test_img_gen.flow_from_dataframe(dataframe=dataset, directory=cfg['PATHS']['RAW_DATA'],
        x_col="filename", y_col='label_str', target_size=tuple(cfg['DATA']['IMG_DIM']), batch_size=cfg['TRAIN']['BATCH_SIZE'],
        class_mode='categorical', validate_filenames=True, shuffle=False)
    class_idx_map = dill.load(open(cfg['PATHS']['OUTPUT_CLASS_INDICES'], 'rb'))
    class_idx_map = {v: k for k, v in class_idx_map.items()}    # Reverse the map

    p = model.predict_generator(test_generator)
    test_predictions = np.argmax(p, axis=1)

    # Get prediction classes in original labelling system
    pred_classes = [class_idx_map[v] for v in list(test_predictions)]
    test_predictions = [cfg['DATA']['CLASSES'].index(c) for c in pred_classes]
    return test_predictions, p


def compute_metrics(cfg, labels, preds, probs):
    '''
    Given labels and predictions, compute some common performance metrics
    :param cfg: project config
    :param labels: List of labels
    :param preds: List of predicted classes
    :param probs: Array of predicted classwise probabilities
    :return: A dictionary of metrics
    '''

    metrics = {}
    class_names = cfg['DATA']['CLASSES']

    precisions = precision_score(labels, preds, average=None)
    recalls = recall_score(labels, preds, average=None)
    f1s = f1_score(labels, preds, average=None)

    metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    metrics['precision'] = {class_names[i]:precisions[i] for i in range(len(precisions))}
    metrics['recall'] = {class_names[i]:recalls[i] for i in range(len(recalls))}
    metrics['f1'] = {class_names[i]:f1s[i] for i in range(len(f1s))}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['macro_mean_auc'] = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
    metrics['weighted_mean_auc'] = roc_auc_score(labels, probs, average='weighted', multi_class='ovr')
    print(metrics)
    return metrics


def compute_metrics_by_encounter(cfg, dataset_files_path, dataset_encounters_path):
    '''
    For a particular dataset, use predictions for each filename to create predictions for whole encounters and save the
    resulting metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    :param dataset_encounters_path: Path to CSV of Dataframe linking encounters to labels
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    preprocessing_fn = get_preprocessing_function(model_type)
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = dataset_files_path.split('/')[-1].split('.')[0] + '_encounters'

    files_dataset = pd.read_csv(dataset_files_path)
    encounters_df = pd.read_csv(dataset_encounters_path)

    encounter_labels = encounters_df['label']
    encounter_names = encounters_df['encounter']
    encounter_pred_classes = []
    avg_pred_probs = np.zeros((encounters_df.shape[0], len(cfg['DATA']['CLASSES'])))
    for i in range(len(encounter_names)):

        # Get records from all files from this encounter
        enc_name = encounter_names[i].split('/')[-1].split(' ')[0]
        enc_files_df = files_dataset[files_dataset['filename'].str.contains(enc_name)]
        print("Making predictions for encounter " + enc_name)

        # Make predictions for each image
        pred_classes, pred_probs = predict_set(cfg, model, preprocessing_fn, enc_files_df)

        # Compute average prediction probabilities for entire encounter
        avg_pred_prob = np.mean(pred_probs, axis=0)
        avg_pred_probs[i] = avg_pred_prob

        # Record predicted class
        encounter_pred = np.argmax(avg_pred_prob)
        encounter_pred_classes.append(encounter_pred)

    metrics = compute_metrics(cfg, np.array(encounter_labels), np.array(encounter_pred_classes), avg_pred_probs)
    print(metrics)
    doc = json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'encounters_' + set_name + '.json', 'w'))
    return


def compute_metrics_by_frame(cfg, dataset_files_path):
    '''
    For a particular dataset, make predictions for each image and compute metrics. Save the resultant metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    preprocessing_fn = get_preprocessing_function(model_type)
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = dataset_files_path.split('/')[-1].split('.')[0] + '_frames'

    files_df = pd.read_csv(dataset_files_path)
    frame_labels = files_df['label']    # Get ground truth

    # Make predictions for each image
    pred_classes, pred_probs = predict_set(cfg, model, preprocessing_fn, files_df)

    # Compute and save metrics
    metrics = compute_metrics(cfg, np.array(frame_labels), np.array(pred_classes), pred_probs)
    doc = json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'frames_' + set_name + '.json', 'w'))
    return


if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    dataset_path = cfg['PATHS']['TEST1_SET']
    encounters_path = cfg['PATHS']['ENCOUNTERS_TEST1']
    compute_metrics_by_encounter(cfg, dataset_path, encounters_path)
    #compute_metrics_by_frame(cfg, dataset_path)

