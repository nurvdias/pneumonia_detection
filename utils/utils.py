import os
import sys
import glob
import pylab
import pydicom
import numpy as np
from tqdm import tqdm
import requests


def download_dataset(dataset_dir, kaggle_username, kaggle_key):
    """
    Download RSNA Pneumonia dataset
    """

    assert dataset_dir != None, 'Provide a destination directory for the dataset'
    assert kaggle_username != None and kaggle_key != None, 'Provide your kaggle username and api key'

    DATASET_ABS_PATH = os.path.abspath(dataset_dir)

    if not os.path.exists(DATASET_ABS_PATH):
        print('Creating dataset directory.')
        os.makedirs(DATASET_ABS_PATH)
    else:
        print('Dataset directory already createad.')
        
    os.chdir(DATASET_ABS_PATH)

    # Download dataset if it does not exist
    if not os.path.exists(os.path.join(DATASET_ABS_PATH, 'test_images')) or \
        not os.path.exists(os.path.join(DATASET_ABS_PATH, 'train_images')) or \
        not os.path.exists(os.path.join(DATASET_ABS_PATH, 'train_labels.csv')) or \
        not os.path.exists(os.path.join(DATASET_ABS_PATH, 'detailed_class_labels.csv')):
        
        print('Downloading dataset. This might take a few minutes....')
        
        # Enter your Kaggle credentionals here
        os.environ['KAGGLE_USERNAME']=kaggle_username
        os.environ['KAGGLE_KEY']=kaggle_key

        os.system('kaggle competitions download -c rsna-pneumonia-detection-challenge')

        # unzipping takes a few minutes
        print('Unzipping dataset files...')
        os.system('unzip -q -o stage_1_test_images.zip -d test_images')
        os.system('unzip -q -o stage_1_train_images.zip -d train_images')
        os.system('unzip -q -o stage_1_train_labels.csv.zip')
        os.system('unzip -q -o stage_1_detailed_class_info.csv.zip')

        # rename csv files
        os.system('mv stage_1_train_labels.csv train_labels.csv')
        os.system('mv stage_1_detailed_class_info.csv detailed_class_labels.csv')

        # remove zip files
        print('Removing zip files...')
        os.system('rm *.zip')
        os.system('rm stage_1_sample_submission.csv')
        os.system('rm GCP%20Credits%20Request%20Link%20-%20RSNA.txt')
        
        os.system('chmod -R 777 {}'.format(DATASET_ABS_PATH))
        
        print('Dataset downloaded successfully.\n')
    else:
        print('Dataset already exists.\n')
        
    os.system('ls -l {}'.format(DATASET_ABS_PATH))
    os.chdir('../')


def download_mask_rcnn():
    """
    CLone the Mask R-CNN library
    """

    if not os.path.exists('Mask_RCNN'):
        print('Cloning Mask R-CNN...')
        os.system('git clone https://github.com/matterport/Mask_RCNN.git')
        print('Finishing cloning R-CNN')
    else:
        print('Local copy of Mask R-CNN already exists.')


def get_image_paths(dicom_dir):
    """
    Returns a list of all image paths in the directory
    :param dicom_dir: dicom image directory
    """

    paths = glob.glob(dicom_dir + '/' + '*.dcm')
    return list(set(paths))


def parse_dataset(dicom_dir, dataframe):
    """
    Returns a list of all image paths as well as that parsed
    dataset as annotations
    :param dicom_dir: dicom image directory
    :param dataframe: train_label dataset
    """

    paths = get_image_paths(dicom_dir)
    annotations = {fp: [] for fp in paths}
    for index, row in dataframe.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')
        annotations[fp].append(row)

    return paths, annotations


def parse_data(dicom_dir, df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the
    data into the following nested dictionary:

      parsed = {

        'patientId-00': {
            'id': 'patientId-00',
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        },
        'patientId-01': {
            'id': 'patientId-01',
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        }, ...

      }
    """
    # --- Define lambda to extract coords in list [x, y, width, height]
    extract_box = lambda row: [row['x'], row['y'], row['width'], row['height']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'id': pid,
                'dicom': dicom_dir + '/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


def draw_dicom(data, plt_pos=None, figsize=None):
    """
    Method to draw single patient with bounding box(es) if present
    """
    # --- Open DICOM file
    dicom = pydicom.read_file(data['dicom'])
    image = dicom.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    image = np.stack([image] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        image = overlay_box(image=image, box=box, rgb=rgb, stroke=6)

    if figsize is not None:
        pylab.figure(figsize=figsize)

    if plt_pos is not None:
        pylab.subplot(plt_pos)
    pylab.imshow(image, cmap=pylab.cm.gist_gray)
    pylab.axis('off')


def overlay_box(image, box, rgb, stroke=1):
    """
    Method to overlay single box on image
    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]

    # --- Extract coordinates
    x1, y1, width, height = box
    y2 = y1 + height
    x2 = x1 + width

    image[y1:y1 + stroke, x1:x2] = rgb
    image[y2:y2 + stroke, x1:x2] = rgb
    image[y1:y2, x1:x1 + stroke] = rgb
    image[y1:y2, x2:x2 + stroke] = rgb

    return image


def splash_class_ids(class_ids):
    """
    Color bounding boxes of detections
    """

    colors = []
    # rgb = np.floor(np.random.rand(3) * 256).astype('int')
    rgb = np.random.rand(3)
    for id in class_ids:
        if id == 1:
            colors.append(tuple(rgb))

    return colors


def detect(model, dicom_dir, config, csv_path='detections.csv', orig_size=1024):
    """
    Make predictions on test 
    """

    assert config != None, 'Inference configuration is required'

    print('Detecting pnuemonia in test dataset...')

    # Assume square image
    resize_factor = orig_size / config.IMAGE_SHAPE[0]

    with open(csv_path, 'w') as file:
        for image_id in tqdm(dicom_dir):
            dicom = pydicom.read_file(image_id)
            image = dicom.pixel_array

            # Convert to RGB for consistency if in grayscale
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            result = results[0]

            out_str = ''
            out_str += patient_id
            assert(len(result['rois']) == len(result['class_ids']) == len(result['scores']))

            if len(result['rois']) == 0:
                pass
            else:
                num_instances = len(result['rois'])
                out_str += ','
                for i in range(num_instances):
                    out_str += ' '
                    out_str += str(round(result['scores'][i], 2))
                    out_str += ' '

                    # x, y, width, height
                    x = result['rois'][i][1]
                    y = result['rois'][i][0]
                    width = result['rois'][i][3] - x
                    height = result['rois'][i][2] - y
                    bbox_str = "{} {} {} {}".format(x*resize_factor, y*resize_factor, 
                        width*resize_factor, height* resize_factor)

                    out_str += bbox_str

            file.write(out_str+'\n')

    print('Detections have been writen to {}'.format(csv_path))


def load_last_model(model_dir, config):
    """
    Load model of last training
    """

    assert model_dir != None, 'Provide a model directory path'

    dir_names = next(os.walk(model_dir))[1]
    key = config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    
    if not dir_names:
        import errno
        raise FileNoteFoundError(errno.ENOENT, 'Could not find model directory under {}'.format(model_dir))
        
    fps = []
    # Pick last directory
    for d in dir_names:
        dir_name = os.path.join(model_dir, d)
        # Find last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            print('No weight files in {}'.format(dir_name))
        else: 
            checkpoint = os.path.join(dir_name, checkpoints[-1])
            fps.append(checkpoint)
    
    weights_path = sorted(fps)[-1]

    return weights_path

