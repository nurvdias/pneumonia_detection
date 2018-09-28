import sys
sys.path.append('Mask_RCNN')

from mrcnn.config import Config


class PneumoniaDetectionConfig(Config):
	"""
	Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

	# Config name
	NAME = 'pneumonia_detection'

	# Using Amazon AMI p2 x.large instance provides 1 GPU and 
	# 61GB memory.
	IMAGES_PER_GPU = 2

	# Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
	VALIDATION_STEPS = 100

	# Number of training steps per epoch
	# This doesn't need to match the size of the training set. Tensorboard
	# updates are saved at the end of each epoch, so setting this to a
	# smaller number means getting more frequent TensorBoard updates.
	# Validation stats are also calculated at each epoch end and they
	# might take a while, so don't set this too small to avoid spending
	# a lot of time on validation stats.
	STEPS_PER_EPOCH = 300

	# Number of classification classes (including background)
	NUM_CLASSES = 1 + 1  # background + 1 pneumonia class

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
	DETECTION_MIN_CONFIDENCE = 0.9  # skip detections less than 90%

    # ROIs kept after non-maximum suppression (training and inference)
	POST_NMS_ROIS_TRAINING = 2000
	POST_NMS_ROIS_INFERENCE = 2000

	# Input image resizing
	IMAGE_MIN_DIM = 1024
	IMAGE_MAX_DIM = 1024
    
    # Length of square anchor side in pixels
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
	# RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 200

    # Max number of final detections
	DETECTION_MAX_INSTANCES = 400

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.9
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
	MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
	USE_MINI_MASK = True

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    
	# Number of ROIs per image to feed to classifier/mask heads
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 128


class PneumoniaDetectionInferenceConfig(PneumoniaDetectionConfig):
    # Set batch size to 1 to run one image at a time
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.95  # skip detections less than 95%





