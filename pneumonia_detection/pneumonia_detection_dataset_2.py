import sys
sys.path.append('Mask_RCNN')

import pydicom
import numpy as np
import cv2

from mrcnn import utils


class PneumoniaDetectionDataset(utils.Dataset):
	"""
	Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """
	
	def __init__(self, dataset, orig_height, orig_width):
		super().__init__(self)

		# Add classes
		self.add_class('pneumonia', 1, 'Lung Opacity')

		# Add images
		for data in dataset:
			image_id = data['id']
			path = data['dicom']
			annotations = data['boxes']
			target = data['label']

			self.add_image('pneumonia', image_id=image_id, path=path, annotations=annotations, 
				orig_height=orig_height, orig_width=orig_width, target=target)


	def load_image(self, image_id):
		"""
		Load a dicom image with `image_id`
		"""

		info = self.image_info[image_id]
		path = info['path']
		dicom = pydicom.read_file(path)
		image = dicom.pixel_array

		# Convert to RGB if in grayscale
		if len(image.shape) != 3 or image.shape[2] != 3:
			image = np.stack([image] * 3, -1)

		return image


	def image_reference(self, image_id):
		"""Return the path of the dicom image."""

		info = self.image_info[image_id]
		return info['path']


	def load_mask(self, image_id):
		"""
		Generate instance masks for an image.
       	Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

		info = self.image_info[image_id]
		annotations = info['annotations']
		count = len(annotations)

		if count == 0:
			mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
			class_ids = np.zeros((1,), dtype=np.int32)
		else:
			mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
			class_ids = np.zeros((count,), dtype=np.int32)

			i = 0
			for ann in annotations:
				if i == 0:
					pass
				else:
					i += 1

				if info['target'] == 1:
					x = int(ann[0])
					y = int(ann[1])
					width = int(ann[2])
					height = int(ann[3])
					mask_instance = mask[:, :, i].copy()
					cv2.rectangle(mask_instance, (x, y), (x+width, y+height), 255, -1)
					mask[:, :, i] = mask_instance
					class_ids[i] = 1

		return mask.astype(np.bool), class_ids.astype(np.int32)