import time
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

import boto3
from pymongo import MongoClient
from io import BytesIO

session = boto3.Session('minio','minio123')
s3 = session.resource('s3',
  endpoint_url='http://minio:9000',
  config=boto3.session.Config(signature_version='s3v4')
)
bucket = s3.Bucket('finlab-bucket')

user = 'user'
password = 'password'
domain = 'domain.com'
mongo_uri = f'mongodb://{user}:{password}@{domain}:27017/?authSource=users&readPreference=primary&ssl=false'
mongo = MongoClient(mongo_uri)
db = mongo.finlab_beta



PATH_TO_MODEL_DIR = 'models/my_ssd_mobilenet'
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR # + "/checkpoint"

CHECKPOINT = 'ckpt-28'
THRESHOLD = 0.10

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, CHECKPOINT)).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

PATH_TO_LABELS = 'annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

IMAGE_DIR = 'images/test'
IMAGE_PATHS = [os.path.join(IMAGE_DIR, x) for x in os.listdir(IMAGE_DIR) if '.png' in x]
IMAGE_PATHS = IMAGE_PATHS[:2]

def load_image_into_numpy_array(path):
    return np.array(Image.open(path).convert('RGB'))

i = 0
for obj in bucket.objects.all():
    if db.DetectionBoxes.find_one({'_id': obj.key}):
        # Skip already existing images
        continue
    # Get image from boto3
    try:
        im = Image.open(BytesIO(obj.get()['Body'].read()))
        image_np = np.array(im.convert('RGB'))
        im.close()
        i += 1
        if i % 100 == 0:
            print(f'Running inference for {i}: {obj.key}... ')

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        THRESHOLD = 0.15
        detection_threshold_indexes = [i for i,v in enumerate(detections['detection_scores']) if v > THRESHOLD]

        formated_detections = []

        for idx in detection_threshold_indexes:
            m,n,l = image_np.shape
            y_min, x_min, y_max, x_max = detections['detection_boxes'][idx]
            x_min *= n
            x_max *= n
            y_min *= m
            y_max *= m

            formated_detections.append({
                'confidence': float(detections['detection_scores'][idx]),
                'min': [x_min, y_min],
                'max': [x_max, y_max],
            })

        document = {
            '_id': obj.key,
            'detections': formated_detections,
        }
        db.DetectionBoxes.insert_one(document)
    except Exception as e:
        print(f'Exception with {obj.key}')
        print(e)
