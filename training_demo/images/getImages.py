import os
import boto3
from tqdm import tqdm

XML_DIR = 'pool'
IMAGE_DIR = XML_DIR

xmls = [file for file in os.listdir(XML_DIR) if '.xml' in file]
images = {file for file in os.listdir(IMAGE_DIR) if '.png' in file}

session = boto3.Session('minio', 'minio123')
s3 = session.client('s3', 
  endpoint_url='http://minio:9000', 
  config=boto3.session.Config(signature_version='s3v4')
)

for xml in tqdm(xmls, total=len(xmls)):
    image_id = xml.split('.')[0]
    if image_id + '.png' in images:
        # already downloaded
        continue
    # Else download image
    image_name = image_id + '.png'
    s3.download_file('finlab-bucket', image_name, os.path.join(IMAGE_DIR, image_name))

