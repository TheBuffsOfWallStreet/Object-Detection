python3 generate_tfrecord.py -x ./training_demo/images/test -l ./training_demo/annotations/label_map.pbtxt -o ./training_demo/annotations/test.record
# python3 generate_tfrecord.py -x ./training_demo/images/train -l ./training_demo/annotations/label_map.pbtxt -o ./training_demo/annotations/train.record
#python3 generate_tfrecord.py -x ./training_demo/images/refined -l ./training_demo/annotations/label_map.pbtxt -o ./training_demo/annotations/refined.record
DATA_DIR=./training_demo/images/pool
LABEL_MAP=./training_demo/annotations/label_map.pbtxt
python3 generate_tfrecord.py -x $DATA_DIR -l $LABEL_MAP -o ./training_demo/annotations/pooled.record
