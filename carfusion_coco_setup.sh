## download the data from the carfusion website

python download_carfusion.py

## scripts to convert hte data from carfusion to coco format
python carfusion2coco.py --path_dir ./datasets/carfusion/train/ --label_dir gt --image_dir images_jpg --output_dir ./datasets/carfusion/annotations/ --output_filename car_keypoints_train.json
python carfusion2coco.py --path_dir ./datasets/carfusion/test/ --label_dir gt --image_dir images_jpg --output_dir  ./datasets/carfusion/annotations/ --output_filename car_keypoints_test.json
