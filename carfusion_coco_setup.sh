python download_carfusion.py
python carfusion2coco.py --path_dir ./datasets/carfusion/train/ --label_dir gt --image_dir image_jpg --output_dir . --output_filename car_keypoints_test.json
python carfusion2coco.py --path_dir ./datasets/carfusion/test/ --label_dir gt --image_dir image_jpg --output_dir . --output_filename car_keypoints_test.json
