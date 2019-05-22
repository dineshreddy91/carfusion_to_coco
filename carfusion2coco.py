#!/usr/bin/env python3

#import pims
#import scipy.io
import argparse
import os
#from IPython import embed
import json
import time
#from PIL import Image
import numpy as np

#from scipy.misc import imsave
#import cv2
from shapely.geometry import Polygon

def num(s):
    try:
        return int(s)
    except:
        return int(float(s))


def getAnnotation(instance):

    width, height = 1920, 1080

    valid_2 = instance[:, 2] == 1
    valid = instance[:, 2] == 2
    #print(instance)

    visible = np.logical_or(valid, valid_2)
    num_keypoints = int(np.sum(visible))

    keypoints = np.zeros((14,3), dtype=np.int32)
    try:
        hull = Polygon([(x[0], x[1]) for x in instance[visible, :2]]).convex_hull
        frame = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        hull = hull.intersection(frame).convex_hull

        bbox = hull.bounds
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x_o = max(bbox[0]-(w/10),0)
        y_o = max(bbox[1]-(h/10),0)
        x_i = min(x_o+(w/4)+w,width)
        y_i = min(y_o+(h/4)+h,height)
        bbox = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]

        segmentation = list(hull.convex_hull.exterior.coords)[:-1]
        segmentation = [[int(x[0]), int(x[1])] for x in segmentation]

        keypoints[:, :] = instance[:, :]
        #print(instance, num_keypoints)

    except:
        #print("failed finding any keypoint")
        #print(instance, num_keypoints)
        #asasas#print(instance, num_keypoints)
        bbox = [0, 0, 0, 0]
        segmentation = []

    keypoints = np.reshape(keypoints, (42,))
    keypoints = keypoints.tolist()
    keypoints = [int(x) for x in keypoints]

    seg = []
    for s in segmentation:
        seg.append(s[0])
        seg.append(s[1])


    return bbox, seg, keypoints, num_keypoints



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_filename", required=True)
    args = parser.parse_args()


    label_dir = args.label_dir
    path_dir = args.path_dir

    #if not os.path.exists(label_dir):
    #    print("Can not locate label folder: " + label_dir)
    #    quit()


    data = {}

    data["info"] = {
            'url': "https://www.andrew.cmu.edu/user/dnarapur/",
            'year': 2018,
            'date_created': time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                time.localtime()),
            'description': "This is a keypoint dataset for object detection.",
            'version': '1.0',
            'contributor': 'CMU'}

    data["categories"] = [{'name': 'car',
        'id': 1,
        'skeleton': [[0,2],
                     [1,3],
                     [0,1],
                     [2,3],
                     [9,11],
                     [10,12],
                     [9,10],
                     [11,12],
                     [4,0],
                     [4,9],
                     [4,5],
                     [5,1],
                     [5,10],
                     [6,2],
                     [6,11],
                     [7,3],
                     [7,12],
                     [6,7]],# TODO
        'supercategory': 'car',
        'keypoints': [str(x) for x in range(14)]}]

    data["licenses"] = [{'id': 1,
                'name': "unknown",
                'url': "unknown"}]


    obj_id = 0
    # expect sub-folder for subsets
    data["images"] = []
    data["annotations"] = []
    json_name = args.output_filename
    loop=0
    count_images=0
    for sub_dir in os.listdir(path_dir):
        loop= loop+1
        im_dir = os.path.join(
                sub_dir,
                args.image_dir)

        sub_dir = path_dir + sub_dir + '/gt/'#os.path.join(path_dir, sub_dir)
        print(sub_dir)
        #sub_dir = os.path.join(sub_dir, '/gt/')
        if not os.path.isdir(sub_dir):
            continue


        # loop through all annotation file inside sub_dir
        # loop through all annotation file inside sub_dir
        #print(sub_dir)
        #asas
        for file_name in os.listdir(sub_dir):
            count_images =count_images+1
            file_str = file_name.split('.')[0]
            vid_str, id_str  = file_str.split('_')
            frame_id = int(id_str)
            video_id = int(vid_str)
            image_id = int(loop*1e8+video_id*1e5+frame_id)

            image_name = os.path.join(
                    im_dir,
                    "{}.jpg".format(file_str))

            width, height = 1920, 1080
	
            data["images"].append({'flickr_url': "unknown",
                'coco_url': "unknown",
                'file_name': image_name,
                'id': image_id,
                'license':1,
		#'has_visible_keypoints':True,
                'date_captured': "unknown",
                'width': width,
                'height': height})

            with open(os.path.join(sub_dir, file_name.split('.')[0]+'.txt')) as f:
                keypoints = f.readlines()
                keypoints = [s.split(',') for s in keypoints]
                keypoints = [list(map(num, s)) for s in keypoints]

            instances = {}

            for keypoint in keypoints:
                if keypoint[3] not in instances:
                    instances[keypoint[3]] = np.zeros((14, 3), dtype=np.int32)
                instances[keypoint[3]][keypoint[2]-1,0] = keypoint[0]
                instances[keypoint[3]][keypoint[2]-1,1] = keypoint[1]
                if keypoint[4] == 2:
                    instances[keypoint[3]][keypoint[2]-1,2] = 1
                elif keypoint[4] == 1:
                    instances[keypoint[3]][keypoint[2]-1,2] = 2#keypoint[4]
                elif keypoint[4] == 3:
                    instances[keypoint[3]][keypoint[2]-1,2] = 2#keypoint[4]

                if keypoint[0] <= 0 or keypoint[1] > height or keypoint[1] <= 0 or keypoint[0] > width:
                    #print(keypoint[1])

                    instances[keypoint[3]][keypoint[2]-1,2] = 0#keypoint[4]\
                    #print(instances[keypoint[3]])

            #print(instances)
            #asas

            for instance in instances.values():

                bbox, segmentation, keypoints, num_keypoints = getAnnotation(instance)
                #print(keypoints)

                if num_keypoints == 0:
                    continue

                data["annotations"].append({
                    'image_id': image_id,
                    'category_id': 1,
                    'iscrowd': 0,
                    #'has_visible_keypoints': True,
                    'id': obj_id,
                    'area': bbox[2]*bbox[3],
                    'bbox': bbox,
                    'num_keypoints': num_keypoints,
                    'keypoints': keypoints,
                    'segmentation': [segmentation]})

                obj_id += 1
    #json_name = 'carfusion.json'
    json_str = json.dumps(data)

    print(json_name,count_images)
    ann_file = os.path.join(args.output_dir, json_name)
    if not os.path.exists(args.output_dir):
         os.mkdir(args.output_dir)
    with open(ann_file, 'w') as f:
         f.write(json_str)




