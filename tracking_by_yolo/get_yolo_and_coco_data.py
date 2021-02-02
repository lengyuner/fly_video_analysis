


import os
import cv2
import time
import json
import datetime
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt

from wyh_analysis import if_path_exist, get_background, process_picture, if_is_fly

__CLASS__ = ['__background__', 'fly']   # class dictionary, background must be in first index.



def sort_position(position_disorder):
    print(position_disorder.shape)
    # position_disorder = np.copy(position_np)
    position_sorted = np.ones_like(position_disorder)
    # K_2=0
    # position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])

    for K_2 in range(int(len(position_disorder) / 7)):
        position_temp = position_disorder[K_2*7:(K_2*7 + 7)]
        # print(position_temp.shape)
        # position_temp = position_disorder[position_disorder[:,2]==(2000*K_2)]
        position_temp = position_temp[np.argsort(position_temp[:, 0]), :]
        # pose = pose_disorder[np.argsort(pose_disorder[:, 0]), :]
        position_sorted[K_2*7:(K_2*7 + 7)] = position_temp

    return position_sorted


def get_coco_train_data(video_name, hours=1 / 60, threshold=40, fps=100,
                        background_interval=1000, save_interval=2000, to_path='H:/data/fly_coco_yolo'):

    if_path_exist(to_path)

    background = get_background(video_name, hours, fps, background_interval)

    position = []

    error_info = []

    start_time = time.time()
    videoCapture = cv2.VideoCapture(video_name)
    num_frame = int(fps * 60 * 60 * hours)
    # num_frame = 4090
    for K_0 in range(num_frame):
        success, frame = videoCapture.read()
        if success and K_0 % save_interval == 0:
            frame_grey = process_picture(frame, background, threshold)
            # plt.figure()
            # plt.imshow(frame_grey)
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_grey, connectivity=4)
            # print(stats)
            K_1 = 0
            for i, stat in enumerate(stats):
                # stat=stats[1]
                if if_is_fly(stat):
                    K_1 += 1
                    x1, y1, w, h, area = stat

                    position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])
            # print(K_1)
            if K_1 == 7:
                # img_name = str(K_0) + '.jpg'
                phase = 'coco'
                img_name = 'fly_' + phase + '_' + str(K_0).zfill(12) + '.jpg'
                print(os.path.join(to_path, img_name))
                plt.imsave(os.path.join(to_path, img_name), frame)

            if K_1 != 7:  # TODO
                print('Error: frame ', K_0, 'has ', K_1, 'flies')
                # print(stats)
                # error_info.append([K_0, stats, frame, frame_grey])
        # else:
        #     return position, background, error_info

        if K_0 % (fps * 60) == 0:
            end_time = time.time()
            print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('K_0:   ', K_0)
    videoCapture.release()
    print('The video :', video_name, ' has', num_frame, )
    return position  # , background, error_info


def creat_json(position_np, images_folder, to_path):
    # import csv
    # video_name = '../tracking/wyh/20210113_145522_A_avi_c.avi'
    # position = get_coco_train_data(video_name, hours=1 / 2, threshold=40, fps=100,
    #                                background_interval=100, save_interval=1000)
    cls = 1

    annotations = {}

    # coco annotations info.
    annotations["info"] = {
        "description": "wyh fly dataset format convert to COCO format",
        "url": "https://github.com/lengyuner/fly_video_analysis",
        "version": "0.1",
        "year": 2021,
        "contributor": "Yun-Er Leng",
        "date_created": "2021/01/27"
    }

    # coco annotations licenses.
    annotations["licenses"] = [{
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        "id": 1,
        "name": "Apache License 2.0"
    }]

    # coco annotations categories.
    annotations["categories"] = []
    for cls, clsname in enumerate(__CLASS__):
        if clsname == '__background__':
            continue
        annotations["categories"].append(
            {
                "supercategory": "object",
                "id": cls, #这里的id是怎么定义的，怎么更换成图片的名字 class = cls
                "name": clsname
            }
        )
        for catdict in annotations["categories"]:
            if "fly" == catdict["name"]:
                catdict["keypoints"] = []
                catdict["skeleton"] = [[]]


    annotations["images"] = []
    for img_name in os.listdir(images_folder):
        # phase = 'coco'
        # img_name = 'fly_' + phase + '_' + str(K_2).zfill(12) + '.jpg'
        filename = os.path.join(images_folder, img_name)
        img = cv2.imread(filename)
        height, width, _ = img.shape
        # img_id
        img_id = int(img_name[-16:-4])
        annotations["images"].append(
            {
                "license": 1,
                "file_name": img_name,
                "coco_url": "",
                "height": height,
                "width": width,
                "date_captured": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "flickr_url": "",
                "id": img_id
            }
        )

    annotations["annotations"] = []
    for K_3 in range(len(position_np)):
        # bbox_1 = []
        centroid_x, centroid_y, K_0, K_1, x1, y1, bh, bw, area = position_np[K_3]

        ann_id = K_1
        img_id = K_0

        # file_name = 'fly_' + phase + '_' + str(K_0).zfill(12) + '.jpg'
        # filename = os.path.join(to_path, img_name)

        annotations["annotations"].append(
            {
                "id": ann_id + img_id,
                "image_id": img_id,
                "category_id": cls + 1,
                "segmentation": [[]],
                "area": bw * bh,
                "bbox": [x1, y1, bw, bh],
                "iscrowd": 0,
                "keypoints": [],
                "num_keypoints": 0
            }
        )

    phase = 'coco'
    json_path = os.path.join(to_path, phase + ".json")
    print(json_path)

    with open(json_path, "w") as f:
        json.dump(annotations, f)


    return None


    # {
    #   "id": 3966,
    #   "image_id": 3966,
    #   "category_id": 1,
    #   "segmentation": [
    #     []
    #   ],
    #   "area": 1548,
    #   "bbox": [
    #     57,
    #     132,
    #     36,
    #     43
    #   ],
    #   "iscrowd": 0,
    #   "keypoints": [
    #     25,
    #     10,
    #     2,
    #     18,
    #     20,
    #     2,
    #     12,
    #     31,
    #     2,
    #     7,
    #     35,
    #     2,
    #     9,
    #     37,
    #     2
    #   ],
    #   "num_keypoints": 5
    # }


def creat_txt_for_yolov5_tf(position_np, images_folder, label_path):

    fd = open(label_path, "w")
    img_index = 0
    K_5 = 0
    for img_name in os.listdir(images_folder):
        # phase = 'coco'
        # img_name = 'fly_' + phase + '_' + str(K_2).zfill(12) + '.jpg'
        filename = os.path.join(images_folder, img_name)

        img_id = int(img_name[-16:-4])
        line = 'H:/tracking/Yolov5_tf/data/fly/train/' + img_name + ' '
        for K_6 in range(7):
            centroid_x, centroid_y, K_0, K_1, x1, y1, bh, bw, area = position_np[K_5]
            assert K_0 == img_id
            line = line + str(int(x1)) + "," + str(int(y1)) + "," + str(int(bh)) + "," + \
                   str(int(bw)) + "," + str("0") + " "
            K_5 += 1

        fd.write(line + "\n")

    fd.close()

    # img = cv2.imread(filename)


    # for K_5 in range(len(position_np)):
    #
    #     centroid_x, centroid_y, K_0, K_1, x1, y1, bh, bw, area = position_np[K_5]
    #
    #     if K_0 == img_index:


    # line = f + " " + str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]) + " " \
    #        + str(points[0]) + "," + str(points[1]) + "," + str(points[2]) + "," + str(points[3]) + "," \
    #        + str(points[4]) + "," + str(points[5]) + "," + str(points[6]) + "," + str(points[7]) + "," \
    #        + str(points[8]) + "," + str(points[9]) \
    # #        + " " + "0"
    # fd.write(line + "\n")



    # phase = 'coco'
    # json_path = os.path.join(to_path, phase + ".json")
    # print(json_path)
    #
    # with open(json_path, "w") as f:
    #     json.dump(annotations, f)

if __name__ == '__main__':

    video_name = '../tracking/wyh/20210113_145522_A_avi_c.avi'
    img_file = 'H:/data/fly_coco_yolo_4'
    position = get_coco_train_data(video_name, hours=1/2, threshold=40, fps=100,
                        background_interval=1000, save_interval=1000, to_path=img_file)

    # position = get_coco_train_data(video_name, hours=1 / 2, threshold=40, fps=100,
    #                                background_interval=100, save_interval=1000)

    position_np = np.asarray(position)
    print(position_np.shape)
    # np.savetxt("new.csv", position_np, delimiter=',', fmt='%.1f')
    position_np = sort_position(position_np)
    # position_np = position_np.astype(np.uint8)
    # np.savetxt("position1.csv", position_np, delimiter=',', fmt='%.1f')
    np.save('position1.npy',position_np)

    print(position_np.shape)

    # position_np[0]
    to_path = 'H:/data'
    # creat_json(position_np, img_file, to_path)

    # import csv
    # position_np = csv.reader("position.csv")

    # position_np = csv.load(position.csv)
    label_path = 'H:/tracking/Yolov5_tf/data/fly/train.txt'
    creat_txt_for_yolov5_tf(position_np, img_file, label_path)

