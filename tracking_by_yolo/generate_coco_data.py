# *_* : coding: utf-8 *_*

'''
datasets process for object detection project.
for convert customer dataset format to coco data format,
'''

import traceback
import argparse
import datetime
import json
import cv2
import os

import csv
import numpy as np

__CLASS__ = ['__background__', 'lpr']   # class dictionary, background must be in first index.

def argparser():
    parser = argparse.ArgumentParser("define argument parser for pycococreator!")
    # parser.add_argument("-r", "--root_path", default="/home/andy/workspace/ccpd_300x300", help="path of root directory")
    parser.add_argument("-r", "--root_path", default="../data/fly_coco", help="path of root directory")
    parser.add_argument("-p", "--phase_folder", default=["coco"], help="datasets path of [train, val, test]")
    parser.add_argument("-po", "--have_points", default=True, help="if have points we will deal it!")
    parser.add_argument("--kp", "--keypoint", default="pose.npy", help="read data of keypoints.")
    return parser.parse_args()

def MainProcessing(args):
    csv_name = '../data/video_CS_20201031_h_0_to_h_13/orientation/video_CS_20201031_h_0_to_h_13_552_713_239_447_4_pose.csv'
    pose_npy_name = csv_name[:-4] + '.npy'
    pose = np.load(pose_npy_name)

    '''main process source code.'''
    annotations = {}    # annotations dictionary, which will dump to json format file.
    root_path = args.root_path # root_path = "../data/fly_coco"
    phase_folder = args.phase_folder # phase_folder = ["coco"]
    #   have_points = True

    # coco annotations info.
    annotations["info"] = {
        "description": "fly dataset format convert to COCO format",
        "url": "https://github.com/lengyuner/fly_video_analysis",
        "version": "0.1",
        "year": 2020,
        "contributor": "Yun-Er Leng",
        "date_created": "2020/12/22"
    }
    # annotations["info"] = {
    #     "description": "customer dataset format convert to COCO format",
    #     "url": "http://cocodataset.org",
    #     "version": "1.0",
    #     "year": 2019,
    #     "contributor": "andy.wei",
    #     "date_created": "2019/01/24"
    # }

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
                "id": cls, #这里的id是怎么定义的，怎么更换成图片的名字
                "name": clsname
            }
        )
        for catdict in annotations["categories"]:
            if "lpr" == catdict["name"] and args.have_points:
                # catdict["keypoints"] = ["top_left", "top_right", "bottom_right", "bottom_left"]
                catdict["keypoints"] = ["Head", "Center", "Tail", "LWing", "RWing"]
                catdict["skeleton"] = [[]]

    for phase in phase_folder:
        # phase = phase_folder[0]
        annotations["images"] = []
        annotations["annotations"] = []
        label_path = os.path.join(root_path, phase+".txt")
        filename_mapping_path = os.path.join(root_path, phase + "_" + "filename" + "_" + "mapping" + ".txt")
        images_folder = os.path.join(root_path, phase)

        # 原来的代码真的是。。。
        # 居然丧心病狂的把所有信息先写到了图片名字里
        # 然后再利用这个函数存在txt里
        fd = open(label_path, "w")
        for f in os.listdir(images_folder):
            # a=os.listdir(images_folder)
            # f=a[0]
            # f=pic_name
            # ff = os.path.join(images_folder, f)
            infos = f.split("-")
            # index_frame = f.split("_")[0]
            pbs = []

            # if len(infos) != 7:#TODO（JZ）
            #     assert ("Error!")

            for info in infos:
                if info:
                    pbs.append(info)
            index_frame = int(pbs[0])


            bboxtemp = pbs[1].split("_")
            bbox = [bboxtemp[0], bboxtemp[2], bboxtemp[1], bboxtemp[3]]

            # x1 = bbox[0]
            # y1 = bbox[1]
            # bw = bbox[2] - bbox[0]
            # bh = bbox[3] - bbox[1]

            # pointstemp = pbs[3].split("_")
            # points = pointstemp[0].split("&") + pointstemp[1].split("&") + pointstemp[2].split("&") + pointstemp[
            #     3].split("&")

            bbox = [int(b) for b in bbox]
            # points = [int(p) for p in points]
            x1 = bbox[0]
            y1 = bbox[1]
            if int(index_frame) <= len(pose):
                points = pose[index_frame, 3:13]
                points = [int(p) for p in points]
            else:
                assert ("Error!")

            for K_1 in range(len(points)):
                if K_1 % 2 == 0:
                    points[K_1] = points[K_1] - x1
                else:
                    points[K_1] = points[K_1] - y1



            line = f + " " + str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]) + " " \
                   + str(points[0]) + "," + str(points[1]) + "," + str(points[2]) + "," + str(points[3]) + "," \
                   + str(points[4]) + "," + str(points[5]) + "," + str(points[6]) + "," + str(points[7]) + "," \
                   + str(points[8]) + "," + str(points[9])  \
                   + " " + "0"
            fd.write(line+"\n")
        fd.close()

        if os.path.isfile(label_path) and os.path.exists(images_folder):
            print("convert datasets {} to coco format!".format(phase))
            fd = open(label_path, "r")
            fd_w = open(filename_mapping_path, "w")
            step = 0
            for id, line in enumerate(fd.readlines()): #    line=fd.readlines()[0]
                if line:
                    label_info = line.split()

                    image_name = label_info[0]
                    bbox = [int(x) for x in label_info[1].split(",")]
                    cls = int(label_info[-1])

                    filename = os.path.join(images_folder, image_name)
                    img = cv2.imread(filename)
                    height, width, _ = img.shape
                    x1 = bbox[0]
                    y1 = bbox[1]
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]

                    # coco annotations images.
                    index_frame = int(image_name.split("-")[0])
                    # file_name = 'fly_' + phase + '_' + str(id).zfill(12) + '.jpg'
                    file_name = 'fly_' + phase + '_' + str(index_frame).zfill(12) + '.jpg'
                    newfilename = os.path.join(images_folder, file_name)
                    os.rename(filename, newfilename)

                    filename_mapping = file_name + " " + image_name + "\n"
                    fd_w.write(filename_mapping)

                    annotations["images"].append(
                        {
                            "license": 1,
                            "file_name": file_name,
                            "coco_url": "",
                            "height": height,
                            "width": width,
                            "date_captured": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "flickr_url": "",
                            "id": id
                        }
                    )
                    # coco annotations annotations.
                    annotations["annotations"].append(
                        {
                            "id": id,
                            "image_id": id,
                            "category_id": cls+1,
                            "segmentation": [[]],
                            "area": bw*bh,
                            "bbox": [x1, y1, bw, bh],
                            "iscrowd": 0,
                        }
                    )
                    if args.have_points:
                        v = 2
                        catdict = annotations["annotations"][id]
                        if "lpr" == __CLASS__[catdict["category_id"]]:
                            points = [int(p) for p in label_info[2].split(",")]
                            # x1 = bbox[0]
                            # y1 = bbox[1]
                            catdict["keypoints"] = [points[0], points[1], v, points[2], points[3], v, \
                                                    points[4], points[5], v, points[6], points[7], v,
                                                    points[8], points[9], v]
                            catdict["num_keypoints"] = 5

                    step += 1
                    if step % 100 == 0:
                        print("processing {} ...".format(step))
            fd.close()
            fd_w.close()
        else:
            print("WARNNING: file path incomplete, please check!")

        json_path = os.path.join(root_path, phase+".json")
        with open(json_path, "w") as f:
            json.dump(annotations, f)


if __name__ == "__main__":
    print("begining to convert customer format to coco format!")
    args = argparser()
    try:
        MainProcessing(args)
    except Exception as e:
        traceback.print_exc()
    print("successful to convert customer format to coco format")
