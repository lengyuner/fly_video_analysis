import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from wyh_analysis import if_path_exist, get_background, process_picture, if_is_fly


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
                        background_interval=1000, save_interval=2000):
    to_path = 'H:/data/fly_coco_yolo'
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
                img_name = str(K_0) + '.jpg'
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


def npy2csv():
    import csv
    video_name = '../tracking/wyh/20210113_145522_A_avi_c.avi'
    position = get_coco_train_data(video_name, hours=1 / 60, threshold=40, fps=100,
                                   background_interval=100, save_interval=2000)

    position_np = np.asarray(position)
    print(position_np.shape)
    # np.savetxt("new.csv", position_np, delimiter=',', fmt='%.1f')

    position_np = sort_position(position_np)
    # position_np = position_np.astype(np.uint8)
    np.savetxt("new2.csv", position_np, delimiter=',', fmt='%.1f')

    print(position_np.shape)

    position_np[0]

    bbox_fly = []
    # K_4 = 0
    annotations = {}
    cls = 1
    for K_3 in range(len(position_np)):
        # bbox_1 = []
        centroid_x, centroid_y, K_0, K_1, x1, y1, bh, bw, area = position_np[K_3]
        id = K_3
        img_id = K_0
        annotations["annotations"].append(
            {
                "id": id,
                "image_id": img_id,
                "category_id": cls + 1,
                "segmentation": [[]],
                "area": bw * bh,
                "bbox": [x1, y1, bw, bh],
                "iscrowd": 0,
            }
        )

    return None
