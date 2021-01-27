

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def if_path_exist(to_path):
    if os.path.isdir(to_path):
        if len(os.listdir(to_path)) > 0:
            print('The file fold is not empty and will stop:\n', to_path)
            return False
    else:
        print('The file fold is not exist and will be created.\n', to_path, '\n')
        os.makedirs(to_path)
    return True


def get_video_size(video_name):
    videoCapture = cv2.VideoCapture(video_name)
    success, frame_temp = videoCapture.read()
    if success:
        y_max, x_max, n_cha = frame_temp.shape
        videoCapture.release()
        return y_max, x_max, n_cha
    else:
        print('No video named:', video_name, '\n')
        return None


def get_background(video_name, hours=1 / 60, fps=100, background_interval=1):
    y_max, x_max, n_cha = get_video_size(video_name)

    background = np.zeros([y_max, x_max, n_cha])
    videoCapture = cv2.VideoCapture(video_name)

    num_frame = int(fps * 60 * 60 * hours)
    #   num_frame = 180000

    start_time = time.time()
    for K_0 in range(num_frame):
        success, frame_temp = videoCapture.read()
        # frame_temp = frame_temp[6:230, 144:368, 3]
        # if success == False and K_0 == 1:
        #     return None
        if K_0 % background_interval == 0:
            background = np.maximum(frame_temp, background)
            # if np.max(frame_temp) < 140:
            #     background = np.maximum(frame_temp, background)

        if K_0 % (background_interval * 100) == 0:
            end_time = time.time()
            # print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('background computing:    K_0:   ', K_0)

    videoCapture.release()
    background = background.astype('uint8')
    # plt.figure()
    # plt.imshow(background)
    return background


def process_picture(frame, background, threshold):
    frame_1 = np.copy(frame)
    # plt.figure()
    # plt.imshow(frame_1)
    frame_clean = background - frame_1
    # plt.imshow(frame_clean)
    frame_blured = cv2.medianBlur(frame_clean, 3)
    # plt.imshow(frame_bi,'gray')
    #
    # frame_bi[frame_bi < threshold] = 0
    # frame_bi[frame_bi >= threshold] = 255
    # # frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # #                                  cv2.THRESH_BINARY, 25, 5)
    # # plt.imshow(frame_3_body,'gray')
    # frame_3_body = 255 - frame_bi.astype(np.uint8)

    # frame_new = background - frame_blured
    #   plt.imshow(frame_blured)

    frame_grey = cv2.cvtColor(frame_blured, cv2.COLOR_RGB2GRAY)
    frame_grey[frame_grey >= 240] = 0
    frame_grey[frame_grey < threshold] = 0
    frame_grey[frame_grey >= threshold] = 255
    # frame_bi = cv2.adaptiveThreshold(frame_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  cv2.THRESH_BINARY, 11, 20)
    #   plt.imshow(frame_grey)
    return frame_grey


def if_is_fly(stat):
    x1, y1, weight, height, area = stat
    # if stat[4] > 50 and stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50:
    #     x1, y1, w, h, area = stat
    result = 1
    if 45 < area < 500:
        result = result * 1
    else:
        result = result * 0

    if 3 < weight < 50:
        result = result * 1
    else:
        result = result * 0

    if 3 < height < 50:
        result = result * 1
    else:
        result = result * 0

    # if 3 < height < 50:
    #     result = result * 1
    # else:
    #     result = result * 0

    # if x1 < 451:    # TODO
    #     result = result * 1
    # else:
    #     result = result * 0

    return result


def get_position_by_background(video_name, hours=1/60, threshold=40, fps=100,
                               background_interval=1):

    background = get_background(video_name, hours, fps, background_interval)

    position = []

    error_info = []

    start_time = time.time()
    videoCapture = cv2.VideoCapture(video_name)
    num_frame = int(fps * 60 * 60 * hours)
    # num_frame = 4090
    for K_0 in range(num_frame):
        success, frame = videoCapture.read()
        if success:
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

            if K_1 != 7:    # TODO
                print('Error: frame ', K_0, 'has ', K_1, 'flies')
                # print(stats)
                error_info.append([K_0, stats, frame, frame_grey])
        # else:
        #     return position, background, error_info

        if K_0 % (fps * 60) == 0:
            end_time = time.time()
            print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('K_0:   ', K_0)
    videoCapture.release()
    print('The video :', video_name, ' has', num_frame, )
    return position, background, error_info



def process_picture_for_test(frame, background, threshold):
    frame_1 = np.copy(frame)
    plt.figure()
    K_1 = 1
    plt.subplot(3, 2, K_1)
    K_1 += 1
    plt.imshow(frame_1)

    plt.subplot(3, 2, K_1)
    K_1 += 1
    plt.imshow(background)

    frame_clean = background - frame_1
    plt.subplot(3, 2, K_1)
    K_1 += 1
    plt.imshow(frame_clean)

    frame_blured = cv2.medianBlur(frame_clean, 3)
    plt.subplot(3, 2, K_1)
    K_1 += 1
    plt.imshow(frame_blured)

    frame_grey = cv2.cvtColor(frame_blured, cv2.COLOR_RGB2GRAY)
    frame_grey[frame_grey >= 240] = 0
    frame_grey[frame_grey < threshold] = 0
    frame_grey[frame_grey >= threshold] = 255
    plt.subplot(3, 2, K_1)
    K_1 += 1
    plt.imshow(frame_grey)

    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_grey, connectivity=4)
    # print(stats)
    K_1 = 0
    for i, stat in enumerate(stats):
        # stat=stats[1]
        if if_is_fly(stat):
            K_1 += 1
            x1, y1, w, h, area = stat
            print(stat)
    return None


# hours = 1 / 60
# threshold = 20
# fps = 100
# background_interval = 100
# FPS = videoCapture.get(cv2.CAP_PROP_FPS)  100

video_name = '../tracking/wyh/20210113_145522_A_avi_c.avi'
position, background, error = get_position_by_background(video_name, hours=1/60, threshold=40, fps=100,
                               background_interval=1000)

position_np = np.asarray(position)
# print(position_np.shape)


def sort_position(position_disorder):
    print(position_disorder.shape)

    position_sorted = np.ones_like(position_disorder)
    # position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])

    for K_2 in range(int(len(position_disorder)/7)):
        position_temp = position_disorder[K_2:(K_2+7)]
        position_temp = position_temp[np.argsort(position_temp[:,0]), :]
        # pose = pose_disorder[np.argsort(pose_disorder[:, 0]), :]
        position_sorted[K_2:(K_2+7)] = position_temp

    return position_sorted


position_np = sort_position(position_np)

print(position_np.shape)

npy_name = video_name[:-4] + '_position' + '_'+time.strftime("%Y%m%d_%H%M") + '.npy'
np.save(npy_name, position_np)


# K_0, stats, frame, frame_grey = error[0]
# threshold = 40
# process_picture_for_test(frame, background, threshold)


# plt.imshow(frame_temp)
#
# plt.imsave('fly.jpg',frame_temp)



def get_coco_train_data(video_name, hours=1/60, threshold=40, fps=100,
                               background_interval=1000):

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
        if success and K_0 % (fps*20) == 0:
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

            if K_1 != 7:    # TODO
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
    return position # , background, error_info


video_name = '../tracking/wyh/20210113_145522_A_avi_c.avi'
position = get_coco_train_data(video_name, hours=1, threshold=40, fps=100,
                               background_interval=1000)
position_np = np.asarray(position)
print(position_np.shape)



def sort_position(position_np):
    print(position_np.shape)

    position_sorted = np.ones_like(position_np)
    # position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])

    for K_2 in range(int(len(position_np)/7)):
        position_temp = position_np[K_2:(K_2+7)]
        position_temp = position_temp[np.argsort(position_temp[:,0]), :]
        # pose = pose_disorder[np.argsort(pose_disorder[:, 0]), :]
        position_sorted[K_2:(K_2+7)] = position_temp

    return position_sorted

position_np = sort_position(position_np)

print(position_np.shape)

import csv






















