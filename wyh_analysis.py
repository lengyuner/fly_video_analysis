
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt




# hours =
# threshold = 20
# fps = 100
# K_0 = 0
# num_frame = int(fps * 60 * 60 *hours)
#
# position = []
# videoCapture = cv2.VideoCapture(video_name)
# for K_0 in range(num_frame):
# # while True:
#     success, frame = videoCapture.read()
#     if success:
#         plt.figure()
#         plt.imshow(frame)


# from egg_position import get_position_by_background


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
    # y_max, x_max, n_cha = frame_1.shape
    # plt.figur
    # background)
    # # plt.imshow(frame_1)
    frame_clean = background - frame_1
    # plt.imshow(frame_clean)
    # frame_new = background - frame_blured
    frame_blured = cv2.medianBlur(frame_clean, 3)
    # frame_bi = cv2.cvtColor(frame_blured, cv2.COLOR_RGB2GRAY)
    # # plt.imshow(frame_bi,'gray')
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
    if 50 < area < 400:
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

    # if x1 < 546:
    #     result = result * 1
    # else:
    #     result = result * 0

    return result

#
# video_name = '../tracking/wyh/20210107_155852_KR0190040036.avi'
# hours = 1 / 60
# threshold = 20
# fps = 100
# background_interval = 1

# FPS = videoCapture.get(cv2.CAP_PROP_FPS)  100

# plt.imshow(frame)

def get_position_by_background(video_name, hours=1/60, threshold=20, fps=100,
                               background_interval=1):

    background = get_background(video_name, hours, fps, background_interval)

    position = []
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

            if K_1 != 7:
                print('Error: frame ', K_0, 'has ',K_1, 'flies')
                print(stats)

        if K_0 % (fps * 60) == 0:
            end_time = time.time()
            print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('K_0:   ', K_0)
    videoCapture.release()
    print('The video :', video_name, ' has', num_frame, )
    return position

#
# def get_position_by_threshold(video_name, hours=1, threshold=50, fps=100):
#     K_0 = 0
#     num_frame = int(fps * 60 * 60 * hours)
#
#     position = []
#     videoCapture = cv2.VideoCapture(video_name)
#     # num_frame =2920
#     for K_0 in range(num_frame):
#         # while True:
#         success, frame = videoCapture.read()
#         if success:
#             x1 = 369
#             x2 = 457
#             y1 = 22
#             y2 = 408
#             frame_1 = np.copy(frame[y1:y2, x1:x2, :])
#             # !!!!!!!!!!!!!!!!!!!!!!!!!!
#             # print(frame_1.shape)
#             # plt.imshow(frame_1)
#             frame_blured = cv2.medianBlur(frame_1, 3)
#             frame_bi = cv2.cvtColor(frame_blured, cv2.COLOR_RGB2GRAY)
#             # plt.imshow(frame_bi,'gray')
#
#             frame_bi[frame_bi < threshold] = 0
#             frame_bi[frame_bi >= threshold] = 255
#             # frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             #                                  cv2.THRESH_BINARY, 25, 5)
#             frame_3_body = 255 - frame_bi.astype(np.uint8)
#             # plt.imshow(frame_3_body,'gray')
#             ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_3_body, connectivity=4)
#             K_1 = 0
#             for i, stat in enumerate(stats):
#                 if stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50:  # and stat[4] > 50:
#                     x1, y1, h, w, area = stat
#                     position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])
#                     K_1 += 1
#             if K_1 != 1:
#                 print(K_0)
#                 print(stats)
#                 # plt.figure()
#                 # plt.imshow(frame_1)
#             K_0 += 1
#             if K_0 % (25 * 100) == 0:
#                 print('K_0:   ', K_0)
#         else:
#             break
#
#     if K_0 >= num_frame - 10:
#         print('Completed. Congratulations!')
#     videoCapture.release()
#     return position
#


video_name = '../tracking/wyh/20210107_155852_KR0190040036.avi'
position = get_position_by_background(video_name, hours=1/60, threshold=20, fps=100,
                               background_interval=1)




position_np = np.asarray(position)
np.save('adfadfasdfasdf.npy',position_np)



# print(len(position))
# for K in range(39080,len(position)):
#     print(position[K])


# temp = get_position_by_threshold(video_name, hours=1, threshold=40, fps=100)





# position_np[position_np[:,2]==569]







