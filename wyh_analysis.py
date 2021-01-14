








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

# threshold=40
def process_picture(frame, background, threshold):
    frame_1 = np.copy(frame)
    # y_max, x_max, n_cha = frame_1.shape
    # plt.figure()
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


def get_position_by_background_for_test(video_name, hours=1 / 60, threshold=40, fps=100,
                               background_interval=1):


    background = get_background(video_name, hours, fps, background_interval)
    plt.figure()
    plt.imshow(background)

    position = []
    error_info = []

    start_time = time.time()
    videoCapture = cv2.VideoCapture(video_name)
    num_frame = int(fps * 60 * 60 * hours)
    # num_frame = 4090

    K_error = 0
    for K_0 in range(num_frame):
        success, frame = videoCapture.read()
        if success:
            frame_grey = process_picture(frame, background, threshold)

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
                K_error += 1
                print('Error: frame ', K_0, 'has ', K_1, 'flies')
                print(stats)
                error_info.append([K_0, stats, frame, frame_grey])
                if K_error < 10:
                    plt.figure()
                    plt.imshow(frame_grey)

            if K_error == 10:
                # break
                return position, background, error_info

        if K_0 % (fps * 60) == 0:
            end_time = time.time()
            print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('K_0:   ', K_0)
    videoCapture.release()
    print('The video :', video_name, ' has', num_frame, )
    return position, background, error_info
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

# threshold  # TODO

# video_name = '../tracking/wyh/20210107_155852_KR0190040036.avi'
video_name = '../tracking/wyh/20210113_145522_A_avi_c.avi'
# #
# video_name = '../tracking/wyh/20210107_155852_KR0190040036_1.avi'
# hours = 1 / 60
# threshold = 20
# fps = 100
# background_interval = 100

# FPS = videoCapture.get(cv2.CAP_PROP_FPS)  100

# plt.imshow(frame)


position, background, error = get_position_by_background(video_name, hours=1, threshold=40, fps=100,
                               background_interval=1000)
position_np = np.asarray(position)
print(position_np.shape)

npy_name = video_name[:-4] + '_position' + '_'+time.strftime("%Y%m%d_%H%M") + '.npy'
np.save(npy_name, position_np)



# position_part, background, error = position


# position_np = np.asarray(position)
# np.save('adfadfasdfasdf.npy',position_np)



# print(len(position))
# for K in range(39080,len(position)):
#     print(position[K])


# temp = get_position_by_threshold(video_name, hours=1, threshold=40, fps=100)





# position_np[position_np[:,2]==569]










# '''
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def cut_video(from_name = '../CS (201031).MTS', to_name = '../data/video_CS_20201031.avi',
#               mins=2, if_save_video=0):
#
#
#
#     y_max, x_max, n_cha = get_video_size(from_name)
#     videoCapture = cv2.VideoCapture(from_name)
#     size = (x_max, y_max)  # 保存视频的大小 #WH
#     seconds = 60 * mins
#     fps = videoCapture.get(cv2.CAP_PROP_FPS)
#
#     videoWriter = cv2.VideoWriter(to_name,
#                                   cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
#
#     num_frame = int(fps * seconds)
#     for i in range(num_frame):
#         success, frame_org = videoCapture.read()
#         if success:
#             # frame = frame_org[y1:y2, x1:x2, :]
#             frame = np.copy(frame_org)
#             # print(frame_1.shape)
#
#             if if_save_video == 1:
#                 videoWriter.write(frame)
#                 if i % (fps * 100) == 0:
#                     print('i = ', i, '      ', )
#         else:
#             break
#     videoCapture.release()
#     return None
#
#
#
# from_name = '../tracking/wyh/20210107_155852_KR0190040036.avi'
#
# to_name = '../tracking/wyh/20210107_155852_KR0190040036_1.avi'
# cut_video(from_name=from_name, to_name=to_name, mins=2, if_save_video=0)
# '''


# for K_2 in range(len(error)):
#     K_0, stats, frame, frame_grey = error[K_2]
#
#     K_1 = 0
#     for i, stat in enumerate(stats):
#         # stat=stats[1]
#         if if_is_fly(stat):
#             K_1 += 1
#             # x1, y1, w, h, area = stat
#             # print(stat)
#     if K_1 > 17:  # TODO
#         # print(stats,'\n')
#         print(K_0,K_2)
#
#
# K_0, stats, frame, frame_grey = error[5]
#
# K_1 = 0
# for i, stat in enumerate(stats):
#     # stat=stats[1]
#     if if_is_fly(stat):
#         K_1 += 1
#         # x1, y1, w, h, area = stat
#         print(stat)
# if K_1 > 17:  # TODO
#     # print(stats,'\n')
#     print(K_0,K_2)
#
#
# K_1 = 0
# for i, stat in enumerate(stats):
#     # stat=stats[1]
#     if if_is_fly(stat):
#         K_1 += 1
#         # x1, y1, w, h, area = stat
#         print(stat)
#
#         # position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])
#
# # if K_1 != 7:  # TODO
# #     print('Error: frame ', K_0, 'has ', K_1, 'flies')
#     # print(stats)
#
#
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(frame)
# plt.subplot(1,2,2)
# plt.imshow(frame_grey)



































