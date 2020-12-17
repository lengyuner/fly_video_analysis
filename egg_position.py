







import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_position_by_threshold(video_name, hours = 1, threshold = 70, fps = 25):
    K_0 = 0
    num_frame = int(fps * 60 * 60 *hours)

    position = []
    videoCapture = cv2.VideoCapture(video_name)
    for K_0 in range(num_frame):
    # while True:
        success, frame = videoCapture.read()
        if success:
            frame_1 = np.copy(frame)
            frame_blured = cv2.medianBlur(frame_1, 3)
            frame_bi = cv2.cvtColor(frame_blured, cv2.COLOR_RGB2GRAY)
            # plt.imshow(frame_bi,'gray')

            frame_bi[frame_bi < threshold] = 0
            frame_bi[frame_bi >= threshold] = 255
            # frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                  cv2.THRESH_BINARY, 25, 5)
            frame_3_body = 255 - frame_bi.astype(np.uint8)
            # plt.imshow(frame_3_body,'gray')
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_3_body, connectivity=4)
            K_1 = 0
            for i, stat in enumerate(stats):
                if stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50:#stat[4] > 100 and
                    x1, y1, h, w, area = stat
                    position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])
                    K_1 += 1
            if not K_1 == 1:
                print(K_0)
                print(stat)
            K_0 += 1
            if K_0 % (25 * 100) == 0:
                print('K_0:   ', K_0)
        else:
            break

    if K_0 >= num_frame - 10:
        print('Completed. Congratulations!')
    videoCapture.release()
    return position






def modify_position(position_np, need_modify=0):
    if need_modify:
        for K in range(1, len(position_np) - 1):
            distance_1 = np.sum(np.square(position_np[K - 1, 0:2] - position_np[K, 0:2]))
            distance_2 = np.sum(np.square(position_np[K + 1, 0:2] - position_np[K, 0:2]))
            distance_3 = np.sum(np.square(position_np[K + 1, 0:2] - position_np[K - 1, 0:2]))
            if distance_1 > 1000 and distance_2 > 1000 and distance_3 < 1000:
                print(position_np[K - 1:K + 2, 0:3])
                print(distance_1, distance_2)
                position_np[K, 0:2] = position_np[K - 1, 0:2]
        return position_np
    else:
        for K in range(1, len(position_np) - 1):
            distance_1 = np.sum(np.square(position_np[K - 1, 0:2] - position_np[K, 0:2]))
            distance_2 = np.sum(np.square(position_np[K + 1, 0:2] - position_np[K, 0:2]))
            distance_3 = np.sum(np.square(position_np[K + 1, 0:2] - position_np[K - 1, 0:2]))
            if distance_1 > 1000 and distance_2 > 1000 and distance_3 < 1000:
                print(position_np[K - 1:K + 2, 0:3])
                print(distance_1, distance_2)
        return None






