










import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from egg_cut_video import cut_video_batch_process
from egg_cut_video import get_video_of_position_heatmap



def show_all_hours_in_one_3D_picture(position_np,from_show_length = 0,to_show_length = -1,interval=1):
    print(position_np.shape)
    # from_show_length = 0
    # to_show_length = len(position_np)
    #
    # from_show_length = 940000#   len(position_np)-3*80000# 0#   1000#                        0#
    # to_show_length = 1100000#len(position_np)#80000#3*80000#

    x = position_np[from_show_length:to_show_length:interval, 0]
    y = position_np[from_show_length:to_show_length:interval, 1]
    y = max(y) - y
    z = position_np[from_show_length:to_show_length:interval, 2]
    print(len(x))

    # figure 1
    # line plot
    fig = plt.figure()
    # fig.xlim((0,200))
    # plt.xlim((0, x_max))
    # plt.ylim((0, y_max))
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='parametric curve')
    plt.show()
    # plt.close()
    return None


def show_all_hours_3D_picture_separately(position_np, from_hour=0, to_hour=13, num_plt_begin=10, fps=25):
    x_max = max(position_np[:, 0])
    y_max = max(position_np[:, 1])

    # from_hour_analysis = 0
    # to_hour_analysis =  1

    # 逐小时分析
    for K in range(from_hour, to_hour):
        from_hour_analysis = K
        to_hour_analysis = K + 1
        from_show_length = fps * 60 * 60 * from_hour_analysis
        to_show_length = fps * 60 * 60 * to_hour_analysis
        print(position_np.shape)
        x = position_np[from_show_length:to_show_length, 0]
        y = position_np[from_show_length:to_show_length, 1]
        y = max(y) - y
        z = position_np[from_show_length:to_show_length, 2]

        # figure 1
        # line plot
        # plt.figure(figsize=(18, 14))
        fig = plt.figure(K+num_plt_begin)
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, label='parametric curve')
        # ax.legend()
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        plt.title(str(K))
        plt.show()
    # plt.close()
    return None


def show_one_hours_3D_picture_separately(position_np, from_hour=10, num_interval=10, num_plt_begin=100, fps=25):
    x_max = max(position_np[:, 0])
    y_max = max(position_np[:, 1])
    print(position_np.shape)
    # from_hour_analysis = 0
    # to_hour_analysis = 1
    # from_show_length = fps * 60 * 60 * from_hour_analysis
    # to_show_length = fps * 60 * 60 * to_hour_analysis
    to_hour = from_hour + 1
    from_frame_analysis = fps * 60 * 60 * from_hour
    to_frame_analysis = fps * 60 * 60 * to_hour

    for K in range(0, num_interval):
        from_hour_analysis = K
        to_hour_analysis = K + 1
        from_show_length = from_frame_analysis + int((to_frame_analysis - from_frame_analysis) / num_interval) * K
        to_show_length = from_frame_analysis + int((to_frame_analysis - from_frame_analysis) / num_interval) * (K + 1)

        x = position_np[from_show_length:to_show_length, 0]
        y = position_np[from_show_length:to_show_length, 1]
        y = max(y) - y
        z = position_np[from_show_length:to_show_length, 2]

        # figure 1
        # line plot
        # plt.figure(figsize=(18, 14))
        fig = plt.figure(K + num_plt_begin)
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, label='parametric curve')
        # ax.legend()
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        plt.title(str(K))
        plt.show()
    return None

#######
# heatmap without time
def heatmap_without_time_not_modified(position_np,if_show_pic=0):
    #############
    # heatmap without time
    # have not modified
    x_org = position_np[:, 0]
    x_modified = x_org - min(x_org)
    y_org = position_np[:, 1]
    y_modified = y_org - min(y_org)

    num_x_interval = int(max(x_modified)) + 1
    num_y_interval = int(max(y_modified)) + 1
    position_heatmap = np.zeros([num_y_interval, num_x_interval])
    for K_0 in range(len(position_np)):
        x = x_modified[K_0]
        y = y_modified[K_0]
        position_heatmap[int(y), int(x)] += 1

    print(position_heatmap.shape)
    if if_show_pic:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(position_heatmap)
        plt.colorbar()
        plt.show()
    return position_heatmap

# have modified
def heatmap_without_time_modified(position_np,distance_threshold = 10,frame_interval=10,max_count=160,if_show_pic=0):
    x_org = position_np[:, 0]
    x_modified = x_org - min(x_org)
    y_org = position_np[:, 1]
    y_modified = y_org - min(y_org)
    y_modified = y_modified * max(x_modified) / (max(y_org) - min(y_org))
    # max(y_modified)

    num_x_interval = int(max(x_modified)) + 1
    num_y_interval = int(max(y_modified)) + 1
    position_heatmap = np.zeros([num_y_interval, num_x_interval])

    for K_0 in range(frame_interval, len(position_np) - frame_interval):
        # distance_1 = np.sum(np.square(position_np[K_0 - frame_interval, 0:2] - position_np[K_0, 0:2]))
        # distance_2 = np.sum(np.square(position_np[K_0 + frame_interval, 0:2] - position_np[K_0, 0:2]))
        distance_1 = (x_modified[K_0 - frame_interval] - x_modified[K_0]) ** 2 + (y_modified[K_0 - frame_interval] - y_modified[K_0]) ** 2
        distance_2 = (x_modified[K_0 + frame_interval] - x_modified[K_0]) ** 2 + (y_modified[K_0 + frame_interval] - y_modified[K_0]) ** 2
        if distance_1 > distance_threshold or distance_2 > distance_threshold:

            # print(position_np[K - 1:K + 2, 0:3])
            # print(distance_1, distance_2)
            x = x_modified[K_0]
            y = y_modified[K_0]
            # distance = np.sum
            if position_heatmap[int(y), int(x)] < max_count:
                position_heatmap[int(y), int(x)] += 1

        # x = x_modified[K_0]
        # y = y_modified[K_0]
        # # distance = np.sum
        # position_heatmap[int(y),int(x)]+=1

    # print(position_heatmap.shape)
    if if_show_pic:
        import matplotlib.pyplot as plt
        # plt.figure()
        plt.imshow(position_heatmap)
        plt.colorbar()
        plt.show()
    # print()
    return position_heatmap


#########
# heatmap with time
# and compass
def heatmap_with_time(position_np, fps=25, interval_short=10, num_interval=51, length_process=-1, if_show_pic=0):
    # heatmap_with_time(position_np, fps=25, interval_short=10, num_interval=51, length_process=len(position_np),
    #                   if_show_pic=0):
    period_short = fps * interval_short

    # position_heatmap = []
    print(max(position_np[:, 0]) - min(position_np[:, 0]))
    print(max(position_np[:, 1]) - min(position_np[:, 1]))

    # num_interval = 51  # 51#14#50
    x_interval = int((max(position_np[:, 0]) - min(position_np[:, 0])) / (num_interval - 1) + 1)
    # K_0 = 0
    # x_K = int(position_np[K_0,0] % x_interval)

    x_org = position_np[:, 0]
    x_modified = x_org - min(x_org)
    y_org = position_np[:, 1]
    y_modified = y_org - min(y_org)

    position_heatmap_time = []

    for K_0 in range(len(position_np)):
        # K_0=0
        x = x_modified[K_0]
        y = y_modified[K_0]
        x_indicate = int(x / x_interval)

        if K_0 == 0:
            # y_range = [0] * num_interval
            y_range_max = np.zeros([1, num_interval])  # [0] * num_interval
            y_range_min = np.ones([1, num_interval]) * max(y_modified) # [0] * num_interval

        if K_0 % period_short != 0:
            if y_range_max[0, x_indicate] < y:
                y_range_max[0, x_indicate] = y
                # y_range_max[0, num_interval] = K_0
            if y_range_min[0, x_indicate] >= y:
                y_range_min[0, x_indicate] = y
                # y_range_min[0, num_interval] = K_0

        if K_0 % period_short == 0:
            temp = []
            for K_1 in range(len(y_range_max[0])):
                temp.append(max(0,y_range_max[0, K_1] - y_range_min[0, K_1]))
            # position_heatmap.append(y_range_max[0,:]-y_range_min[0,:])
            position_heatmap_time.append(temp)
            y_range_max = np.zeros([1, num_interval])
            y_range_min = np.zeros([1, num_interval])
            print(K_0)

    # a=np.array(position_heatmap)
    # print(a.shape)
    print(len(position_heatmap_time))
    if if_show_pic:
        plt.figure()
        plt.imshow(position_heatmap_time)
        plt.colorbar()
        plt.title('time ↓ ')
        plt.show()
    return position_heatmap_time


def heatmap_with_time_sum(position_heatmap_time, interval_long=10, num_interval=51, max_count=1000, if_show_pic=0):

    # period_long = period_short * 10
    position_heatmap_time_sum = []
    for K_0 in range(len(position_heatmap_time)):
        if K_0 == 0:
            sum = np.zeros([1, num_interval])
        if K_0 % interval_long != 0:
            sum += np.array(position_heatmap_time[K_0])
            # sum += np.array(position_heatmap[K_0])
        if K_0 % interval_long == 0:
            temp = []
            for K_1 in range(num_interval): #len(y_range_max[0])):

                temp.append(min(sum[0, K_1],max_count))
            # position_heatmap.append(y_range_max[0,:]-y_range_min[0,:])
            position_heatmap_time_sum.append(temp)
            # position_heatmap_sum.append(sum)
            sum = np.zeros([1, num_interval])

    # a = np.array(position_heatmap_time_sum)
    # print(a.shape)
    print(len(position_heatmap_time_sum))
    if if_show_pic:
        plt.figure()
        plt.imshow(position_heatmap_time_sum)
        plt.colorbar()
        plt.title('time ↓ ')
        plt.show()
    return position_heatmap_time_sum
    # plt.plot()
    # len()


def distance_with_time(position_np, time_interval):
    x_org = position_np[:, 0]
    x_modified = x_org - min(x_org)
    y_org = position_np[:, 1]
    y_modified = y_org - min(y_org)

    y_mid = max(y_modified) / 2

    y_01 = np.copy(y_modified)
    for K_0 in range(len(position_np)):
        y_01[K_0] = 1 if y_modified[K_0] > y_mid else 0

    y_temp = np.copy(y_modified[0:int(len(y_01) / time_interval)])
    for K_0 in range(int(len(y_01) / time_interval)):
        y_temp[K_0] = sum(y_01[(K_0 * time_interval):(K_0 * time_interval + time_interval)])

    # plt.figure()
    plt.plot(range(len(y_temp)), y_temp)
    return y_temp

#
# video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi'
# position_name = video_name[:-4] + '_position.npy'
# position_np = np.load(position_name)
# print(position_np.shape)

# [centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area]
# x_max = np.max(position_np[:,4])
# y_max = np.max(position_np[:,5])


























