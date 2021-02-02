

# '''
# 1. 找到所有關鍵點
# 2. 然後寫一個自動切割的程序，記錄每個position的四個點
# 3. 輸入數組，
# '''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


from egg_cut_video import cut_video_batch_process
from egg_cut_video import cut_video, get_processed_pic, save_pic_from_video

from egg_position import get_position_by_threshold, get_position_by_background, modify_position

from egg_heatmap import heatmap_without_time_not_modified, heatmap_without_time_modified
from egg_heatmap import heatmap_with_time, heatmap_with_time_sum

from egg_heatmap import show_all_hours_in_one_3D_picture,show_all_hours_3D_picture_separately
from egg_heatmap import show_one_hours_3D_picture_separately

from egg_heatmap import distance_with_time

# get_position_by_background_and_save_pictures(video_name, hours =13, edge=[6, 8], threshold=70, fps=25,
#                                to_path='../data/picture/train2020/', if_save_pic=1,
#                                      background_interval=1000)


#######################################################
#######################################################
#######################################################
#######################################################

# cut videos
# 找出框架點，裁剪视频

task = 'w1118d1'
need_save_video = 0
if task == 'CS':
    from_name = '../CS (201031).MTS'
    video_name = from_name
    videoCapture = cv2.VideoCapture(video_name)
    K_0 = 0
    position = []
    success, frame = videoCapture.read()
    plt.imshow(frame)
    videoCapture.release()

    from_hour = 0
    to_hour = 13
    from_path = '../'
    # from_video_name = 'CS_picture (200905).avi'
    from_video_name = 'CS (201031).MTS'
    from_name = from_path + from_video_name

    task_name = 'video_CS_20201031' + '_h_' + str(from_hour) + '_to_h_' + str(to_hour)
    hours = to_hour - from_hour
    minutes = 60 * hours
    print('\n' * 5, from_name, '\n', task_name)

    x_y_1 = [[557, 708, 244, 442, 4],
             [733, 882, 245, 440, 5],
             [558, 709, 643, 834, 12],
             [735, 885, 642, 836, 13],
             ]

    # minutes =2
    if need_save_video == 1:
        cut_video_batch_process(from_name=from_name, task_name=task_name,
                            x_y=x_y_1, edge=5, fps=25, mins=minutes, if_save_video=1)
elif task == 'CS_picture':
    video_name = '../CS_picture (200905).MTS'
    videoCapture = cv2.VideoCapture(video_name)
    K_0 = 0
    position = []
    success, frame = videoCapture.read()

    plt.imshow(frame)
    videoCapture.release()
    # plt.imsave('cs_pitcture_frame_1.bmp',frame)

    from_hour = 0
    to_hour = 13
    from_path = '../'
    # from_video_name = 'CS_picture (200905).avi'
    from_video_name = 'CS_picture (200905).MTS'
    from_name = from_path + from_video_name

    task_name = 'video_CS_picture_20200905' + '_h_' + str(from_hour) + '_to_h_' + str(to_hour)
    hours = to_hour - from_hour
    minutes = 60 * hours
    print('\n' * 5, from_name, '\n', task_name)

    x_y_1 = [[559, 710, 227, 425 + 5, 4],
             [738, 888, 226, 424 + 5, 5],
             [561, 710, 624, 825, 12],
             [740, 889, 625, 825, 13], ]

    # minutes = 2
    if need_save_video == 1:
        cut_video_batch_process(from_name=from_name, task_name=task_name,
                            x_y=x_y_1, edge=5, fps=25, mins=minutes, if_save_video=1)
elif task == 'w1118':
    video_name = '../w1118 (200930).MTS'
    videoCapture = cv2.VideoCapture(video_name)
    K_0 = 0
    position = []
    success, frame = videoCapture.read()

    plt.imshow(frame)
    videoCapture.release()
    # plt.imsave('cs_pitcture_frame_1.bmp',frame)

    from_hour = 0
    to_hour = 13
    from_path = '../'
    # from_video_name = 'CS_picture (200905).avi'
    from_video_name = 'w1118 (200930).MTS'
    from_name = from_path + from_video_name

    task_name = 'video_w1118_20200930' + '_h_' + str(from_hour) + '_to_h_' + str(to_hour)
    hours = to_hour - from_hour
    minutes = 60 * hours
    print('\n' * 5, from_name, '\n', task_name)

    x_y_1 = [[558, 711, 237, 440, 4],
             [737, 889, 240, 440, 5],
             [553, 705, 637, 826, 12],
             [730, 884, 639, 828, 13], ]

    minutes = 2
    if need_save_video == 1:
        cut_video_batch_process(from_name=from_name, task_name=task_name,
                            x_y=x_y_1, edge=20, fps=25, mins=minutes, if_save_video=1)



# 分析视频，得到position_np 并保存为npy
task = 'w1118'
video_name_all = []
if task == 'CS':
    video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_728_887_240_445_5.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_553_714_638_839_12.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_730_890_637_841_13.avi'
    video_name_all.append(video_name)
elif task == 'CS_picture':
    video_name = '../data/video_CS_picture_20200905_h_0_to_h_13/video_CS_picture_20200905_h_0_to_h_13_554_715_222_435_4.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_CS_picture_20200905_h_0_to_h_13/video_CS_picture_20200905_h_0_to_h_13_733_893_221_434_5.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_CS_picture_20200905_h_0_to_h_13/video_CS_picture_20200905_h_0_to_h_13_556_715_619_830_12.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_CS_picture_20200905_h_0_to_h_13/video_CS_picture_20200905_h_0_to_h_13_735_894_620_830_13.avi'
    video_name_all.append(video_name)
elif task == 'w1118':
    video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_538_731_217_460_4.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_717_909_220_460_5.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_533_725_617_846_12.avi'
    video_name_all.append(video_name)
    video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_710_904_619_848_13.avi'
    video_name_all.append(video_name)
    # video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_543_726_222_455_4.avi'
    # video_name_all.append(video_name)
    # video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_722_904_225_455_5.avi'
    # video_name_all.append(video_name)
    # video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_538_720_622_841_12.avi'
    # video_name_all.append(video_name)
    # video_name = '../data/video_w1118_20200930_h_0_to_h_13/video_w1118_20200930_h_0_to_h_13_715_899_624_843_13.avi'
    # video_name_all.append(video_name)


# 分析视频，得到position_np 并保存为npy
need_compute_position = 0
need_save_position_np = 0
for K in range(len(video_name_all)):
    video_name = video_name_all[K]
    if need_compute_position == 1:
        # position = get_position_by_threshold(video_name, hours=13, threshold=90, fps=25)

        position = get_position_by_background(video_name, hours=2, edge=[6, 8], threshold=70, fps=25,
                                              to_path='../data/picture/train2020/', if_save_pic=0,
                                              background_interval=1000)
    position_np = np.array(position)
    position_name = video_name[:-4] + '_position.npy'
    if need_save_position_np == 1:
        np.save(position_name, position_np)


# 加载position数据
position_np_all = []
for K in range(len(video_name_all)):
    video_name = video_name_all[K]
    position_name = video_name[:-4] + '_position.npy'
    position_np = np.load(position_name)
    print(position_np.shape)
    position_np_all.append(position_np)
    del position_np


print(len(position_np_all))

#######################################################
#######################################################
#######################################################
#######################################################
# 修改position，展示position

K_position_np = 0
position_np = position_np_all[K_position_np]
print(video_name_all[K_position_np])



need_modify = 1
need_modify = 0
# modify_position(position_np,need_modify=0)
# position_np = modify_position(position_np,need_modify=0)
position_np = modify_position(position_np,need_modify=1)
modify_position(position_np,need_modify=0)



show_all_hours_in_one_3D_picture(position_np,from_show_length = 0,to_show_length = len(position_np),interval=10)

show_all_hours_3D_picture_separately(position_np, from_hour=0, to_hour=2, num_plt_begin =10)

show_one_hours_3D_picture_separately(position_np, from_hour=2, num_interval=3, num_plt_begin =100)


#######################################################
#######################################################
#######################################################
#######################################################
# 展示果蝇在近端和远端的变化，随时间变化


M=2
N=4
plt.figure()
heatmap_all = []
K_all = [1, 2, 5, 6, 3, 4, 7, 8]
for K in range(len(video_name_all)):
    plt.subplot(M,N,K_all[K])
    plt.title(video_name_all[K].split('/')[2][6:-12]+'_'+video_name_all[K].split('/')[3][-6:-4])
    # heatmap = heatmap_without_time_modified(position_np_all[K][0:180000], distance_threshold=5, frame_interval=10,
    #                                         max_count=20, if_show_pic=1)
    fps = 25
    time_interval = fps * 60
    distance_with_time(position_np_all[K][0:180000], time_interval)
    # time_interval = fps * 60 * 5
    # distance_with_time(position_np_all[K], time_interval)
    print(video_name_all[K])
    # heatmap_all.append(heatmap)




# position_heatmap = heatmap_without_time_not_modified(position_np,if_show_pic=0)




M=2
N=4
plt.figure()
heatmap_all = []
K_all = [1, 2, 5, 6, 3, 4, 7, 8]
for K in range(len(video_name_all)):
    plt.subplot(M,N,K_all[K])
    plt.title(video_name_all[K].split('/')[2][6:-12]+'_'+video_name_all[K].split('/')[3][-6:-4])
    # heatmap = heatmap_without_time_modified(position_np_all[K], distance_threshold=10, frame_interval=10, max_count=160, if_show_pic=1)
    # heatmap = heatmap_without_time_modified(position_np_all[K],distance_threshold = 5,frame_interval=3,max_count=45,if_show_pic=1)
    # heatmap = heatmap_without_time_modified(position_np_all[K][0:180000], distance_threshold=5, frame_interval=1, max_count=20, if_show_pic=1)
    heatmap = heatmap_without_time_modified(position_np_all[K][0:180000], distance_threshold=5, frame_interval=10,
                                            max_count=20, if_show_pic=1)
    print(video_name_all[K])
    heatmap_all.append(heatmap)



position_heatmap_time=heatmap_with_time(position_np, fps=25, interval_short=10,
                                        num_interval=51, length_process=len(position_np), if_show_pic=1)


a = heatmap_with_time_sum(position_heatmap_time, interval_long=10,
                        num_interval=51, max_count=70000, if_show_pic=1)


#######################################################
#######################################################
#######################################################
#######################################################


from egg_streamplot import get_speed, draw_2D_speed_centered, draw_3D_speed_centered
from egg_streamplot import draw_speed_streamplot

video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi'
position_name = video_name[:-4] + '_position.npy'
position_np = np.load(position_name)
print(position_np.shape)


speed_np = get_speed(position_np, distance_threshold=10, frame_interval=10, save_interval=100)

draw_2D_speed_centered(speed_np)

draw_3D_speed_centered(speed_np)


stream_map = np.copy(speed_np)
print(stream_map.shape)

# stream_map[0]
# K_0, x, y, x_1 - x, y_1 - y, x_2 - x, y_2 - y, x_3 - x, y_3 - y, x_4 - x, y_4 - y

draw_speed_streamplot(stream_map, n_scale=3, plt_density=3)


#######################################################
#######################################################
#######################################################
#######################################################

# acceleration

from egg_streamplot import get_acceleration, draw_acceleration_streamplot


time = 10
speed = get_speed(position_np, distance_threshold=10,frame_interval=time, save_interval=time)

acceleration = get_acceleration(speed)

draw_acceleration_streamplot(acceleration, from_hour_analysis=0, to_hour_analysis=2,
                                 fps=25, save_interval=time, n_scale=3, plt_density=3)

# acceleration = np.copy(speed[1:, 0:7])
# acceleration.shape
# acceleration[:, 5:7] = np.copy(speed[1:,3:5]-speed[:-1,3:5])
# print(acceleration.shape)




#######################################################
#######################################################
#######################################################
#######################################################

##### pose

from egg_pose_nj import load_pose_csv_and_save_npy, pose_to_orentation
from egg_pose_nj import pose_to_orentation_by_part, draw_pose_streamplot,draw_orentation_scatterplot



csv_name = '../data/video_CS_20201031_h_0_to_h_13/orientation/video_CS_20201031_h_0_to_h_13_552_713_239_447_4_pose.csv'

pose_npy_name = csv_name[:-4]+'.npy'
pose = np.load(pose_npy_name)


use_int = 1


orentation_np = pose_to_orentation(pose, frame_interval=100, use_int=1)


orentation_part = pose_to_orentation_by_part(pose, frame_interval=10, y_oren_threshold=36,)





draw_pose_streamplot(orentation_np, plt_density=5 )

draw_orentation_scatterplot(orentation_np)




draw_pose_streamplot(orentation_part, plt_density=5 )

draw_orentation_scatterplot(orentation_part)

#######################################################
#######################################################
#######################################################
#######################################################

##### rubbish


rubbish=0
if rubbish==1:
    # pose = load_pose_csv_and_save_npy(csv_name)

    # [  0,  55, 172,  47, 166,  56, 173,  66, 181,  66, 181,  66, 180],
    # [  1,  55, 173,  46, 167,  56, 174,  66, 180,  66, 182,  68, 179],
    # [  2,  55, 173,  45, 167,  57, 174,  64, 182,  65, 181,  66, 180],
    # [  3,  55, 172,  47, 166,  56, 173,  66, 180,  66, 182,  66, 180]
    pose[:4]

    # a[0:4]
    #
    # pose_disorder
    #
    # frame_interval
    #

    # position_np
    # distance_threshold = 10
    #
    # max_count=160
    # if_show_pic=0

    # pose = np.load('pose.npy')

    orentation_np[0:4]
    # [  0,  48, 120, -19, -11],
    # [100,  48, 120, -20, -10],
    # [200,  47, 120, -21, -10],
    # [300,  42, 126, -19,  10]


rubbish=0
if rubbish==1:
    # draw_acceleration_streamplot
    # acceleration_heatmap_without_time_modified
    # def draw_acceleration_streamplot(acceleration, from_hour_analysis=0, to_hour_analysis=2,
    #                                                distance_threshold=10, frame_interval=10, max_count=160,
    #                                                if_show_pic=0, fps=25):
    #     x_all = acceleration[:, 1]
    #     y_all = acceleration[:, 2]
    #
    #     num_x_interval = int(max(x_all)) + 1
    #     num_y_interval = int(max(y_all)) + 1
    #     position_heatmap = np.zeros([num_y_interval, num_x_interval])
    #
    #     from_show_length = fps * 60 * 60 * from_hour_analysis
    #     to_show_length = fps * 60 * 60 * to_hour_analysis
    #
    #     for K_0 in range(from_show_length, to_show_length):
    #
    #         distance_1 = (x_all[K_0 - frame_interval] - x_all[K_0]) ** 2 + (y_all[K_0 - frame_interval] - y_all[K_0]) ** 2
    #         distance_2 = (x_all[K_0 + frame_interval] - x_all[K_0]) ** 2 + (y_all[K_0 + frame_interval] - y_all[K_0]) ** 2
    #         if distance_1 > distance_threshold or distance_2 > distance_threshold:
    #
    #             x = x_all[K_0]
    #             y = y_all[K_0]
    #             # distance = np.sum
    #             if position_heatmap[int(y), int(x)] < max_count:
    #                 position_heatmap[int(y), int(x)] += 1
    #
    #         # x = x_modified[K_0]
    #         # y = y_modified[K_0]
    #         # # distance = np.sum
    #         # position_heatmap[int(y),int(x)]+=1
    #
    #     # print(position_heatmap.shape)
    #     if if_show_pic:
    #         import matplotlib.pyplot as plt
    #         # plt.figure()
    #         plt.imshow(position_heatmap)
    #         plt.colorbar()
    #         plt.show()
    #     # print()
    #     return position_heatmap

    # a = speed.astype('int')
    # b = acceleration.astype('int')
    #
    #
    # a[1000:1004]
    # b[1000:1002]
    #
    #
    # speed[0:2]
    # acceleration_1[0:2]
    #
    # x_position = acceleration_1[K_0, 1]
    # y_position = acceleration_1[K_0, 2]

    # position_np[K_0, 2], x, y, x_1 - x, y_1 - y, x_2 - x, y_2 - y, x_3 - x, y_3 - y, x_4 - x, y_4 - y,

    # speed

    # acceleration = []
    # # v_x_1 = speed[:, 3,4]
    # for K_0 in range(len(speed)):
    #     # v_x_1 = speed[K_0, ]
    #
    #     acceleration.append([
    #         speed[K_0,0:2], speed[K_0]
    #     ])
    #
    #     a=1
    #
    # def get_acceleration(speed_np,distance_threshold = 10,frame_interval=10, save_interval = 10):
    #     '''
    #     speed_distance = 5
    #     第一个代表大小
    #     第二个维度代表x
    #     第三个维度代表y
    #     或者添加一个角度？
    #
    #     综合一下，某个位置的全部信息？求一个平均？
    #     代表果蝇在这个位置的一个
    #
    #
    #     还可以用一下聚类？
    #     把他所在点的前后的信息都找出来？
    #
    #     位置，时间，
    #     前后移动的距离
    #     前后五个点的位置
    #     最好用差值？
    #     就是和这个位置的差值？
    #     '''
    #
    #     x_org = position_np[:, 0]
    #     x_modified = x_org - min(x_org)
    #
    #     y_org = position_np[:, 1]
    #     y_modified = y_org - min(y_org)
    #     y_modified = y_modified * max(x_modified) / (max(y_org) - min(y_org))
    #
    #     # for K_0 in range(len_position):
    #     # x = position_np
    #
    #     # num_x_interval = int(max(x_modified)) + 1
    #     # num_y_interval = int(max(y_modified)) + 1
    #
    #     # position_heatmap = np.zeros([num_y_interval, num_x_interval])
    #
    #     # 找出来静止的店和运动的点，可以在找到运动速度图之后再搞这个
    #
    #
    #     speed = []
    #     for K_0 in range(0, len(position_np) - 4 * frame_interval):
    #         x = x_modified[K_0]
    #         y = y_modified[K_0]
    #         x_1 = x_modified[K_0 + frame_interval]
    #         y_1 = y_modified[K_0 + frame_interval]
    #
    #         x_2 = x_modified[K_0 + 2 * frame_interval]
    #         y_2 = y_modified[K_0 + 2 * frame_interval]
    #
    #         x_3 = x_modified[K_0 + 3 * frame_interval]
    #         y_3 = y_modified[K_0 + 3 * frame_interval]
    #
    #         x_4 = x_modified[K_0 + 4 * frame_interval]
    #         y_4 = y_modified[K_0 + 4 * frame_interval]
    #         if K_0 % save_interval == 0:
    #             speed.append(
    #                 [position_np[K_0, 2], x, y, x_1 - x, y_1 - y, x_2 - x, y_2 - y, x_3 - x, y_3 - y, x_4 - x, y_4 - y, ])
    #
    #     speed_np = np.asarray(speed)
    #
    #     print(speed_np.shape)
    #     return speed_np

    # x_body = orentation_np[:, 1]
    # y_body = orentation_np[:, 2]
    # y_body = max(y_body) -y_body
    #
    # x = orentation_np[:, 3]
    # y = orentation_np[:, 4]
    # y = -y
    # index = orentation_np[:, 0]
    #
    #
    # plt.streamplot(x_body, y_body, x, y)

    # density=1, linewidth=None, color=None,
    # cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
    # minlength=0.1, transform=None, zorder=None, start_points=None,
    # maxlength=4.0, integration_direction='both', *, data=None)

    # M=2
    # N=4
    # plt.figure()
    # heatmap_all = []
    # for K in range(len(video_name_all)):
    #     plt.subplot(M,N,K_all[K])
    #     plt.title(video_name_all[K].split('/')[2][6:-12]+'_'+video_name_all[K].split('/')[3][-6:-4])
    #
    #     print(video_name_all[K])
    #     heatmap_all.append(heatmap)

    # for K in range(len(video_name_all)):
    #     # K=0
    #     video_name = video_name_all[K]
    #     position_name = video_name[:-4] + '_position.npy'
    #     position_np = np.load(position_name)
    #     print(position_np.shape)

    # position_np
    from_show_length = 0
    to_show_length = -1
    interval = 1

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

    # figure 2
    # scatter plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, marker='o', s=5, c=z, cmap='summer')
    # ax.legend()
    plt.show()

    # figure 3

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(x, y, z, color="g")
    plt.show()

    # figure 4

    plt.figure()
    plt.plot(x, y, linewidth='1')

    # figure 5
    plt.figure()
    plt.scatter(x, y, marker='o', c=z, cmap='summer')

    # figure 6
    plt.figure()
    plt.xlim((0, 160))
    plt.ylim((0, 210))
    plt.gca(projection='3d')
    plt.plot(x, y, z, label='parametric curve')
    plt.show()

    plt.figure()
    plt.xlim(0, 160)
    # plt.ylim((0,210))
    plt.gca(projection='3d')
    plt.plot(x, y, z)
    plt.show()


# from_name =  CS_picture (200905).MTS
#
#
#
#
#
#
#
#
#
# task_name = 'video_CS_picture_20200905'

# to_path = '../data/'+task_name+'/'
# to_video_name =  task_name + '_h_' + str(from_hour) + '_to_h_' + str(to_hour) #+'.avi'
# to_name = to_path + to_video_name
# cut_video_batch_process(from_name = from_name, task_name = task_name,
#               x_y = x_y_1, fps = 25, mins = minutes, if_save_video = 1)
#

# cut_video_batch_process(from_name = '../CS (201031).MTS', task_name = 'video_CS_20201031_mins_2',
#               x_y = x_y_1, fps = 25, mins = 2,#minutes,#2,
#                          if_save_video = 1)
############################
############################
############################
############################
############################
############################
############################
# task_name = 'video_CS_picture_20200905'
#
# to_path = '../data/'+task_name+'/'
# to_video_name =  task_name + '_h_' + str(from_hour) + '_to_h_' + str(to_hour) #+'.avi'
# to_name = to_path + to_video_name
# x_y_1 = [[557,708,244,442,4],
#          [733,882,245,440,5],
#          [558,709,643,834,12],
#          [735,885,642,836,13],
#          ]
############################
############################
############################
############################
#
# cut_video_batch_process(from_name = '../CS (201031).MTS', task_name = 'video_CS_20201031_mins_2',
#               x_y = x_y_1, fps = 25, mins = 2,#minutes,#2,
#                          if_save_video = 1)



# for K in range(len(video_name_all)):
#     video_name = video_name_all[K]
#     position = get_position_by_threshold(video_name, hours = 13, threshold = 70, fps = 25)
#     to_name = video_name
#     position_np = np.array(position)
#     position_name = to_name[:-4] + '_position.npy'
#     np.save(position_name, position_np)







#
#
#
#
#
#
#
#
# # 分析视频，得到position_np 并保存为npy
#
# video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi'
# position_name = video_name[:-4] + '_position.npy'
# position_np = np.load(position_name)
# print(position_np.shape)
#
# position_np,distance_threshold = 10
# frame_interval=10
# max_count=160
# # if_show_pic=0
#
# len_position = len(position_np)
# x_org = position_np[:, 0]
# x_modified = x_org - min(x_org)
# y_org = position_np[:, 1]
# y_modified = y_org - min(y_org)
# y_modified = y_modified * max(x_modified) / (max(y_org) - min(y_org))
#
# # for K_0 in range(len_position):
#     # x = position_np
#
# num_x_interval = int(max(x_modified)) + 1
# num_y_interval = int(max(y_modified)) + 1
# position_heatmap = np.zeros([num_y_interval, num_x_interval])
#
# # 找出来静止的店和运动的点，可以在找到运动速度图之后再搞这个
#
#
# fps = 250
# speed = []
# for K_0 in range(0, len(position_np) - 4*frame_interval):
#     x = x_modified[K_0]
#     y = y_modified[K_0]
#     x_1 = x_modified[K_0+frame_interval]
#     y_1 = y_modified[K_0+frame_interval]
#
#     x_2 = x_modified[K_0 + 2*frame_interval]
#     y_2 = y_modified[K_0 + 2*frame_interval]
#
#     x_3 = x_modified[K_0 + 3*frame_interval]
#     y_3 = y_modified[K_0 + 3*frame_interval]
#
#     x_4 = x_modified[K_0 + 4*frame_interval]
#     y_4 = y_modified[K_0 + 4*frame_interval]
#     if K_0 % fps ==0:
#         speed.append([position_np[K_0,2],x,y, x_1-x, y_1-y, x_2-x, y_2-y, x_3-x, y_3-y,   x_4-x, y_4-y,   ])
#
# speed_np = np.asarray(speed)
# print(speed_np.shape)
#
# x=speed_np[:,3]
# y=speed_np[:,4]
# index = speed_np[:,0]
#
# plt.figure()
# plt.scatter(x,y,marker='o', s=5, c=index, cmap='summer')
# plt.show()
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # plt.figure()
# ax.scatter(x,y,index,marker='o', s=5, c=index, cmap='summer')
# # ax.legend()
# plt.show()




# x_modified[K_0 - frame_interval]
# y_modified[K_0 - frame_interval]
#
#
#
#
#
# K_1=1
# for K_0 in range(len(position_np)):
#     if position_np[K_0,3]>=1:
#         print(position_np[(K_0-1):(K_0+1),0:3])
#         print('\n'*1)
#         K_1 += 1
#
#     # distance_1 = np.sum(np.square(position_np[K_0 - frame_interval, 0:2] - position_np[K_0, 0:2]))
#     # distance_2 = np.sum(np.square(position_np[K_0 + frame_interval, 0:2] - position_np[K_0, 0:2]))
#     distance_1 = (x_modified[K_0 - frame_interval] - x_modified[K_0]) ** 2 + (
#                 y_modified[K_0 - frame_interval] - y_modified[K_0]) ** 2
#     distance_2 = (x_modified[K_0 + frame_interval] - x_modified[K_0]) ** 2 + (
#                 y_modified[K_0 + frame_interval] - y_modified[K_0]) ** 2
#     # if distance_1 > distance_threshold:
#         # or distance_2 > distance_threshold:
#         # print(position_np[K - 1:K + 2, 0:3])
#         # print(distance_1, distance_2)
#         x = x_modified[K_0]
#         y = y_modified[K_0]
#         # distance = np.sum
#         if position_heatmap[int(y), int(x)] < max_count:
#             position_heatmap[int(y), int(x)] += 1
#     print('1')








