









import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def get_speed(position_np, distance_threshold = 10, frame_interval=10, save_interval = 10):
    '''
    speed_distance = 5
    第一个代表大小
    第二个维度代表x
    第三个维度代表y
    或者添加一个角度？

    综合一下，某个位置的全部信息？求一个平均？
    代表果蝇在这个位置的一个


    还可以用一下聚类？
    把他所在点的前后的信息都找出来？

    位置，时间，
    前后移动的距离
    前后五个点的位置
    最好用差值？
    就是和这个位置的差值？
    '''

    x_org = position_np[:, 0]
    x_modified = x_org - min(x_org)

    y_org = position_np[:, 1]
    y_modified = y_org - min(y_org)
    y_modified = y_modified * max(x_modified) / (max(y_org) - min(y_org))

    # for K_0 in range(len_position):
    # x = position_np

    # num_x_interval = int(max(x_modified)) + 1
    # num_y_interval = int(max(y_modified)) + 1

    # position_heatmap = np.zeros([num_y_interval, num_x_interval])

    # 找出来静止的店和运动的点，可以在找到运动速度图之后再搞这个


    speed = []
    for K_0 in range(0, len(position_np) - 4 * frame_interval):
        x = x_modified[K_0]
        y = y_modified[K_0]
        x_1 = x_modified[K_0 + frame_interval]
        y_1 = y_modified[K_0 + frame_interval]

        x_2 = x_modified[K_0 + 2 * frame_interval]
        y_2 = y_modified[K_0 + 2 * frame_interval]

        x_3 = x_modified[K_0 + 3 * frame_interval]
        y_3 = y_modified[K_0 + 3 * frame_interval]

        x_4 = x_modified[K_0 + 4 * frame_interval]
        y_4 = y_modified[K_0 + 4 * frame_interval]
        if K_0 % save_interval == 0:
            speed.append(
                [position_np[K_0, 2], x, y, x_1 - x, y_1 - y, x_2 - x, y_2 - y, x_3 - x, y_3 - y, x_4 - x, y_4 - y, ])

    speed_np = np.asarray(speed)

    print(speed_np.shape)
    return speed_np


def draw_2D_speed_centered(speed_np):
    x = speed_np[:, 3]
    y = speed_np[:, 4]
    index = speed_np[:, 0]

    plt.figure()
    plt.scatter(x, y, marker='o', s=5, c=index, cmap='summer')
    plt.show()
    return None


def draw_3D_speed_centered(speed_np):
    x = speed_np[:, 3]
    y = speed_np[:, 4]
    index = speed_np[:, 0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plt.figure()
    ax.scatter(x, y, index, marker='o', s=5, c=index, cmap='summer')
    # ax.legend()
    plt.show()
    return None


def draw_speed_streamplot(stream_map, n_scale=3, plt_density=5):
    # max(stream_map[:, 1])
    # max(stream_map[:, 2])

    max_x = int(max(stream_map[:, 1]) / n_scale) + 1
    max_y = int(max(stream_map[:, 2]) / n_scale) + 1

    if abs(max_x - max_y) > 2:
        print('x and y have not benn modified. break.')
        # return None
    else:
        max_x_y = max(max_x, max_y)
        speed_x_streamplot = np.zeros([max_x_y, max_x_y])
        speed_y_streamplot = np.zeros([max_x_y, max_x_y])
        speed_count_streamplot = np.ones([max_x_y, max_x_y])
        for K_0 in range(len(stream_map)):
            # y = stream_map[K_0, 2] / n_sacle
            # y = (max(orentation_np[:, 2]) - orentation_np[K_0, 2]) / n_sacle
            x = stream_map[K_0, 1] / n_scale
            y = (max(stream_map[:, 2]) - stream_map[K_0, 2]) / n_scale
            speed_x_streamplot[int(y), int(x)] += stream_map[K_0, 3]
            speed_y_streamplot[int(y), int(x)] -= stream_map[K_0, 4]
            # speed_y_streamplot[int(y), int(x)] += stream_map[K_0, 4]
            speed_count_streamplot[int(y), int(x)] += 1
        # for K_1 in range(speed_x_streamplot.shape[0]):
        #     for K_2
        speed_x_streamplot /= speed_count_streamplot
        speed_y_streamplot /= speed_count_streamplot
        print(speed_x_streamplot[:4, :4])

    print(speed_x_streamplot.shape)

    # w = 3
    Y, X = np.mgrid[0:max_x_y, 0:max_x_y]
    # U = -1 - X ** 2 + Y
    # V = 1 + X - Y ** 2
    # speed = np.sqrt(U ** 2 + V ** 2)
    # fig = plt.figure()
    U = speed_x_streamplot
    V = speed_y_streamplot
    plt.figure()
    plt.streamplot(X, Y, U, V, density=[plt_density, plt_density])
    # plt.set_title('Speed')
    plt.title('Speed')
    return None


def get_acceleration(speed):
    # time = 10
    # speed = get_speed(position_np, distance_threshold=10, frame_interval=time, save_interval=time)
    acceleration = np.copy(speed[1:, 0:7])
    # acceleration.shape
    acceleration[:, 5:7] = np.copy(speed[1:, 3:5] - speed[:-1, 3:5])
    print(acceleration.shape)
    return acceleration


def draw_acceleration_streamplot(acceleration, from_hour_analysis=0, to_hour_analysis=2,
                                 fps=25, save_interval=10, n_scale=3, plt_density=5):
    # max(stream_map[:, 1])
    # max(stream_map[:, 2])

    max_x = int(max(acceleration[:, 1]) / n_scale) + 1
    max_y = int(max(acceleration[:, 2]) / n_scale) + 1

    from_show_length = int(fps * 60 * 60 * from_hour_analysis / save_interval)
    to_show_length = int(fps * 60 * 60 * to_hour_analysis / save_interval)

    # for K_0 in range(from_show_length, to_show_length)

    if abs(max_x - max_y) > 2:
        print('x and y have not benn modified. break.')
        # return None
    else:
        print('Processing data.')
        max_x_y = max(max_x, max_y)
        acceleration_x_streamplot = np.zeros([max_x_y, max_x_y])
        acceleration_y_streamplot = np.zeros([max_x_y, max_x_y])
        acceleration_count_streamplot = np.ones([max_x_y, max_x_y])
        for K_0 in range(from_show_length, to_show_length):
            x = acceleration[K_0, 1] / n_scale
            y = (max(acceleration[:, 2]) - acceleration[K_0, 2]) / n_scale
            acceleration_x_streamplot[int(y), int(x)] += acceleration[K_0, 5]
            acceleration_y_streamplot[int(y), int(x)] -= acceleration[K_0, 6]
            # speed_y_streamplot[int(y), int(x)] += stream_map[K_0, 4]
            acceleration_count_streamplot[int(y), int(x)] += 1
        # for K_1 in range(speed_x_streamplot.shape[0]):
        #     for K_2
        acceleration_x_streamplot /= acceleration_count_streamplot
        acceleration_y_streamplot /= acceleration_count_streamplot
        print(acceleration_x_streamplot[:4, :4])

    print(acceleration_x_streamplot.shape)
    print('Begin to plot.')
    Y, X = np.mgrid[0:max_x_y, 0:max_x_y]
    U = acceleration_x_streamplot
    V = acceleration_y_streamplot
    plt.figure()
    plt.streamplot(X, Y, U, V, density=[plt_density, plt_density])
    plt.title('Acceleration')
    return None

#TODO(JZ)useless
def example_streamplot():
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2
    speed = np.sqrt(U ** 2 + V ** 2)

    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Varying Density')
    return None


def get_heatmap_time(position_np):
    '''
    随时间变化的heatmap
    活动频率，
    :return: figure
    '''
    # for
    return 1
# speed_time = 1  #frame
# 分析视频，得到position_np 并保存为npy
# def speed_to_stream_plot():
    # dsaf

# fig = plt.figure(figsize=(7, 9))
# gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

#  Varying density along a streamline
# ax0 = fig.add_subplot(gs[0, 0])
# ax0.streamplot(X, Y, U, V, density=[0.5, 1])
# ax0.set_title('Varying Density')

# c = speed_x_streamplot[:4,:4]
# c = speed_x_streamplot[:10,:10]
# c[c==None] = 0

# def heatmap_without_time_not_modified(position_np, if_show_pic=0):
#     #############
#     # heatmap without time
#     # have not modified
#     x_org = position_np[:, 0]
#     x_modified = x_org - min(x_org)
#     y_org = position_np[:, 1]
#     y_modified = y_org - min(y_org)
#
#     num_x_interval = int(max(x_modified)) + 1
#     num_y_interval = int(max(y_modified)) + 1
#     position_heatmap = np.zeros([num_y_interval, num_x_interval])
#     for K_0 in range(len(position_np)):
#         x = x_modified[K_0]
#         y = y_modified[K_0]
#         position_heatmap[int(y), int(x)] += 1

# max_count=160
# # if_show_pic=0
#
# len_position = len(position_np)





# def get_speed():
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
#
#
#
#     '''
#     return 1
#
# def get_heatmap_time(position_np):
#     '''
#     随时间变化的heatmap
#     活动频率，
#     :return: figure
#     '''
#     # for
#     return 1
# speed_time = 1  #frame

