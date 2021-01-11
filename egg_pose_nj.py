














import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv



def load_pose_csv_and_save_npy(csv_name='../data/test.csv', need_save=0):
    with open(csv_name, 'r') as f:
        reader = csv.reader(f)
        pose_list_disorder = list(reader)
        # print(type(reader))
        # for row in reader:
        #     print(row)

    pose_list = []

    for K_0 in range(1, len(pose_list_disorder)):
        pose_temp = pose_list_disorder[K_0]
        # x_all = []
        pose_lsit_temp = []
        pose_lsit_temp.append(int(pose_temp[0]))

        pose_lsit_temp.append(int(float(pose_temp[3])))  # x
        pose_lsit_temp.append(int(float(pose_temp[4])))  # y

        x_all = pose_temp[8]
        y_all = pose_temp[9]
        x_all_split = x_all.split(' ')
        y_all_split = y_all.split(' ')
        for x, y in zip(x_all_split, y_all_split):
            # print(x,y)
            pose_lsit_temp.append(int(float(x)))
            pose_lsit_temp.append(int(float(y)))
        pose_list.append(pose_lsit_temp)

    pose_disorder = np.array(pose_list)

    print(pose_disorder.shape)

    pose = pose_disorder[np.argsort(pose_disorder[:, 0]), :]

    pose_npy_name = csv_name[:-4]+ '.npy'
    if need_save==1:
        np.save(pose_npy_name, pose)
    return pose


def pose_to_orentation(pose, frame_interval = 10, use_int = 1):
    # len_position = len(pose)
    x_org = pose[:, 1]
    x_modified = x_org - min(x_org)
    y_org = pose[:, 2]
    y_modified = y_org - min(y_org)
    y_modified = y_modified * max(x_modified) / (max(y_org) - min(y_org))
    # len_position = len(pose)

    x_head_org = pose[:, 3]
    x_head_modified = x_head_org - min(x_org)
    y_head_org = pose[:, 4]
    y_head_modified = y_head_org - min(y_org)
    y_head_modified = y_head_modified * max(x_modified) / (max(y_org) - min(y_org))

    x_tail_org = pose[:, 7]
    x_tail_modified = x_tail_org - min(x_org)
    y_tail_org = pose[:, 8]
    y_tail_modified = y_tail_org - min(y_org)
    y_tail_modified = y_tail_modified * max(x_modified) / (max(y_org) - min(y_org))

    orentation = []
    for K_0 in range(0, len(pose)):

        x_body = x_modified[K_0]
        y_body = y_modified[K_0]

        x_head = x_head_modified[K_0]
        y_head = y_head_modified[K_0]

        x_tail = x_tail_modified[K_0]
        y_tail = y_tail_modified[K_0]

        if K_0 % frame_interval == 0:
            if use_int == 1:
                orentation.append(
                [pose[K_0, 0], int(x_body), int(y_body), int(x_head - x_tail), int(y_head - y_tail), ])
            else:
                orentation.append(
                    [pose[K_0, 0], x_body, y_body, x_head - x_tail, y_head - y_tail ])
            # x = pose[K_0]
            # y = y_modified[K_0]
            # x_1 = x_modified[K_0 + frame_interval]
            # y_1 = y_modified[K_0 + frame_interval]
            #
            # x_2 = x_modified[K_0 + 2 * frame_interval]
            # y_2 = y_modified[K_0 + 2 * frame_interval]
            #
            # x_3 = x_modified[K_0 + 3 * frame_interval]
            # y_3 = y_modified[K_0 + 3 * frame_interval]
            #
            # x_4 = x_modified[K_0 + 4 * frame_interval]
            # y_4 = y_modified[K_0 + 4 * frame_interval]

    orentation_np = np.asarray(orentation)
    print(orentation_np.shape)
    return orentation_np

def pose_to_orentation_by_part(pose, frame_interval = 10,y_oren_threshold=36,):
    # len_position = len(pose)
    x_org = pose[:, 1]
    x_modified = x_org - min(x_org)
    y_org = pose[:, 2]
    y_modified = y_org - min(y_org)
    y_modified = y_modified * max(x_modified) / (max(y_org) - min(y_org))
    # len_position = len(pose)

    x_head_org = pose[:, 3]
    x_head_modified = x_head_org - min(x_org)
    y_head_org = pose[:, 4]
    y_head_modified = y_head_org - min(y_org)
    y_head_modified = y_head_modified * max(x_modified) / (max(y_org) - min(y_org))

    x_tail_org = pose[:, 7]
    x_tail_modified = x_tail_org - min(x_org)
    y_tail_org = pose[:, 8]
    y_tail_modified = y_tail_org - min(y_org)
    y_tail_modified = y_tail_modified * max(x_modified) / (max(y_org) - min(y_org))

    orentation = []
    for K_0 in range(0, len(pose)):

        x_body = x_modified[K_0]
        y_body = y_modified[K_0]

        x_head = x_head_modified[K_0]
        y_head = y_head_modified[K_0]

        x_tail = x_tail_modified[K_0]
        y_tail = y_tail_modified[K_0]

        if K_0 % frame_interval == 0 and abs(y_head - y_tail)<y_oren_threshold and y_body>100:

            orentation.append(
                [pose[K_0, 0], int(x_body), int(y_body), int(x_head - x_tail), int(y_head - y_tail), ])
            # x = pose[K_0]
            # y = y_modified[K_0]
            # x_1 = x_modified[K_0 + frame_interval]
            # y_1 = y_modified[K_0 + frame_interval]
            #
            # x_2 = x_modified[K_0 + 2 * frame_interval]
            # y_2 = y_modified[K_0 + 2 * frame_interval]
            #
            # x_3 = x_modified[K_0 + 3 * frame_interval]
            # y_3 = y_modified[K_0 + 3 * frame_interval]
            #
            # x_4 = x_modified[K_0 + 4 * frame_interval]
            # y_4 = y_modified[K_0 + 4 * frame_interval]

    orentation_np = np.asarray(orentation)
    print(orentation_np.shape)
    return orentation_np



def draw_pose_streamplot(orentation_np, n_sacle=3, plt_density=5):
    # max(orentation_np[:, 1])
    # max(orentation_np[:, 2])

    max_x = int(max(orentation_np[:, 1]) / n_sacle) + 1
    max_y = int(max(orentation_np[:, 2]) / n_sacle) + 1
    if abs(max_x - max_y) > 2:
        print('x and y have not benn modified. break.')
        # return None
    else:
        max_x_y = max(max_x, max_y)
        pose_x_streamplot = np.zeros([max_x_y, max_x_y])
        pose_y_streamplot = np.zeros([max_x_y, max_x_y])
        pose_count_streamplot = np.ones([max_x_y, max_x_y])
        for K_0 in range(len(orentation_np)):
            x = orentation_np[K_0, 1] / n_sacle
            y = (max(orentation_np[:, 2])-orentation_np[K_0, 2]) / n_sacle

            pose_x_streamplot[int(y), int(x)] += orentation_np[K_0, 3]
            pose_y_streamplot[int(y), int(x)] -= orentation_np[K_0, 4]
            pose_count_streamplot[int(y), int(x)] += 1
        # for K_1 in range(speed_x_streamplot.shape[0]):
        #     for K_2
        pose_x_streamplot /= pose_count_streamplot
        pose_y_streamplot /= pose_count_streamplot
        print(pose_x_streamplot[:4, :4])

    print(pose_x_streamplot.shape)
    # w = 3
    Y, X = np.mgrid[0:max_x_y, 0:max_x_y]
    # U = -1 - X ** 2 + Y
    # V = 1 + X - Y ** 2
    # speed = np.sqrt(U ** 2 + V ** 2)
    # fig = plt.figure()
    U = pose_x_streamplot
    V = pose_y_streamplot
    plt.figure()
    plt.streamplot(X, Y, U, V, density=[plt_density, plt_density])
    # plt.set_title('Speed')
    plt.title('Pose')
    return None

def draw_orentation_scatterplot(orentation_np):
    threshold = 26

    x_body = orentation_np[:, 1]
    y_body = orentation_np[:, 2]
    y_body = max(y_body) -y_body

    x = orentation_np[:, 3]
    y = orentation_np[:, 4]
    y = -y
    index = orentation_np[:, 0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plt.figure()
    ax.scatter(x_body, y_body, index, marker='o', s=5, c=x, cmap='rainbow')
    # ax.legend()
    # ax.set_xlim(-threshold,threshold)
    # ax.set_ylim(-threshold,threshold)
    plt.title('x')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plt.figure()
    ax.scatter(x_body, y_body, index, marker='o', s=5, c=y, cmap='rainbow')
    # ax.legend()
    # ax.set_xlim(-threshold,threshold)
    # ax.set_ylim(-threshold,threshold)
    plt.title('y')
    plt.show()
    return None

# rainbow
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='plasma')
# plt.subplot(232)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='Oranges')
# plt.subplot(233)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='summer')
# plt.subplot(234)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='Spectral')
# plt.subplot(235)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='Set1')
# plt.subplot(236)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='rainbow')



def draw_orentation_scatterplot_by_part(orentation_np, fps=25):
    threshold = 30
    orentation_modified = np.copy(orentation_np)
    for K_0 in range(len(orentation_modified)):
        if abs(orentation_modified[K_0, 3]) > threshold or abs(orentation_modified[K_0, 4]) > threshold:
            print(orentation_np[K_0])
            orentation_modified[K_0, 3:5] = orentation_modified[K_0 - 1, 3:5]

    K = 0
    from_hour_analysis = K
    to_hour_analysis = K + 1
    from_show_length = fps * from_hour_analysis
    to_show_length = -1  # fps  * to_hour_analysis * 40
    print(orentation_modified.shape)

    x_body = orentation_modified[from_show_length:to_show_length, 1]
    y_body = orentation_modified[from_show_length:to_show_length, 2]
    y_body = max(y_body) - y_body

    x = orentation_modified[from_show_length:to_show_length, 3]
    y = orentation_modified[from_show_length:to_show_length, 4]
    y = -y
    index = orentation_modified[from_show_length:to_show_length, 0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x_body, y_body, index, marker='o', s=5, c=y, cmap='rainbow')
    # ax.colorbar()
    plt.show()

    plt.figure()
    plt.scatter(y_body, y)

    # x = orentation_modified[from_show_length:to_show_length, 3]
    # y = orentation_modified[from_show_length:to_show_length, 4]
    # # y = max(y) - y
    # y = -y
    # index = orentation_modified[from_show_length:to_show_length, 0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, index, label='parametric curve')
    ax.set_xlim(-threshold, threshold)
    ax.set_ylim(-threshold, threshold)
    plt.show()

    plt.figure()
    plt.scatter(x, y, marker='o', s=5, c=index, cmap='rainbow')
    plt.xlim([-threshold, threshold])
    plt.ylim([-threshold, threshold])
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plt.figure()
    ax.scatter(x, y, index, marker='o', s=5, c=index, cmap='rainbow')
    # ax.legend()
    ax.set_xlim(-threshold, threshold)
    ax.set_ylim(-threshold, threshold)
    plt.show()

    return None

# type(row)
#
#
# row
# row[9]