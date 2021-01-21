







import os
import cv2
import time
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
                if stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50 and stat[4] > 50 and stat[4] < 300:
                    x1, y1, w, h, area = stat
                    position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, w, h, area])
                    K_1 += 1
            if not K_1 == 1:
                print(K_0)
                print(stats)
            K_0 += 1
            if K_0 % (fps * 100) == 0:
                print('K_0:   ', K_0)
        else:
            break

    if K_0 >= num_frame - 10:
        print('Completed. Congratulations!')
    videoCapture.release()
    return position



# video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi'

# hours = 1/60
# edge = 10
# threshold = 70
# fps = 25
# edge_x = 6
# edge_y = 8
def get_position_by_background_and_save_pictures(video_name, hours=1/60, edge=[6, 8], threshold=70, fps=25,
                               to_path='../data/picture/train2020/', if_save_pic=0,
                                     background_interval=1000):
    to_path     = '../data/temp/fly_coco/6/coco/'
    to_path_ann = '../data/temp/fly_coco/6/ann/'

    draw_points = 1
    # os.path.isdir(feat_file):
    # os.path.isdir(to_path):
    # os.mkdir(to_path)

    edge_x = edge[0]
    edge_y = edge[1]

    if os.path.isdir(to_path):
        if len(os.listdir(to_path)) > 0 :
            print('The file fold is not empty')
    else:
        print('The file fold is not exist and will be created.')
        os.makedirs(to_path)

    if os.path.isdir(to_path_ann):
        if len(os.listdir(to_path_ann)) > 0:
            print('The file fold is not empty')
    else:
        print('The file fold is not exist and will be created.')
        os.makedirs(to_path_ann)



    start_time = time.time()
    if draw_points == 1:
        csv_name = '../data/video_CS_20201031_h_0_to_h_13/orientation/video_CS_20201031_h_0_to_h_13_552_713_239_447_4_pose.csv'
        pose_npy_name = csv_name[:-4] + '.npy'
        pose = np.load(pose_npy_name)


    videoCapture = cv2.VideoCapture(video_name)
    success, frame_temp = videoCapture.read()
    if success:
        y_max, x_max, n_cha = frame_temp.shape
    videoCapture.release()


    background = np.zeros([y_max, x_max, n_cha])
    videoCapture = cv2.VideoCapture(video_name)
    # num_frame = 180000

    num_frame = int(fps * 60 * 60 * hours)
    #   num_frame = 180000
    for K_0 in range(num_frame):
        success, frame_temp = videoCapture.read()
        # frame_temp = frame_temp[6:230, 144:368, 3]
        background = np.maximum(frame_temp, background)
        if K_0 % (fps * 100) == 0:
            end_time = time.time()
            # print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('bg computing:    K_0:   ', K_0)

    videoCapture.release()
    background = background.astype('uint8')
    # plt.imshow(background)


    num_frame = int(fps * 60 * 60 * hours)

    videoCapture = cv2.VideoCapture(video_name)
    # num_frame = 4001
    # num_frame = 225000
    start_time = time.time()
    for K_0 in range(num_frame):
        # while True:
        # K_0=0
        success, frame = videoCapture.read()
        if success and K_0 % (fps * 10) == 0:
            frame_1 = np.copy(frame)
            # y_max, x_max, n_cha = frame_1.shape
            # plt.figure()
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
            # plt.figure()
            # plt.imshow(frame_bi)
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_grey, connectivity=4)
            # print(stats)

            for i, stat in enumerate(stats):
                # stat=stats[1]
                if stat[4] > 50 and stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50:  # stat[4] > 100 and
                    x1, y1, w, h, area = stat
                    # position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])
                    x2, y2 = x1 + w, y1 + h
                    # print(y1 - edge,y2 + edge,x1 - edge,x2 + edge)
                    y1 = max(0, y1 - edge_y)
                    y2 = min(y_max, y2 + edge_y)
                    x1 = max(0, x1 - edge_x)
                    x2 = min(x_max, x2 + edge_x)

                    pose_x = pose[K_0, 3:12:2]
                    pose_y = pose[K_0, 4:13:2]


                    proof = 1
                    for K_4 in range(len(pose_x)):
                        if pose_x[K_4] >= x1 and pose_x[K_4] <= x2:
                            proof = proof * 1
                        else:
                            proof = proof * 0

                        if pose_y[K_4] >= y1 and pose_y[K_4] <= y2:
                            proof = proof * 1
                        else:
                            proof = proof * 0


                    if draw_points == 1:
                        if proof == 0 and draw_points == 1:
                            center = x1, y1  # pose[K_0, 3], pose[K_0, 4]  # , pose[K_0, 3]#x,y#
                            radius = 5  # int(radius)
                            # print(K_0,'      ',center,'___',y1,y2,'_',x1,x2,'\n')
                            # center = 0, 120
                            frame_2 = np.copy(frame_1)

                            cv2.circle(frame_2, center, radius, (255, 0, 0), -1)
                            # plt.imshow(frame_1)
                            center = x2, y2
                            radius = 5
                            cv2.circle(frame_2, center, radius, (255, 0, 0), -1)

                            img_fly = np.copy(frame_grey)
                            # img_fly_ann = np.copy(frame_clean[y1:y2, x1:x2, :])
                            img_fly_ann = np.copy(frame_clean)
                            plt.imshow(img_fly)
                            plt.imshow(img_fly_ann)
                            if if_save_pic == 1:
                                pic_name = str(K_0) + '-' + str(x1) + '_' + str(x2) + '_' \
                                           + str(y1) + '_' + str(y2) + '-' + '.jpg'
                                plt.imsave(to_path + pic_name, img_fly)
                                plt.imsave(to_path_ann + pic_name, img_fly_ann)
                    #     center = pose[K_0, 3], pose[K_0, 4]#, pose[K_0, 3]#x,y#
                    #     radius = 5  # int(radius)
                    #     print(K_0,'      ',center,'___',y1,y2,'_',x1,x2,'\n')
                    #     # center = 0, 120
                    #     frame_2 = np.copy(frame_1)
                    #
                    #     frame_2 = cv2.circle(frame_2, center, radius, (255, 0, 0), -1)
                    #     # plt.figure()
                    #     # plt.imshow(frame)
                    # center = x1,y1#pose[K_0, 3], pose[K_0, 4]  # , pose[K_0, 3]#x,y#
                    # radius = 5  # int(radius)
                    # # print(K_0,'      ',center,'___',y1,y2,'_',x1,x2,'\n')
                    # # center = 0, 120
                    # # frame_2 = np.copy(frame_1)
                    #
                    # cv2.circle(frame_1, center, radius, (255, 0, 0), -1)
                    # plt.imshow(frame_1)
                    # center = x2, y2
                    # radius = 5
                    # cv2.circle(frame_1, center, radius, (255, 0, 0), -1)
                    #
                    # img_fly = np.copy(frame_1)
                    # img_fly_ann = np.copy(frame_clean[y1:y2, x1:x2,:])

                    # img_fly_ann = np.copy(frame_grey[y1:y2, x1:x2,:])
                    #     # print(img_fly.shape)
                    #     # plt.figure()
                    #     # plt.imshow(img_fly)
                    #
                    # if if_save_pic == 1:
                    #     pic_name = str(K_0) + '-' + str(x1) + '_' + str(x2) + '_' \
                    #                + str(y1) + '_' + str(y2) + '-' + '.jpg'
                    #     # plt.imsave(to_path + pic_name, img_fly)
                    #     plt.imsave(to_path_ann + pic_name, img_fly_ann)
                    #
                    #
                    #
                    #     # plt.imsave(to_path+str(K_0)+'.bmp',frame)
                    # plt.imsave(to_path+str(K_0)+'_'+str(i)+'.jpg',img_fly)
                    # if K_1 == 0:
                    #     input = []

                    # input.append(process_single_picture(img_fly))
                    # if K_1 == 0:
                    #     onnx_input_h = 96
                    #     onnx_input_w = 128
                    #     print(img_fly.shape)
                    #     a = cv2.resize(img_fly, (onnx_input_w, onnx_input_h), interpolation=cv2.INTER_CUBIC)
                    #     # a = img_fly
                    # K_1 += 1
                    #
                    # # if K_1 % batch_size != 0:
                    # #     input.append(process_single_picture(img_fly))
                    #
                    # if K_1 % batch_size == 0:
                    #     img_data_by_batch = np.array(input)
                    #     result = sess.run(outputs, {"x:0": img_data_by_batch})
                    #     conf_map, paf = result
                    #     conf_by_batch = transform_infer_rusult(a, conf_map)
                    #     pose_result.append(conf_by_batch)
                    #     input = []

        if K_0 % (fps * 100) == 0:
            end_time = time.time()
            # print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('K_0:   ', K_0)
    videoCapture.release()
    return None


def get_position_by_background(video_name, hours=1/60, edge=[6, 8], threshold=70, fps=25,
                               to_path='../data/picture/train2020/', if_save_pic=0,
                                     background_interval=1000):


    if os.path.isdir(to_path):
        if len(os.listdir(to_path)) > 0 :
            print('The file fold is not empty and will stop')
            return None
    else:
        print('The file fold is not exist and will be created.')
        os.makedirs(to_path)


    edge_x = edge[0]
    edge_y = edge[1]

    position = []

    start_time = time.time()

    videoCapture = cv2.VideoCapture(video_name)
    success, frame_temp = videoCapture.read()
    if success:
        y_max, x_max, n_cha = frame_temp.shape
    videoCapture.release()


    background = np.zeros([y_max, x_max, n_cha])
    videoCapture = cv2.VideoCapture(video_name)
    # num_frame = 180000

    num_frame = int(fps * 60 * 60 * hours)
    #   num_frame = 180000
    for K_0 in range(num_frame):
        success, frame_temp = videoCapture.read()
        # frame_temp = frame_temp[6:230, 144:368, 3]
        background = np.maximum(frame_temp, background)
        if K_0 % (fps * 100) == 0:
            end_time = time.time()
            # print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('bg computing:    K_0:   ', K_0)

    videoCapture.release()
    background = background.astype('uint8')
    # plt.imshow(background)


    num_frame = int(fps * 60 * 60 * hours)

    videoCapture = cv2.VideoCapture(video_name)
    # num_frame = 4001
    # num_frame = 225000
    start_time = time.time()
    for K_0 in range(num_frame):
        # while True:
        # K_0=0
        success, frame = videoCapture.read()
        if success:     # and K_0 % (fps * 10) == 0:
            frame_1 = np.copy(frame)
            # y_max, x_max, n_cha = frame_1.shape
            # plt.figure()
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
            # plt.figure()
            # plt.imshow(frame_bi)
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_grey, connectivity=4)
            # print(stats)
            K_1 = 0
            for i, stat in enumerate(stats):
                # stat=stats[1]
                if stat[4] > 50 and stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50:
                    x1, y1, w, h, area = stat
                    K_1 += 1
                    position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])

                    x2, y2 = x1 + w, y1 + h
                    # print(y1 - edge,y2 + edge,x1 - edge,x2 + edge)
                    y1 = max(0, y1 - edge_y)
                    y2 = min(y_max, y2 + edge_y)
                    x1 = max(0, x1 - edge_x)
                    x2 = min(x_max, x2 + edge_x)




            # img_fly = np.copy(frame_grey)
            # # img_fly_ann = np.copy(frame_clean[y1:y2, x1:x2, :])
            # img_fly_ann = np.copy(frame_clean)
            # plt.imshow(img_fly)
            # plt.imshow(img_fly_ann)
            # if if_save_pic == 1:
            #     pic_name = str(K_0) + '-' + str(x1) + '_' + str(x2) + '_' \
            #                + str(y1) + '_' + str(y2) + '-' + '.jpg'
            #     plt.imsave(to_path + pic_name, img_fly)


                # plt.imsave(to_path_ann + pic_name, img_fly_ann)
                    #     center = pose[K_0, 3], pose[K_0, 4]#, pose[K_0, 3]#x,y#
                    #     radius = 5  # int(radius)
                    #     print(K_0,'      ',center,'___',y1,y2,'_',x1,x2,'\n')
                    #     # center = 0, 120
                    #     frame_2 = np.copy(frame_1)
                    #
                    #     frame_2 = cv2.circle(frame_2, center, radius, (255, 0, 0), -1)
                    #     # plt.figure()
                    #     # plt.imshow(frame)
                    # center = x1,y1#pose[K_0, 3], pose[K_0, 4]  # , pose[K_0, 3]#x,y#
                    # radius = 5  # int(radius)
                    # # print(K_0,'      ',center,'___',y1,y2,'_',x1,x2,'\n')
                    # # center = 0, 120
                    # # frame_2 = np.copy(frame_1)
                    #
                    # cv2.circle(frame_1, center, radius, (255, 0, 0), -1)
                    # plt.imshow(frame_1)
                    # center = x2, y2
                    # radius = 5
                    # cv2.circle(frame_1, center, radius, (255, 0, 0), -1)
                    #
                    # img_fly = np.copy(frame_1)
                    # img_fly_ann = np.copy(frame_clean[y1:y2, x1:x2,:])

                    # img_fly_ann = np.copy(frame_grey[y1:y2, x1:x2,:])
                    #     # print(img_fly.shape)
                    #     # plt.figure()
                    #     # plt.imshow(img_fly)
                    #
                    # if if_save_pic == 1:
                    #     pic_name = str(K_0) + '-' + str(x1) + '_' + str(x2) + '_' \
                    #                + str(y1) + '_' + str(y2) + '-' + '.jpg'
                    #     # plt.imsave(to_path + pic_name, img_fly)
                    #     plt.imsave(to_path_ann + pic_name, img_fly_ann)
                    #
                    #
                    #
                    #     # plt.imsave(to_path+str(K_0)+'.bmp',frame)
                    # plt.imsave(to_path+str(K_0)+'_'+str(i)+'.jpg',img_fly)
                    # if K_1 == 0:
                    #     input = []

                    # input.append(process_single_picture(img_fly))
                    # if K_1 == 0:
                    #     onnx_input_h = 96
                    #     onnx_input_w = 128
                    #     print(img_fly.shape)
                    #     a = cv2.resize(img_fly, (onnx_input_w, onnx_input_h), interpolation=cv2.INTER_CUBIC)
                    #     # a = img_fly
                    # K_1 += 1
                    #
                    # # if K_1 % batch_size != 0:
                    # #     input.append(process_single_picture(img_fly))
                    #
                    # if K_1 % batch_size == 0:
                    #     img_data_by_batch = np.array(input)
                    #     result = sess.run(outputs, {"x:0": img_data_by_batch})
                    #     conf_map, paf = result
                    #     conf_by_batch = transform_infer_rusult(a, conf_map)
                    #     pose_result.append(conf_by_batch)
                    #     input = []

        if K_0 % (fps * 100) == 0:
            end_time = time.time()
            print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('K_0:   ', K_0)
    videoCapture.release()
    return position


def get_position_by_background_old(video_name, hours=1/60, edge=[6, 8], threshold=70, fps=25,
                               to_path='../data/picture/train2020/', if_save_pic=0,
                                     background_interval=1000):


    if os.path.isdir(to_path):
        if len(os.listdir(to_path)) > 0 :
            print('The file fold is not empty and will stop')
            return None
    else:
        print('The file fold is not exist and will be created.')
        os.makedirs(to_path)


    edge_x = edge[0]
    edge_y = edge[1]

    position = []

    start_time = time.time()

    videoCapture = cv2.VideoCapture(video_name)
    success, frame_temp = videoCapture.read()
    if success:
        y_max, x_max, n_cha = frame_temp.shape
    videoCapture.release()


    background = np.zeros([y_max, x_max, n_cha])
    videoCapture = cv2.VideoCapture(video_name)
    # num_frame = 180000

    num_frame = int(fps * 60 * 60 * hours)
    #   num_frame = 180000
    for K_0 in range(num_frame):
        success, frame_temp = videoCapture.read()
        # frame_temp = frame_temp[6:230, 144:368, 3]
        background = np.maximum(frame_temp, background)
        if K_0 % (fps * 100) == 0:
            end_time = time.time()
            # print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('bg computing:    K_0:   ', K_0)

    videoCapture.release()
    background = background.astype('uint8')
    # plt.imshow(background)


    num_frame = int(fps * 60 * 60 * hours)

    videoCapture = cv2.VideoCapture(video_name)
    # num_frame = 4001
    # num_frame = 225000
    start_time = time.time()
    for K_0 in range(num_frame):
        # while True:
        # K_0=0
        success, frame = videoCapture.read()
        if success:     # and K_0 % (fps * 10) == 0:
            frame_1 = np.copy(frame)
            # y_max, x_max, n_cha = frame_1.shape
            # plt.figure()
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
            # plt.figure()
            # plt.imshow(frame_bi)
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_grey, connectivity=4)
            # print(stats)
            K_1 = 0
            for i, stat in enumerate(stats):
                # stat=stats[1]
                if stat[4] > 50 and stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50:
                    x1, y1, w, h, area = stat
                    K_1 += 1
                    position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])

                    x2, y2 = x1 + w, y1 + h
                    # print(y1 - edge,y2 + edge,x1 - edge,x2 + edge)
                    y1 = max(0, y1 - edge_y)
                    y2 = min(y_max, y2 + edge_y)
                    x1 = max(0, x1 - edge_x)
                    x2 = min(x_max, x2 + edge_x)




            # img_fly = np.copy(frame_grey)
            # # img_fly_ann = np.copy(frame_clean[y1:y2, x1:x2, :])
            # img_fly_ann = np.copy(frame_clean)
            # plt.imshow(img_fly)
            # plt.imshow(img_fly_ann)
            # if if_save_pic == 1:
            #     pic_name = str(K_0) + '-' + str(x1) + '_' + str(x2) + '_' \
            #                + str(y1) + '_' + str(y2) + '-' + '.jpg'
            #     plt.imsave(to_path + pic_name, img_fly)


                # plt.imsave(to_path_ann + pic_name, img_fly_ann)
                    #     center = pose[K_0, 3], pose[K_0, 4]#, pose[K_0, 3]#x,y#
                    #     radius = 5  # int(radius)
                    #     print(K_0,'      ',center,'___',y1,y2,'_',x1,x2,'\n')
                    #     # center = 0, 120
                    #     frame_2 = np.copy(frame_1)
                    #
                    #     frame_2 = cv2.circle(frame_2, center, radius, (255, 0, 0), -1)
                    #     # plt.figure()
                    #     # plt.imshow(frame)
                    # center = x1,y1#pose[K_0, 3], pose[K_0, 4]  # , pose[K_0, 3]#x,y#
                    # radius = 5  # int(radius)
                    # # print(K_0,'      ',center,'___',y1,y2,'_',x1,x2,'\n')
                    # # center = 0, 120
                    # # frame_2 = np.copy(frame_1)
                    #
                    # cv2.circle(frame_1, center, radius, (255, 0, 0), -1)
                    # plt.imshow(frame_1)
                    # center = x2, y2
                    # radius = 5
                    # cv2.circle(frame_1, center, radius, (255, 0, 0), -1)
                    #
                    # img_fly = np.copy(frame_1)
                    # img_fly_ann = np.copy(frame_clean[y1:y2, x1:x2,:])

                    # img_fly_ann = np.copy(frame_grey[y1:y2, x1:x2,:])
                    #     # print(img_fly.shape)
                    #     # plt.figure()
                    #     # plt.imshow(img_fly)
                    #
                    # if if_save_pic == 1:
                    #     pic_name = str(K_0) + '-' + str(x1) + '_' + str(x2) + '_' \
                    #                + str(y1) + '_' + str(y2) + '-' + '.jpg'
                    #     # plt.imsave(to_path + pic_name, img_fly)
                    #     plt.imsave(to_path_ann + pic_name, img_fly_ann)
                    #
                    #
                    #
                    #     # plt.imsave(to_path+str(K_0)+'.bmp',frame)
                    # plt.imsave(to_path+str(K_0)+'_'+str(i)+'.jpg',img_fly)
                    # if K_1 == 0:
                    #     input = []

                    # input.append(process_single_picture(img_fly))
                    # if K_1 == 0:
                    #     onnx_input_h = 96
                    #     onnx_input_w = 128
                    #     print(img_fly.shape)
                    #     a = cv2.resize(img_fly, (onnx_input_w, onnx_input_h), interpolation=cv2.INTER_CUBIC)
                    #     # a = img_fly
                    # K_1 += 1
                    #
                    # # if K_1 % batch_size != 0:
                    # #     input.append(process_single_picture(img_fly))
                    #
                    # if K_1 % batch_size == 0:
                    #     img_data_by_batch = np.array(input)
                    #     result = sess.run(outputs, {"x:0": img_data_by_batch})
                    #     conf_map, paf = result
                    #     conf_by_batch = transform_infer_rusult(a, conf_map)
                    #     pose_result.append(conf_by_batch)
                    #     input = []

        if K_0 % (fps * 100) == 0:
            end_time = time.time()
            print("Time used: ", end_time - start_time, 's    ', 'K_0: ', K_0)
            print('K_0:   ', K_0)
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


