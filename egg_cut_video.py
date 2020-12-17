



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def cut_video(from_name = '../CS (201031).MTS', to_name = '../data/video_CS_20201031.avi',
              x1=556, x2=708, y1=242, y2=446, fps = 25, mins = 2,
              if_save_video = 0, if_save_pic = 0, if_get_one_pic=0):
    videoCapture = cv2.VideoCapture(from_name)

    size = (x2 - x1, y2 - y1)  # 保存视频的大小 #WH
    seconds = 60 * mins
    videoWriter = cv2.VideoWriter(to_name,
                                  cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)

    # if if_get_one_pic == 1:
    #     num_frame=1
    # else:
    #     num_frame=int(fps*seconds)
    num_frame = int(fps * seconds)
    for i in range(num_frame):
        success, frame_org = videoCapture.read()
        if success:
            frame = frame_org[y1:y2, x1:x2, :]
            # print(frame_1.shape)

            if if_save_video == 1:
                videoWriter.write(frame)
                if i % (fps * 10) == 0:
                    print('i = ', i, '      ', )


            if if_save_pic == 1:
                if not os.path.isdir(to_name[:-4]):
                    os.mkdir(to_name[:-4])

                frame_bi = np.copy(frame)
                frame_bi = cv2.cvtColor(frame_bi, cv2.COLOR_RGB2GRAY)
                frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25,
                                                 5)
                plt.imsave(to_name[:-4] + '/' + str(i) + '.jpg', frame_bi[2:-2, 2:-2])

                if i % (fps * 10) == 0:
                    print('i = ', i, '      ', )

        else:
            break
    videoCapture.release()
    return None




def get_video_of_position_heatmap(from_name='../CS (201031).MTS', to_name='../data/video_CS_20201031.avi',
                                fps=25, mins=120, interval=25 * 60,
                                if_save_video=1, if_save_pic=0, if_get_one_pic=0):
    videoCapture = cv2.VideoCapture(from_name)
    # size = (x2 - x1, y2 - y1)  # 保存视频的大小 #WH
    size = (videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT), videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))

    seconds = 60 * mins
    videoWriter = cv2.VideoWriter(to_name,
                                  cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)

    num_frame = int(fps * seconds)
    for K_0 in range(num_frame):
        success, frame_org = videoCapture.read()
        if success:
            if K_0 % interval == 0:
                frame = np.copy(frame_org)

                if if_save_video == 1:
                    videoWriter.write(frame)
                    if K_0 % (interval * 100) == 0:
                        print('i = ', K_0, '      ', )

        else:
            break
    videoCapture.release()
    return None


def cut_video_batch_process(from_name = '../CS (201031).MTS', task_name = 'video_CS_20201031',
              x_y = [[556,708,242,446,5]], edge=20 , fps = 25, mins = 2, if_save_video = 0):
    # to_path = '../data/video_CS_20201031/'
    to_path = os.path.join('../data/', task_name)
    videoCapture = cv2.VideoCapture(from_name)
    len_xy = len(x_y)
    size = []
    to_name = []
    videoWriter = []
    # hour_length
    for K_0 in range(len_xy):
        assert len(x_y[K_0]) == 5, 'every unit of x_y must be 5 length'
        x1, x2, y1, y2, position = x_y[K_0]
        x1, x2, y1, y2 = x1-edge, x2+edge, y1-edge, y2+edge
        size.append((x2 - x1, y2 - y1))# 保存视频的大小 #WH
        # size[K_0] = (x2 - x1, y2 - y1)
        video_name = task_name + '_' + str(x1) + '_' + str(x2)+ '_' \
                     + str(y1) + '_' + str(y2) + '_' + str(position) +'.avi'
        if not os.path.isdir(to_path):
            os.mkdir(to_path)
        to_name.append( os.path.join(to_path, video_name))
        print(to_name[K_0])
        videoWriter.append(cv2.VideoWriter(to_name[K_0],
                                      cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size[K_0]))
    seconds = 60 * mins
    num_frame = int(fps * seconds)
    for K_1 in range(num_frame):
        success, frame_org = videoCapture.read()
        if success:
            # print(frame_1.shape)
            for K_2 in range(len_xy):
                x1, x2, y1, y2, position = x_y[K_2]
                x1, x2, y1, y2 = x1 - edge, x2 + edge, y1 - edge, y2 + edge
                frame = frame_org[y1:y2, x1:x2, :]
                if if_save_video == 1:
                    videoWriter[K_2].write(frame)
            if K_1 % (fps * 10) == 0:
                print('Have processed: ', K_1, '/', num_frame, '    %.2f%%'%(100*K_1/num_frame)  )
            K_4 = K_1
        else:
            break
    videoCapture.release()
    if K_4 >= num_frame - 10:
        print('Completed. Congratulations!')
    for K_3 in range(len_xy):
        videoWriter[K_3].release()
    return None



#TODO(JZ)uncompleted, using formatfactory is faster
def transform_video(from_name = '../CS (201031).MTS', to_name = '../data/video_CS_20201031.avi',
              x1=556, x2=708, y1=242, y2=446, fps = 25, mins = 2,
              if_save_video = 0, if_save_pic = 0, if_get_one_pic=0):
    videoCapture = cv2.VideoCapture(from_name)

    size = (x2 - x1, y2 - y1)  # 保存视频的大小 #WH
    seconds = 60 * mins
    videoWriter = cv2.VideoWriter(to_name,
                                  cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)

    # if if_get_one_pic == 1:
    #     num_frame=1
    # else:
    #     num_frame=int(fps*seconds)
    num_frame = int(fps * seconds)
    for i in range(num_frame):
        success, frame_org = videoCapture.read()
        if success:
            frame = frame_org[y1:y2, x1:x2, :]
            # print(frame_1.shape)

            if if_save_video == 1:
                videoWriter.write(frame)
                if i % (fps * 10) == 0:
                    print('i = ', i, '      ', )


            if if_save_pic == 1:
                if not os.path.isdir(to_name[:-4]):
                    os.mkdir(to_name[:-4])

                frame_bi = np.copy(frame)
                frame_bi = cv2.cvtColor(frame_bi, cv2.COLOR_RGB2GRAY)
                frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25,
                                                 5)
                plt.imsave(to_name[:-4] + '/' + str(i) + '.jpg', frame_bi[2:-2, 2:-2])

                if i % (fps * 10) == 0:
                    print('i = ', i, '      ', )

        else:
            break
    videoCapture.release()
    videoWriter.release()
    return None

def get_processed_pic(from_name='../CS (201031).MTS', to_name='../data/video_CS_20201031.avi',
              x1=556, x2=708, y1=242, y2=446, fps=25, mins=2,
              if_save_video=0, if_save_pic=0, if_get_one_pic=1):
    videoCapture = cv2.VideoCapture(from_name)

    num_frame = 1
    for i in range(num_frame):
        success, frame_org = videoCapture.read()
        if success:
            frame = frame_org[y1:y2, x1:x2, :]
            # print(frame_1.shape)
            # if not os.path.isdir(to_name[:-4]):
            #     os.mkdir(to_name[:-4])
            if if_get_one_pic == 1:
                frame_bi = np.copy(frame)
                frame_bi = cv2.cvtColor(frame_bi, cv2.COLOR_RGB2GRAY)
                frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 25, 5)
        else:
            break
    videoCapture.release()
    return frame_bi[2:-2, 2:-2]


def save_pic_from_video(from_name = '../data/video_CS_20201031.avi', to_path = '../data/video_CS_20201031',
              fps = 25, mins = 2, save_interval=1000,
              if_save_video = 0, if_save_pic = 0, if_get_one_pic=0):
    videoCapture = cv2.VideoCapture(from_name)

    # size = (x2 - x1, y2 - y1)  # 保存视频的大小 #WH
    seconds = 60 * mins
    num_frame = int(fps * seconds)

    for i in range(num_frame):
        success, frame = videoCapture.read()
        if success:
            if if_save_pic == 1:
                if not os.path.isdir(to_path):
                    os.mkdir(to_path)
                if i%save_interval == 0:
                    frame_bi = np.copy(frame)
                    frame_bi = cv2.cvtColor(frame_bi, cv2.COLOR_RGB2GRAY)
                    frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 25, 5)
                    plt.imsave(to_path + '/' + str(i) + '.jpg', frame_bi[2:-2, 2:-2])
                    print('i = ', i, '      ', )
        else:
            break
    videoCapture.release()
    return None













# os.path
# os.getcwd()
# print(cv2.__version__)
#
# from_name = '../CS (201031).MTS'
#
# to_name = '../data/video_CS_20201031.avi'
#
# videoCapture = cv2.VideoCapture(from_name)
#
#
# x1, x2 = 556,708
# y1, y2 = 242,446
# fps = 25  # 保存视频的帧率
# size = (x2-x1, y2-y1) # 保存视频的大小 #WH
# mins = 2
# seconds = 60*mins
# videoWriter = cv2.VideoWriter(to_name,
#                               cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
# # videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 50)
# # videoWriter = cv2.VideoWriter(OUTPUT_FILE,
# #                               cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
#
# i=0
# for i in range(10):    #    (fps*seconds): #
#     success, frame_org = videoCapture.read()
#     if success:
#         frame = frame_org[y1:y2, x1:x2, :]
#         # print(frame_1.shape)
#         val_write=0
#         if val_write==1:
#             videoWriter.write(frame)
#         val_pic_write = 1
#         if not os.path.isdir(to_name[:-4]):
#             os.mkdir(to_name[:-4])
#         if val_pic_write==1:
#             frame_bi = np.copy(frame)
#             frame_bi = cv2.cvtColor(frame_bi, cv2.COLOR_RGB2GRAY)
#             frame_bi = cv2.adaptiveThreshold(frame_bi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
#
#             plt.imsave(to_name[:-4] + '/' + str(i) + '.jpg', frame_bi[2:-2,2:-2])
#         if i % (fps * 10) == 0:
#             print('i = ', i, '      ', )
#     else:
#         break
# videoCapture.release()



























# cv2.imshow('a',frame_1)
# cv2.destroyAllWindows()
#
#
#
#
# ##################################################
#
#
# # videoCapture = cv2.VideoCapture('IMG_2789.MOV')
#
# name = '../CS (201031).MTS'
# OUTPUT_FILE='../data/video_CS_20201031.avi'
#
# videoCapture = cv2.VideoCapture(name)
#
# # print(frame.shape)
# # Out[10]: (1024, 1280, 3)
# fps = 30  # 保存视频的帧率
# # size = (1024, 1280)  # 保存视频的大小
# size = (1280, 1024)
# # videoWriter = cv2.VideoWriter('video4_20201024.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
#
#
#
# videoWriter = cv2.VideoWriter(OUTPUT_FILE,
#                               cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
#
# # OUTPUT_FILE='C:/Users/ps/Desktop/djz/video4_20201024.avi'
# # (width, height)=size
# # videoWriter = cv2.VideoWriter(OUTPUT_FILE,
# #                 cv2.VideoWriter_fourcc('I', '4', '2', '0'),
# #                 30, # fps
# #                 (1280, 1024))
#                 # (width, height))
# i = 0
# # if frame.shape[0]
# while True:
#     success, frame = videoCapture.read()
#     plt.imshow(frame)
#     i += 1
#
#     if (i >= 0 and i <= fps * 10):
#         print('i = ', i)
#         videoWriter.write(frame)
#     else:
#         print('end')
#         break
#
# import cv2
#
# cap = cv2.VideoCapture('XXX.avi')  # 返回一个capture对象
# cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # 设置要获取的帧号
# a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
# cv2.imshow('b', b)
# cv2.waitKey(1000)
# plt.imshow(frame_bi)
# import cv2
#
# # img1 = cv2.imread('./Image/letter.png', cv2.IMREAD_GRAYSCALE)
# #
# # img1 = cv2.resize(img1, (300, 300), interpolation=cv2.INTER_AREA)
# # cv2.imshow('img1', img1)
#
# import numpy as np
#
# img1 = np.copy(frame_1)
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
# res1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
# res2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)
# plt.figure(1)
# plt.subplot(121)
# plt.imshow(res1)
# plt.subplot(122)
# plt.imshow(res2)
#
# plt.imsave(str(i)+'jpg',res1)
# cv2.imshow('res1', res1)
# cv2.imshow('res2', res2)


