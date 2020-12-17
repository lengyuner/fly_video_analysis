







import os
import cv2
import time
import numpy as np
import onnxruntime as rt
import matplotlib.pyplot as plt









def process_single_picture(img0, onnx_input_h=96, onnx_input_w=128):
    img1 = cv2.resize(img0, (onnx_input_w, onnx_input_h), interpolation=cv2.INTER_CUBIC)
    # if 3 in img0.shape:
    #     img2 = np.copy(img1)
    # else:
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img3 = img2.astype(np.float32) / 255.0
    # return input_image
    image_processed = np.transpose(img3, [2, 0, 1])
    return image_processed


def transform_infer_rusult(conf, onnx_input_h=96, onnx_input_w=128):
    pose_by_batch = []
    for K_1 in range(len(conf)):
        conf_transposed_all = np.transpose(conf[K_1], [2, 0, 1])
        # conf_transposed_all = conf[K_1]
        pose = []
        for K_2 in range(5):

            conf_all = np.abs(conf_transposed_all[K_2, :, :])
            # conf_all = np.abs(conf_transposed_all[:, :, K_2])
            heatmap_avg = cv2.resize(conf_all,
                                 (onnx_input_w, onnx_input_h),#(ori_image.shape[1], ori_image.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
            x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
            # center = y, x
            # radius = 3
            # cv2.circle(input_image, center, radius, (255, 0, 0), -1)
            # if K_1==0 and K_2==4:
            #     plt.figure()
            #     plt.imshow(input_image)
            pose.append([y, x])
        pose_by_batch.append(pose)

    # a=np.array(pose_by_batch)
    # print(a.shape)
    return pose_by_batch

def transform_infer_rusult_and_show(input_image, conf, onnx_input_h = 96, onnx_input_w = 128):
    pose_by_batch = []
    for K_1 in range(len(conf)):
        conf_transposed_all = np.transpose(conf[K_1], [2, 0, 1])
        # conf_transposed_all = conf[K_1]
        pose = []
        for K_2 in range(5):

            conf_all = np.abs(conf_transposed_all[K_2, :, :])
            # conf_all = np.abs(conf_transposed_all[:, :, K_2])
            heatmap_avg = cv2.resize(conf_all,
                                 (onnx_input_w, onnx_input_h),#(ori_image.shape[1], ori_image.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
            x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
            center = y, x
            radius = 3
            cv2.circle(input_image, center, radius, (255, 0, 0), -1)
            if K_1==0 and K_2==4:
                plt.figure()
                plt.imshow(input_image)
            pose.append([y, x])
        pose_by_batch.append(pose)

    a=np.array(pose_by_batch)
    print(a.shape)
    return pose_by_batch

# if find_all == 1:
#             conf_map_head = np.transpose(conf_map[0], [2, 0, 1])
#             for K_1 in range(5):
#                 conf_head = np.abs(conf_map_head[K_1, :, :])
#                 heatmap_avg = cv2.resize(conf_head,
#                                          (input_image.shape[1], input_image.shape[0]),
#                                          interpolation=cv2.INTER_CUBIC)
#                 x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
#                 center = y, x  # x, y      #int(x*8),int(y*8)    #y, x
#                 radius = 3  # int(radius)
#                 cv2.circle(input_image, center, radius, (1, 0, 0), -1)
#             plt.imsave(path_to + img_name, input_image)
#         find_head = 0
#         if find_head == 1:
#             # np.abs(conf_map[0, :, :])
#
#             conf_map_head = np.transpose(conf_map[0], [2, 0, 1])
#             conf_head=np.abs(conf_map_head[0, :, :])
#             # show_conf_map = np.abs(conf_map[0, :, :])
#             # plt.imshow(conf_head)
#             heatmap_avg = cv2.resize(conf_head,
#                                      (input_image.shape[1], input_image.shape[0]),
#                                      interpolation=cv2.INTER_CUBIC)
#             x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
#             # x, y = np.unravel_index(np.argmax(conf_head), conf_head.shape)
#             center = y,x#x, y      #int(x*8),int(y*8)    #y, x
#             radius = 3  # int(radius)
#             cv2.circle(input_image, center, radius, (1, 0, 0), -1)

# get_picture_by_batch

video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi'
# fps=25
# hours=1/6000
threshold = 90
edge = 10
onnx_eng_name='../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx'
def pose_estimation_by_batch(video_name, hours = 1, threshold = 90, fps = 25, edge = 10,
                         onnx_eng_name='../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx'):

    onnx_name = onnx_eng_name
    sess = rt.InferenceSession(onnx_name)
    outputs = ["Identity:0", "Identity_1:0"]
    batch_size = 4

    # position = []
    pose_result = []
    videoCapture = cv2.VideoCapture(video_name)
    K_2 = 0
    K_1 = 0
    K_0 = 0
    start_time = time.time()

    num_frame = int(fps * 60 * 60 * hours)
    for K_0 in range(num_frame):
    # while True:
    # K_0=0
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
            # plt.imshow(frame_3_body,'gray')
            frame_3_body = 255 - frame_bi.astype(np.uint8)
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_3_body, connectivity=4)

            for i, stat in enumerate(stats):
                # stat=stats[1]
                if stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50: #stat[4] > 100 and
                    x1, y1, h, w, area = stat
                    # position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])
                    x2, y2 = x1+w, y1+h
                    img_fly = np.copy(frame_1[(y1-edge):(y2+edge), (x1-edge):(x2+edge)])
                    if K_1 == 0:
                        input = []

                    input.append(process_single_picture(img_fly))
                    if K_1==0:
                        onnx_input_h = 96
                        onnx_input_w = 128
                        print(img_fly.shape)
                        a= cv2.resize(img_fly, (onnx_input_w, onnx_input_h), interpolation=cv2.INTER_CUBIC)
                        # a = img_fly
                    K_1 += 1

                    # if K_1 % batch_size != 0:
                    #     input.append(process_single_picture(img_fly))

                    if K_1 % batch_size == 0:
                        img_data_by_batch = np.array(input)
                        result = sess.run(outputs, {"x:0": img_data_by_batch})
                        conf_map, paf = result
                        conf_by_batch = transform_infer_rusult(a,conf_map)
                        pose_result.append(conf_by_batch)
                        input = []


            if K_0 % (25 * 100) == 0:
                end_time = time.time()
                print("Time used: ", end_time - start_time, 's')
                print('K_0:   ', K_0)
            # K_1 += 1
            # K_2+=1
            # if not K_1 == 1:
            #     print(K_0)
            #     print(stat)
            # K_0 += 1
            #
            # if K_0 == 0:
            #     input = []
            # if K_2 % batch_size == 0:
            #     pose_estimation_by_batch()
            #     K_2 = 0
        else:
            break

    if K_0 >= num_frame - 10:
        print('Completed. Congratulations!')
    videoCapture.release()
    return pose_result


fly_pose_estimation_by_batch = pose_estimation_by_batch(video_name, hours = 1/6000, threshold = 90, fps = 25, edge = 0,
                         onnx_eng_name='../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx')





























def save_picture_by_batch(video_name, hours = 1/60, threshold = 90, fps = 25,edge_x=20, edge_y = 15,
                         onnx_eng_name='../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx'):

    onnx_name = onnx_eng_name
    path_to = '../data/picture_fly/1/'
    sess = rt.InferenceSession(onnx_name)
    outputs = ["Identity:0", "Identity_1:0"]
    batch_size = 4

    # position = []
    pose_result = []
    videoCapture = cv2.VideoCapture(video_name)
    K_2 = 0
    K_1 = 0
    K_0 = 0
    start_time = time.time()

    num_frame = int(fps * 60 * 60 * hours)
    for K_0 in range(num_frame):
    # while True:
    # K_0=0
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
            # plt.imshow(frame_3_body,'gray')
            frame_3_body = 255 - frame_bi.astype(np.uint8)
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(frame_3_body, connectivity=4)

            for i, stat in enumerate(stats):
                # stat=stats[1]
                if stat[4] < 300 and stat[2] < 50 and stat[2] > 3 and stat[3] > 3 and stat[3] < 50: #stat[4] > 100 and
                    x1, y1, h, w, area = stat
                    # position.append([centroid[i][0], centroid[i][1], K_0, K_1, x1, y1, h, w, area])
                    x2, y2 = x1+w, y1+h
                    # img_fly = np.copy(frame_1[(y1-edge):(y2+edge), (x1-edge):(x2+edge)])
                    # print(x1,x2,'\t',y1,y2)
                    # print(centroid[i][1],centroid[i][0],'\n')
                    cent_x = int(centroid[i][0])
                    cent_y = int(centroid[i][1])
                    # edge_x = 20
                    # edge_y = 15
                    img_fly = np.copy(frame_1[(cent_y - edge_y):(cent_y + edge_y), (cent_x - edge_x):(cent_x + edge_x)])
                    # if K_1 == 0:
                    #     input = []


                    # input.append(process_single_picture(img_fly))
                    # if K_1==0:
                    onnx_input_h = 96
                    onnx_input_w = 128
                    if K_0 % (100*25) ==0:
                        print(x1,x2,'\t',y1,y2)
                        print(centroid[i][1],centroid[i][0],'\n')
                        print(img_fly.shape)
                        pic_by_batch = cv2.resize(img_fly, (onnx_input_w, onnx_input_h), interpolation=cv2.INTER_CUBIC)
                        plt.imsave(path_to+str(K_0)+'.jpg', pic_by_batch)
                        # a = img_fly
                    K_1 += 1

                    # if K_1 % batch_size != 0:
                    #     input.append(process_single_picture(img_fly))

                    # if K_1 % batch_size == 0:
                    #     img_data_by_batch = np.array(input)
                    #     result = sess.run(outputs, {"x:0": img_data_by_batch})
                    #     conf_map, paf = result
                    #     conf_by_batch = transform_infer_rusult(a,conf_map)
                    #     pose_result.append(conf_by_batch)
                    #     input = []


            if K_0 % (25 * 100) == 0:
                end_time = time.time()
                print("Time used: ", end_time - start_time, 's')
                print('K_0:   ', K_0)
            # K_1 += 1
            # K_2+=1
            # if not K_1 == 1:
            #     print(K_0)
            #     print(stat)
            # K_0 += 1
            #
            # if K_0 == 0:
            #     input = []
            # if K_2 % batch_size == 0:
            #     pose_estimation_by_batch()
            #     K_2 = 0
        else:
            break

    if K_0 >= num_frame - 10:
        print('Completed. Congratulations!')
    videoCapture.release()
    return pose_result



# save_picture_by_batch(video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi',
#                         hours =2, threshold = 90, fps = 25, edge_x=20, edge_y = 15,
#                          onnx_eng_name='../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx')

# def pose_estimation_by_batch(pic,model):
#     # 调用
#     print('a')
#     return 1


# img_fly = img0
# onnx_name = '../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx'
# sess = rt.InferenceSession(onnx_name)
# K_0 = 0
# batch_size = 64
# if K_0 == 0:
#     input=[]
# if K_0 % batch_size != 0:
#     input.append(process_single_picture(img_fly))
# if K_0 % batch_size == 0:
#     img_data_by_batch = np.array(input)
#     outputs = ["Identity:0", "Identity_1:0"]
#     result = sess.run(outputs, {"x:0": img_data_by_batch})
#     conf_map, paf = result
#     conf_by_batch = transform_infer_rusult(conf_map)
#     # result = sess.run(outputs, {"x:0": img_data})
#     input = []






# def transform_img_into_batch_input(onnx_input_h=96, onnx_input_w=128):

    # b2 = []
    # b2.append(input_image)
    # b2.append(input_image)
    # # b3=b2.numpy()
    # b3 = np.array(b2)
    # b3.shape





# img_fly = img0
# onnx_name = '../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx'
# sess = rt.InferenceSession(onnx_name)
# K_0 = 0
# batch_size = 64
# if K_0 == 0:
#     input=[]
# if K_0 % batch_size != 0:
#     input.append(process_single_picture(img_fly))
# if K_0 % batch_size == 0:
#     img_data_by_batch = np.array(input)
#     outputs = ["Identity:0", "Identity_1:0"]
#     result = sess.run(outputs, {"x:0": img_data_by_batch})
#     conf_map, paf = result
#     conf_by_batch = transform_infer_rusult(conf_map)
#     # result = sess.run(outputs, {"x:0": img_data})
#     input = []





# input=[]

# for K in range(64):
#     input.append(process_single_picture())
#     # b3=b2.numpy()
#     b3 = np.array(b2)

# def infer_orentation(onnx_name, img_bybatch, onnx_input_h=96, onnx_input_w=128, outputs = ["Identity:0", "Identity_1:0"]):
#     onnx_name = '../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx'
#     sess = rt.InferenceSession(onnx_name)
#
#     # start_time = time.time()
#     result = sess.run(outputs, {"x:0": img_data})
#     # end_time = time.time()
#     # print("Time used: ", end_time - start_time, 's')
#     conf_map, paf = result




# read pic
# img0 = cv2.imread("../data/models/000000000399.jpg")
# print(img0.shape)
# # img1 = cv2.resize(img0, (432, 368))
# # resize the pic
# # img1 = cv2.resize(img0, (216, 184))
# img1 = cv2.resize(img0, (128, 96))
# # plt.imshow(img1)
# print(img1.shape)
# ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# input_image = ori_image.astype(np.float32) / 255.0
# # input_image = np.transpose(input_image, [2, 0, 1])
#
# # if (model.data_format == "channels_first"):
# #     input_image = np.transpose(input_image, [2, 0, 1])
# # img_c, img_h, img_w = input_image.shape
#
# img_data = input_image[np.newaxis, :, :, :]
# a1= np.copy(img_data)
# # b=np.hstack((a1,a1))
# b2=np.vstack((a1,a1,a1))
#
# b2=[]
# b2.append(input_image)
# b2.append(input_image)
# # b3=b2.numpy()
# b3 = np.array(b2)
# b3.shape
# # b= [input_image,a1]
#
# # using tool `Netron` to get the name of input node and output node
# outputs = ["Identity:0","Identity_1:0"]
# result = sess.run(outputs, {"x:0": b3})
# print(len(result))
# print(len(result[0]))
# conf_map,paf = result
# img_data.shape
# conf_map.shape
#
# for K in range(len(conf_map)):
#     conf_head = np.abs(conf_map[6, :, :, 0])
#     heatmap_avg = cv2.resize(conf_head,
#                              (ori_image.shape[1], ori_image.shape[0]),
#                              interpolation=cv2.INTER_CUBIC)
#     x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
#     center = y, x






# start_time = time.time()
# for K in range(400):
#     result = sess.run(outputs, {"x:0": img_data})
# end_time = time.time()
# print("Time used: ", end_time - start_time, 's')
#
# conf_map,paf = result
#
#
# conf_head=np.abs(conf_map[6,:,:,0])
# heatmap_avg = cv2.resize(conf_head,
#                          (ori_image.shape[1], ori_image.shape[0]),
#                          interpolation=cv2.INTER_CUBIC)
# x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
# center = y, x
# radius = 3  # int(radius)
# cv2.circle(ori_image, center, radius, (255, 0, 0), 2)
#
#
# plt.imshow(ori_image)
#
# plt.imsave("output.jpg", ori_image)

# plt.imsave(path_to + img_name, ori_image)





##############################################
##############################################
##############################################


# save_all_pic = 1
# if save_all_pic == 1:
#     path_from = 'C:/Users/ps/Desktop/djz/hyperpose/data/fly/val2020/'
#     path_from = 'C:/Users/ps/Desktop/djz/hyperpose/data/fly/train2020/'
#     # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_20201121_1621/val/'
#     # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_20201121_1621/val_head_output/'
#     # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_Openpose_Mobilenetv2_20201124_2030/val_output/'
#     # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_Openpose_Mobilenetv2_20201124_2030/val_head_output/'
#     # C:\Users\ps\Desktop\djz\hyperpose\save_dir\USERDEF_Openpose_Resnet18_20201130_1642\model_dir
#     # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/rubbish/val_head_output/'
#     # path_ta = 'C:\Users\ps\Desktop\djz\hyperpose\save_dir\USERDEF_Openpose_Resnet18_20201130_2259\val_head_output/'
#     # path_to = './save_dir/USERDEF_Openpose_Resnet18_20201202_2325/train_all/'
#
#     path_to = '../data/video_CS_20201031_h_0_to_h_13/orientation/'
#     read_name = os.listdir(path_from)
#     number_of_pic = len(read_name)
#
#     start_time = time.time()
#     for K_0 in range(number_of_pic):
#         img_name = read_name[K_0]
#         img0 = cv2.imread(path_from+img_name)
#         img1 = cv2.resize(img0, (128, 96))
#         # plt.imshow(img1)
#         ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#         input_image = ori_image.astype(np.float32) / 255.0
#         # if (model.data_format == "channels_first"):
#         #     input_image = np.transpose(input_image, [2, 0, 1])
#         img_c, img_h, img_w = input_image.shape
#
#         conf_map, paf_map = model.infer(input_image[np.newaxis, :, :, :])
#         find_all_part = 0
#         if find_all_part == 1:
#             visualize = Model.get_visualize(Config.MODEL.Openpose)
#             # def visualize(img,conf_map,paf_map,save_name="maps",save_dir="./save_dir/vis_dir",data_format="channels_first",save_tofile=True):
#             vis_parts_heatmap, vis_limbs_heatmap = visualize(input_image, conf_map[0], paf_map[0],
#                                                          save_name=img_name[:-4],
#                                                          save_dir=path_to,
#                                                          data_format=model.data_format )
#
#         find_all = 0
#         if find_all == 1:
#             conf_map_head = np.transpose(conf_map[0], [2, 0, 1])
#             for K_1 in range(5):
#                 conf_head = np.abs(conf_map_head[K_1, :, :])
#                 heatmap_avg = cv2.resize(conf_head,
#                                          (input_image.shape[1], input_image.shape[0]),
#                                          interpolation=cv2.INTER_CUBIC)
#                 x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
#                 center = y, x  # x, y      #int(x*8),int(y*8)    #y, x
#                 radius = 3  # int(radius)
#                 cv2.circle(input_image, center, radius, (1, 0, 0), -1)
#             plt.imsave(path_to + img_name, input_image)
#         find_head = 0
#         if find_head == 1:
#             # np.abs(conf_map[0, :, :])
#
#             conf_map_head = np.transpose(conf_map[0], [2, 0, 1])
#             conf_head=np.abs(conf_map_head[0, :, :])
#             # show_conf_map = np.abs(conf_map[0, :, :])
#             # plt.imshow(conf_head)
#             heatmap_avg = cv2.resize(conf_head,
#                                      (input_image.shape[1], input_image.shape[0]),
#                                      interpolation=cv2.INTER_CUBIC)
#             x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
#             # x, y = np.unravel_index(np.argmax(conf_head), conf_head.shape)
#             center = y,x#x, y      #int(x*8),int(y*8)    #y, x
#             radius = 3  # int(radius)
#             cv2.circle(input_image, center, radius, (1, 0, 0), -1)
#             plt.imsave(path_to + img_name, input_image)
#             # plt.imsave(img_name, input_image)
#
#         vis_and_head = 0
#         if vis_and_head==1:
#
#             img, conf_map, paf_map = input_image, conf_map[0], paf_map[0]
#             save_name = img_name[:-4]
#             save_dir = path_to
#             data_format = model.data_format
#             save_tofile = True
#             if (type(img) != np.ndarray):
#                 img = img.numpy()
#             if (type(conf_map) != np.ndarray):
#                 conf_map = conf_map.numpy()
#             if (type(paf_map) != np.ndarray):
#                 paf_map = paf_map.numpy()
#
#             if (data_format == "channels_last"):
#                 conf_map = np.transpose(conf_map, [2, 0, 1])
#                 paf_map = np.transpose(paf_map, [2, 0, 1])
#             elif (data_format == "channels_first"):
#                 img = np.transpose(img, [1, 2, 0])
#             os.makedirs(save_dir, exist_ok=True)
#             ori_img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
#             vis_img = ori_img.copy()
#             fig = plt.figure(figsize=(8, 8))
#             # show input image
#             a = fig.add_subplot(2, 2, 1)
#             a.set_title("input image")
#             plt.imshow(vis_img)
#             # show conf_map
#             show_conf_map = np.abs(conf_map[0, :, :])
#             #np.amax(np.abs(conf_map[:-1, :, :]), axis=0)
#             a = fig.add_subplot(2, 2, 3)
#             a.set_title("conf_map")
#             plt.imshow(show_conf_map)
#             # show paf_map
#             show_paf_map = np.amax(np.abs(paf_map[0:2, :, :]), axis=0)
#             a = fig.add_subplot(2, 2, 4)
#             a.set_title("paf_map")
#             plt.imshow(show_paf_map)
#             # save
#             if (save_tofile):
#                 plt.savefig(f"{save_dir}/{save_name}_visualize.png")
#                 plt.close('all')
#
#         if K_0%10 == 0:
#             print(K_0)
#     end_time = time.time()
#     print("Time used: ", end_time - start_time, 's')



##############################################
##############################################
##############################################

# import cv2
# import time
# import numpy as np
# import onnxruntime as rt
# import matplotlib.pyplot as plt
#
# onnx_name = '../data/models/USERDEF_Openpose_Resnet18_20201202_2325.onnx'
#
#
# sess = rt.InferenceSession(onnx_name)
#
# # read pic
# img0 = cv2.imread("../data/models/000000000399.jpg")
# print(img0.shape)
# # img1 = cv2.resize(img0, (432, 368))
# # resize the pic
# # img1 = cv2.resize(img0, (216, 184))
# img1 = cv2.resize(img0, (128, 96))
# # plt.imshow(img1)
# print(img1.shape)
# ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# input_image = ori_image.astype(np.float32) / 255.0
# input_image = np.transpose(input_image, [2, 0, 1])
# # if (model.data_format == "channels_first"):
# #     input_image = np.transpose(input_image, [2, 0, 1])
# # img_c, img_h, img_w = input_image.shape
# img_data = input_image[np.newaxis, :, :, :]
#
# # using tool `Netron` to get the name of input node and output node
# outputs = ["Identity:0","Identity_1:0"]
#
#
# start_time = time.time()
# for K in range(400):
#     result = sess.run(outputs, {"x:0": img_data})
# end_time = time.time()
# print("Time used: ", end_time - start_time, 's')
#
# conf_map,paf = result
#
#
# conf_head=np.abs(conf_map[0,:,:,0])
# heatmap_avg = cv2.resize(conf_head,
#                          (ori_image.shape[1], ori_image.shape[0]),
#                          interpolation=cv2.INTER_CUBIC)
# x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
# center = y, x
# radius = 3  # int(radius)
# cv2.circle(ori_image, center, radius, (255, 0, 0), 2)
#
#
# plt.imshow(ori_image)
#
# plt.imsave("output.jpg", ori_image)
#
# # plt.imsave(path_to + img_name, ori_image)