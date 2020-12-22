





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy

import numpy as np
from mayavi import mlab

# from sympy import *
# from mayavi.mlab import *



# def Demo():
#     x, y, z = np.mgrid[0:1:20j, 0:1:20j, 0:1:20j]
#
#     u = -y
#     v = x
#     w = z  # 矢量场三坐标分量表达式
#
#     x = sympy.Symbol('x')  # 引入符合x
#     expr = sympy.sin(x) / x  # 表达式形式
#     f = sympy.lambdify(x, expr, "numpy")  # 表达式中x符合用数组代替
#     data = np.linspace(1, 10, 10000)  # 取数组1至10，线性分割1000份
#
#     print(data)
#     print(f(data))  # 将数组data带入表达式
#
#     mlab.quiver3d(u, v, w)  # 绘制矢量场
#     mlab.outline()  # 绘制边框


def test_flow():
    x, y, z = np.mgrid[-5:5:40j, -5:54:40j, 0:4:20j]  # x y z网格化，形成填充三维坐标数组
    u = y  # 矢量场x分量
    v = -x  # 矢量场y分量
    w = np.ones_like(z) * 0.05  # 数组用1来填充  #矢量场z分量
    mlab.quiver3d(u, v, w, mask_points=10)  # 绘制矢量场
    obj = mlab.flow(u, v, w)  # 在矢量场中放置可移动物体以检查流场
    return obj


test_flow()

# mlab.quiver3d(u, v, w)  # 绘制矢量场
# mlab.outline()  # 绘制边框


# test_flow()


def draw_speed_mayavi(stream_map, n_sacle=3, plt_density=5):
    # max(stream_map[:, 1])
    # max(stream_map[:, 2])

    max_x = int(max(stream_map[:, 1]) / n_sacle) + 1
    max_y = int(max(stream_map[:, 2]) / n_sacle) + 1

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
            x = stream_map[K_0, 1] / n_sacle
            y = (max(stream_map[:, 2]) - stream_map[K_0, 2]) / n_sacle
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





video_name = '../data/video_CS_20201031_h_0_to_h_13/video_CS_20201031_h_0_to_h_13_552_713_239_447_4.avi'
position_name = video_name[:-4] + '_position.npy'
position_np = np.load(position_name)
print(position_np.shape)


from egg_streamplot import get_speed

speed_np = get_speed(position_np, distance_threshold=10,frame_interval=10, save_interval=100)



stream_map = np.copy(speed_np)
print(stream_map.shape)

stream_map
n_sacle=3
plt_density=5

# max(stream_map[:, 1])
# max(stream_map[:, 2])

max_x = int(max(stream_map[:, 1]) / n_sacle) + 1
max_y = int(max(stream_map[:, 2]) / n_sacle) + 1

if abs(max_x - max_y) > 2:
    print('x and y have not benn modified. break.')
    # return None
# else:
max_x_y = max(max_x, max_y)
speed_x_streamplot = np.zeros([max_x_y, max_x_y, max_x_y])
speed_y_streamplot = np.zeros([max_x_y, max_x_y, max_x_y])
speed_count_streamplot = np.ones([max_x_y, max_x_y, max_x_y])

len_peroid = int(len(stream_map/max_x_y))+1
for K_0 in range(max_x_y):#len(stream_map)):
    # y = stream_map[K_0, 2] / n_sacle
    # y = (max(orentation_np[:, 2]) - orentation_np[K_0, 2]) / n_sacle

    x = stream_map[K_0, 1] / n_sacle
    y = (max(stream_map[:, 2]) - stream_map[K_0, 2]) / n_sacle
    # for K_1 in range(len_peroid):
    #     speed_x_streamplot[int(y), int(x), K_1] += stream_map[K_0, 3]
    #     speed_y_streamplot[int(y), int(x), K_1] -= stream_map[K_0, 4]
    #     speed_count_streamplot[int(y), int(x), K_0] += 1

    speed_x_streamplot[int(y), int(x), K_0] += stream_map[K_0, 3]
    speed_y_streamplot[int(y), int(x), K_0] -= stream_map[K_0, 4]
    # speed_y_streamplot[int(y), int(x)] += stream_map[K_0, 4]
    speed_count_streamplot[int(y), int(x), K_0] += 1
# for K_1 in range(speed_x_streamplot.shape[0]):
#     for K_2
speed_x_streamplot /= speed_count_streamplot
speed_y_streamplot /= speed_count_streamplot
print(speed_x_streamplot[:4, :4,:4])

print(speed_x_streamplot.shape)

# w = 3
Y, X, Z = np.mgrid[0:max_x_y, 0:max_x_y, 0:len(stream_map)]
# U = -1 - X ** 2 + Y
# V = 1 + X - Y ** 2
# speed = np.sqrt(U ** 2 + V ** 2)
# fig = plt.figure()
U = speed_x_streamplot
V = speed_y_streamplot
# W = np.zeros([max_x_y, max_x_y, len(stream_map)])
W = np.ones([max_x_y, max_x_y, max_x_y])
mlab.quiver3d(U, V, W, mask_points=10)  # 绘制矢量场
obj = mlab.flow(U, V, W)



plt.figure()
plt.streamplot(X, Y, Z, U, V, W, density=[plt_density, plt_density])
# plt.set_title('Speed')
plt.title('Speed')









