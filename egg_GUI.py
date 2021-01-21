# -*- coding: utf-8 -*-

"""
PyQt5 tutorial 

In this example, we determine the event sender
object.

author: py40.com
last edited: 2017年3月
"""

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit,
                             QInputDialog, QApplication)

def draw_gray_hist(self, ax, img_fg):
    ax.cla()
    hist_b = np.array(range(256))
    hist_f = np.zeros((256,))
    img_f = img_fg.flatten().astype(int)
    for i in hist_b:
        hist_f[i] += img_f[img_f == i].size
    ax.bar(hist_b[1:-10], hist_f[1:-10])
    ax.set_xlim(0, 255)

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):
        # from PyQt5.QtCore import *

        # dialog = QFileDialog.getExistingDirectory(
        #     self,
        #     directory="/",
        #     caption="Select Directory",
        #     options=QFileDialog.DontUseNativeDialog,
        # )
        # self.folder = str(dialog)
        # print(self.folder)

        self.select_file_fold = QPushButton('select file fold', self)
        self.select_file_fold.move(20, 20)
        self.select_file_fold.clicked.connect(self.show_dialog)


        self.open_video = QPushButton('open video', self)
        self.open_video.move(120, 20)
        self.open_video.clicked.connect(self.show_dialog_1)

        self.le = QLineEdit(self)
        self.le.move(30, 122)

        self.setGeometry(300, 300, 290 * 3, 150 * 3)
        self.setWindowTitle('Input dialog')
        # self.show()

        self.show()

    def show_dialog(self):

        text, ok = QInputDialog.getText(self, 'Input Dialog',
                                        'Enter your name:')

        if ok:
            self.le.setText(str(text))

    def show_dialog_1(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        # print(fname)
        if fname[0]:
            print(fname[0])
            img = cv2.imread(fname[0])
            # plt.imshow(img)
            print(img)

        # if fname[0]:
        #     f = open(fname[0], 'r')
        #
        #     with f:
        #         data = f.read()
        #         self.textEdit.setText(data)



if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())