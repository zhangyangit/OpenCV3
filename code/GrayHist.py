# 导入相关模块
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 计算灰度直方图
def calcGrayHist(image):
    # 存储灰度直方图的高，宽
    rows, cols = image.shape[:2]
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist

# 绘制直方图
def drawGrayHist(grayHist):
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=2, c='black')
    # 设置坐标轴范围
    y_maxvalue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxvalue])
    # 设置坐标标签
    plt.xlabel('gray Level')
    plt.ylabel('number of pixels')
    plt.show()

image = cv2.imread("../image/a1.jpg", cv2.IMREAD_ANYCOLOR)
grayHist = calcGrayHist(image)
drawGrayHist(grayHist)