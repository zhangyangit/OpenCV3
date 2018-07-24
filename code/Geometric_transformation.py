# -*- coding: utf-8 -*-
__author__ = 'Morgan'

'''
python opencv | 03--geometric transformation
'''
import numpy as np
import cv2
import sys
import math

# 仿射变换
def affine_transformation():
    image = cv2.imread("D:\\Entertainment\\Picture\\a1.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('image.jpg', image)
    h,w = image.shape[:2]
    # 缩放
    A1 = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
    d1 = cv2.warpAffine(image, A1, (w, h), borderValue=125)
    # 缩放+平移
    A2 = np.array([[0.5, 0, w/4], [0, 0.5, h/4]], np.float32)
    d2 = cv2.warpAffine(image, A2, (w, h), borderValue=125)
    # d2 上旋转
    A3 = cv2.getRotationMatrix2D((w/2.0, h/2.0), 30, 1)
    d3 = cv2.warpAffine(d2, A3, (w, h), borderValue=125)
    # 新特性旋转
    rImg = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('image', image)
    cv2.imshow('d1', d1)
    cv2.imshow('d2', d2)
    cv2.imshow('d3', d3)
    cv2.imshow('rImg', rImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 投影变换
def projection_transformation():
    image = cv2.imread("D:\\Entertainment\\Picture\\a2.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    src = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], np.float32)
    dst = np.array([[50, 50], [w/3, 50], [50, h-1], [w-1, h-1]], np.float32)
    # 计算投影变换矩阵
    p = cv2.getPerspectiveTransform(src, dst)
    # 进行投影变换
    r = cv2.warpPerspective(image, p, (w, h), borderValue=125)
    cv2.imshow('image', image)
    cv2.imshow('warpPerspective', r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 极坐标变换
def Polar_Coordinate( ):
    I = cv2.imread("D:\\Entertainment\\Picture\\a2.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = I.shape
    cx, cy = 508, 503
    cv2.circle(I, (int(cx), int(cy)), 10, (255.0, 0, 0), 3)
    r = (200, 550)
    # 距离的范围
    theta = (0, 360)
    rstep = 1.0
    thetastep = 360.0 / (180 * 8)
    minr, maxr = r
    # 角度范围
    mintheta, maxtheta = theta
    # h, w
    H = int((maxr - minr)/rstep) + 1
    W = int((maxtheta - mintheta)/thetastep) + 1
    O = 125 * np.ones((H, W), I.dtype)
    # 极坐标变换
    r = np.linspace(minr, maxr, H)
    r = np.tile(r, (W, 1))
    r = np.transpose(r)
    theta = np.linspace(mintheta, maxtheta, W)
    theta = np.tile(theta, (H, 1))
    x, y = cv2.polarToCart(r, theta, angleInDegrees=True)
    # 最近邻值
    for i in range(H):
        for j in range(W):
            px = int(round(x[i][j]) + cx)
            py = int(round(y[i][j]) + cy)
            if ((px >= 0 and px <= w-1)and(py >= 0 and px <= h-1)):
                O[i][j] = I[py][px]
    return O



if __name__ == "__main__":
    affine_transformation()
    projection_transformation()
