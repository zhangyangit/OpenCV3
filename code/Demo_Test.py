# _*_ coding: utf-8 _*_

__author__ = 'Morgan'

'''
python spider -- OpenCV | | Demo2
'''
import cv2
import sys

# main()
if __name__ == "__main__":

#    if len(sys.argv[1]) > 1:
#        # input the image
#        image = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
#    else:
#        print("Usge: python Demo_Test.py imageFile")
    image = cv2.imread("../image/a1.jpg", cv2.IMREAD_ANYCOLOR)
    # display the picture
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()