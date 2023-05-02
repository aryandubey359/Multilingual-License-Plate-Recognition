
# # def imagePorcessing(__init__):
# #Cropped Image of License Plate


# import os 
# import cv2 as cv
# from matplotlib.image import imread
# import numpy as np
# import PIL
# img= 'cropped.jpg'
# orgImage = cv.imread(img)
# x, y, w, h = 100, 100, 200, 200
# crop_img = orgImage[y:y+h, x:x+w]
# resize_img = cv.resize(orgImage, (orgImage.shape[1], orgImage.shape[0]))
# cv.imwrite('images/resize.jpg', resize_img)
# #Grayscale Conversion
# grayImage= cv.cvtColor(resize_img, cv.COLOR_BGR2GRAY)
# cv.imwrite('images/gray.jpg', grayImage)
# equalImage = cv.equalizeHist(grayImage)
# cv.imwrite('images/equal.jpg', equalImage)

# #Gaussian Blur of Grayscaled Image
# blurImage= cv.GaussianBlur(grayImage, (5,5),0)
# # cv.imshow('blur image', blurImage)
# cv.imwrite('images/blur.jpg', blurImage)
#     # cv.waitkey(0)


# # _, thresh = cv.threshold(blurImage, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
# # cv.imwrite('images/otsu.jpg', thresh)

# # rect_kern = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# # dilation = cv.dilate(thresh,rect_kern,iterations=1)
# # cv.imwrite('images/dilate.jpg',dilation)

# # contours, hierarchy = cv.findContours(dilation,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# # sort_contours = sorted(contours,key=lambda ctr:cv.boundingRect(ctr)[0])
# # cv.imwrite('images/con.jpg',sort_contours)