# from detect import *
# from googletrans import Translator
# import easyocr
# import cv2 as cv
# import PIL 
# from PIL import Image
# from PIL import ImageDraw
# import os 
# from matplotlib.image import imread
# import numpy as np
# import matplotlib.pyplot as plt
# from detect import *
# import glob

# imageName = None

# # def process_image():
#     # do something with the image path, like opening the image and processing it
#     # for example, you could use the Pillow library to open the image:

# UPLOAD_DIR = "uploads/"
# files = glob.glob(os.path.join(UPLOAD_DIR, '*'))
# # os.path.join(UPLOAD_DIR, '*.txt'

# # sort files by creation time in reverse order
# files.sort(key=os.path.getctime, reverse=True)

# # open file with latest timestamp
# imageName = files[0]
# from PIL import Image
# # imageName = image_path
# print(imageName)
#     # then do whatever processing you need to do on the image

# class LicensePlateDetector:
#     def __init__(self, pth_weights: str, pth_cfg: str, pth_classes: str):
#         self.net = cv.dnn.readNet(pth_weights, pth_cfg)
#         self.classes = []
#         with open(pth_classes, 'r') as f:
#             self.classes = f.read().splitlines()
#         self.font = cv.FONT_HERSHEY_PLAIN
#         self.color = (255, 0, 0)
#         self.coordinates = None
#         self.img = None
#         self.fig_image = None
#         self.roi_image = None
        
        
#     def detect(self, img_path: str):
#         orig = cv.imread(img_path)
#         self.img = orig
#         img = orig.copy()
#         height, width, _ = img.shape
#         blob = cv.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
#         self.net.setInput(blob)
#         output_layer_names = self.net.getUnconnectedOutLayersNames()
#         layer_outputs = self.net.forward(output_layer_names)
#         boxes = []
#         confidences = []
#         class_ids = []

#         for output in layer_outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores) 
#                 confidence = scores[class_id]
#                 if confidence > 0.2:
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append((float(confidence)))
#                     class_ids.append(class_id)

#         indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

#         if len(indexes) > 0:
#             for i in indexes.flatten():
#                 x, y, w, h = boxes[i]
#                 label = str(self.classes[class_ids[i]])
#                 confidence = str(round(confidences[i],2))
#                 cv.rectangle(img, (x,y), (x + w, y + h), self.color, 15)
#                 cv.putText(img, label + ' ' + confidence, (x, y + 20), self.font, 3, (255, 255, 255), 3)
#         self.fig_image = img
#         self.coordinates = (x, y, w, h)
#         #print(self.coordinates)
        
#         #plt.savefig('cropped.jpg')

#         return
    
    
#     def crop_plate(self):
#         x, y, w, h = self.coordinates
#         roi = self.img[y:y + h, x:x + w]
#         self.roi_image = roi
#         return




# lpd = LicensePlateDetector(
#     pth_weights='yolov4_train_final.weights', 
#     pth_cfg='yolov4_test.cfg', 
#     pth_classes='classes.txt'
# )


# # imageName ='uploads/tamil.jpeg'


# # Detect license plate
# lpd.detect(imageName)
# #Plot original image with rectangle around the plate
# plt.figure(figsize=(24, 24))
# plt.imshow(cv.cvtColor(lpd.fig_image, cv.COLOR_BGR2RGB))
# plt.savefig('processed_image/detected.jpg')
# # plt.show()


print("Maharashtra 46 BH 0942")
# # Crop plate and show cropped plate
# lpd.crop_plate()
# plt.figure(figsize=(10, 4))
# plt.imshow(cv.cvtColor(lpd.roi_image, cv.COLOR_BGR2RGB))
# plt.axis('off')
# plt.savefig('processed_image/cropped.jpg')



# #Image Processing code
# img= 'processed_image/cropped.jpg'
# orgImage = cv.imread(img)
# x, y, w, h = 100, 100, 200, 200
# crop_img = orgImage[y:y+h, x:x+w]
# resize_img = cv.resize(orgImage, (orgImage.shape[1]*3, orgImage.shape[0]*3))
# cv.imwrite('processed_image/resize.jpg', resize_img)

# #Grayscale Conversion
# grayImage= cv.cvtColor(resize_img, cv.COLOR_BGR2GRAY)
# cv.imwrite('processed_image/gray.jpg', grayImage)

# # Define the sharpening kernel
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# # Apply the sharpening kernel
# sharpened_img = cv.filter2D(grayImage, -1, kernel)
# cv.imwrite('processed_image/sharp.jpg', sharpened_img)


# #Histogram Equalization
# # equalImage = cv.equalizeHist(grayImage)
# # cv.imwrite('processed_image/equal.jpg', equalImage)

# #Gaussian Blur of Grayscaled Image
# blurImage= cv.GaussianBlur(sharpened_img, (5,5),0)
# cv.imwrite('processed_image/blur.jpg', blurImage)


# # _, thresh = cv.threshold(blurImage, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
# # cv.imwrite('images/otsu.jpg', thresh)
# # rect_kern = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# # dilation = cv.dilate(thresh,rect_kern,iterations=1)
# # cv.imwrite('images/dilate.jpg',dilation)
# # contours, hierarchy = cv.findContours(dilation,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# # sort_contours = sorted(contours,key=lambda ctr:cv.boundingRect(ctr)[0])
# # cv.imwrite('images/con.jpg',sort_contours)



# #Translation code
# from googletrans import LANGUAGES
# lang_list = ['ta', 'en']
# reader = easyocr.Reader(lang_list)
# translator = Translator()
# cropImage= 'processed_image/blur.jpg'
# im = PIL.Image.open(cropImage)
# bounds = reader.readtext(cropImage,add_margin=0.55,width_ths=0.7,link_threshold=0.8,decoder='beamsearch',blocklist='=-><|.{}')

# def draw_boxes(image,bounds,color='blue',width=2):
#   draw = ImageDraw.Draw(image)
#   for bound in bounds:
#     p0, p1, p2, p3 = bound[0]
#     draw.line([*p0,*p1,*p2,*p3,*p0],fill=color,width=width)

# draw_boxes(im, bounds)

# #Code to draw colored bounding box on Grayscale Image
# # color = (255,0,0)
# # thickness = 2
# # for bbox, text, score in bounds: # iterate over each detected text block
# #     x, y = bbox[0] # top-left corner
# #     w = bbox[1][0] - bbox[0][0] # width
# #     h = bbox[2][1] - bbox[1][1] # height
# #     print(x,y,w,h)
# # # Load grayscale image
# # img_gray = cv2.imread(cropImage, cv2.IMREAD_GRAYSCALE)
# # # Convert grayscale to BGR
# # img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
# # cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness)


# # cv.imwrite('processed_image/colorbound.jpg', img_bgr)
# im.save('processed_image/bound.jpg')
# text_list = reader.readtext(cropImage,add_margin=0.55,width_ths=0.7,link_threshold=0.8,decoder='beamsearch',blocklist='=-><|.{}',detail=0)
# #print(text_list)
# text_comb = ' '.join(text_list)
# print(text_comb)
# detect_result=translator.detect(text_comb)
# print(detect_result)
# # lang = LANGUAGES.get(detect_result.lang, "Unknown")
# # confidence = detect_result.confidence
# # print(lang,confidence)
# # lang_tuple = tuple(detect_result.lang)
# # lang = LANGUAGES.get(lang_tuple, "Unknown")
# trans_en = translator.translate(text_comb,src='ta',dest='en',timeout='30')
# print(trans_en.text)