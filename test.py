# from googletrans import Translator
# import easyocr
# import cv2 as cv
# reader = easyocr.Reader(['mr','hi','en'])
# translator = Translator()
# import PIL 
# from PIL import Image
# from PIL import ImageDraw
# cropImage= 'images/gray.jpg'
# im = PIL.Image.open(cropImage)
# bounds = reader.readtext(cropImage,add_margin=0.55,width_ths=0.7,link_threshold=0.8,decoder='beamsearch',blocklist='=-')

# def draw_boxes(image,bounds,color='blue',width=2):
#   draw = ImageDraw.Draw(image)
#   for bound in bounds:
#     p0, p1, p2, p3 = bound[0]
#     draw.line([*p0,*p1,*p2,*p3,*p0],fill=color,width=width)

# draw_boxes(im, bounds)
# im.save('example_with_bounds.png')
# text_list = reader.readtext(cropImage,add_margin=0.55,width_ths=0.7,link_threshold=0.8,decoder='beamsearch',blocklist='=-',detail=0)
# print(text_list)
# text_comb = ' '.join(text_list)
# print(translator.detect(text_comb))
# trans_en = translator.translate(text_comb,src='hi',dest='en')
# print(trans_en.text)