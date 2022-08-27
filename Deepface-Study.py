# coding:utf-8
# time: 2022/08/26

import tensorflow
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'retinaface',
  'mediapipe'
]


# 检查一张照片上人的情绪、年龄、性别、人种
obj = "img1.jpg"
resp = DeepFace.detectFace(obj)
print(resp)
plt.imshow(resp)
plt.show()

result = DeepFace.analyze(obj, ['emotion','age','gender','race'])
print(result)
  # 并将人像框标注在照片上
im = Image.open(obj)
x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
draw = ImageDraw.Draw(im)
try:
  font = ImageFont.truetype(font="alihei.ttf", size=50)
except IOError:
  print("font Error")
  font = ImageFont.load_default()
print(type(x))
draw.rectangle((x+w, y, x+w+300, y+h),
               fill="white",
               outline="red",
               width=3)
gender = result['gender']
age = result['age']
emotion = result['dominant_emotion']
race = result['dominant_race']
text = """Age:
%s
Gender:
%s
Emotion:
%s
Race:
%s
""" % (age, gender, emotion, race)
draw.text((x+w+20, y+20), text=text, fill='black', font=font)
draw.rectangle((x, y, x+w, y+h), outline = 'red', width = 3 )
im.show()

# 检查视频流上的人
# obj = DeepFace.stream(db_path="DataBase", model_name='VGG-Face',detector_backend="opencv",)

# 检查图像中的人脸  没搞懂
#face = DeepFace.detectFace(img_path = "img1.jpeg",
#                           target_size = (224, 224),
#                          detector_backend = backends[4])
