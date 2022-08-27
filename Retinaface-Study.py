# coding:utf-8
# time: 2022/08/27

from retinaface import RetinaFace
from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def analyzeImg(image):
    # 检查一张照片上人的情绪、年龄、性别、人种
    obj = image
    result = DeepFace.analyze(obj, ['emotion', 'age', 'gender', 'race'])
    print(result)
    return result


def detectFace(image, analyzeFace):
    obj = image
    resp = RetinaFace.detect_faces(obj)
    print(resp)
    im = Image.open(obj)
    draw = ImageDraw.Draw(im)
    count = 0
    for i in resp:
        # 找到每个脸的坐标
        faceArea = resp[i]['facial_area']
        x, y, x2, y2 = faceArea[0], faceArea[1], faceArea[2], faceArea[3]
        # 在人脸上标注框框
        draw.rectangle((x, y, x2, y2), outline='red', width=3)
        # 每发现一个人脸，count加一
        count += 1

    for o in resp:
        # 找到每个脸的坐标
        faceArea = resp[o]['facial_area']
        x, y, x2, y2 = faceArea[0], faceArea[1], faceArea[2], faceArea[3]

        # 当开启人脸分析时
        if analyzeFace:
            try:
                # 把人脸裁剪出来
                face = im.crop((x - (x2 - x) * 0.1, y - (y2 - y) * 0.1, x2 + (x2 - x) * 0.1, y2 + (y2 - y) * 0.1))
                face.save("facetemp.jpg")
                # face.show()
                # 再用deepface来分析
                result = analyzeImg("facetemp.jpg")
                gender = result['gender']
                age = result['age']
                emotion = result['dominant_emotion']
                race = result['dominant_race']
                text = """Age:%s
Gender:%s
Emotion:%s
Race:%s""" % (age, gender, emotion, race)
                # 给人脸写上标注
                draw.text((x, y), text=text, fill='green', font=font)
            except:
                print("Error")
                break

    # 把总人脸数显示在图上
    draw.text((1, 1), str(count), fill="red", font=ImageFont.truetype(font="alihei.ttf", size=80))
    # 展示照片
    im.show()


# 设置字体
try:
    font = ImageFont.truetype(font="alihei.ttf", size=30)
except IOError:
    print("font Error")

# 识别人脸
# analyzeImg('facetemp.jpg')
obj = 'lastshoot.jpeg'
detectFace(obj, analyzeFace=True)

