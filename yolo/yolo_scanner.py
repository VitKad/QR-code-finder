import cv2
import numpy as np
from glob import glob

# функция для отображения указаного числа значений после запятой
def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"

# загрузка алгоритма
net = cv2.dnn.readNet("yolov3_work_2000.weights", "yolov3_testing.cfg")
# имена для найденных объектов
classes = ["QR-Code"]

files = glob(r"D:/QR-scanner/*.jpg")
#список всех слоев используемых в сети
layers = net.getLayerNames()
#получить индекса трех выходных слоев
outputLayers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

#цвета для выделения и текста
colorBox = (255, 0, 0)
colorText = (0, 255, 255)
colorTextBox = (0, 0, 0)

coordFile = open("../coord.txt", "w+")
coordFile.write("Координаты распознанных штрихкодов:\n")
coordFile.write("\n")
coordFile.close()

countFiles = 0

for file in files:
    img = cv2.imread(file)
	# опредление ширины, высоты и количества каналов
    height, width, channels = img.shape
    # преобразование в бинарный объект. 1/255 - нормализация к 0 до 1. Ширина и высота на вход в нейросеть.
    blob = cv2.dnn.blobFromImage(img, 0.00392, (832, 832), (0, 0, 0), True, crop=False)

	# передача блоб изображения на вход сети
    net.setInput(blob)
    results = net.forward(outputLayers)

    allIdClass = []
    confidences = []
    rect = []
    conf = []


	#перебор каждого выхода слоя
    for result in results:
		#перебор каждого обнаруженого объекта
        for detection in result:
            scores = detection[5:]
			# идентификатор класа. 
            idClass = np.argmax(scores)
			# достоверность объекта
            confid = scores[idClass]
            # запомнить объекты с заданой достоверностью
            if confid > 0.5:
                # определение центра полученного кода. умножаем на шир и выс оригинала
                xCenter = int(detection[0] * width)
                yCenter = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
              
                # определние угла
                x = int(xCenter - w / 2)
                y = int(yCenter - h / 2)

                # запоминание объекта
                rect.append([x, y, w, h])
                confidences.append(float(confid))
                allIdClass.append(idClass)
    
    coordFile = open("../coord.txt", "a+")
	# фильтрация объект. убрать повторы
    filter = cv2.dnn.NMSBoxes(rect, confidences, 0.1, 0.1)
    countFiles = countFiles + 1
    coordFile.write(f"Изображение №{countFiles}\n")
    fontText = cv2.FONT_HERSHEY_PLAIN
    countCode = 0

    # отображение результатов на фотографии
    for i in range(len(rect)):
        if i in filter:
		    # извлекаются координаты объекта
            x, y, w, h = rect[i]
            # приведение к процентам
            conf = confidences[i] * 100
            countCode = countCode + 1;
            # преобразование координаты с угловой в центр
            fx=int(x + w / 2)
            fy=int(y + h / 2)

            coordFile.write(f"Код №{countCode}: ")
            coordFile.write(f"x={str(fx)}, y={str(fy)}\n")
            
			# имя объекта, к какому классу он принадлежит
            label = (f'{classes[allIdClass[i]]}')
            label2= (f'{toFixed(conf, 2)}%')
			# выделение объекта на изображение
            cv2.rectangle(img, (x, y), (x + w, y + h), colorBox, 4)
			
            # параметры для отображения текста типа объекта и достоверности
            (widthText, heightText) = cv2.getTextSize(label, fontText, fontScale=3, thickness=2)[0]
            (widthTextConf, heightTextConf) = cv2.getTextSize(label2, fontText, fontScale=3, thickness=2)[0]
            xOffsetText = x
            yOffsetText = y - 5
            rectCoords = ((xOffsetText, yOffsetText), (xOffsetText + widthText, yOffsetText - heightText - heightTextConf - 20))
            cv2.rectangle(img, rectCoords[0], rectCoords[1], colorTextBox, thickness=-1)
            cv2.putText(img, label, (x, y - 45), fontText, 3, colorText, 4)
            cv2.putText(img, label2, (x, y - 5), fontText, 3, colorText, 4)

    coordFile.write("----------------------------------\n")
    coordFile.close()
    cv2.imwrite(f'../result{countFiles}.jpg', img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()