import cv2
import numpy as np
from glob import glob

def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"

# Загрузить алгоритм
net = cv2.dnn.readNet("yolov3_work_2000.weights", "yolov3_testing.cfg")
#net = cv2.dnn.readNet("work_yolo.weights", "yolov3_testing.cfg")
# Имена для класса
classes = ["QR-Code"]

# Images path
files = glob(r"../images/1/*.jpg")
#files = glob(r"D:/QR-scanner/QR-scanner/qr/4/*.jpg")
#список всех слоев используемых в сети
layers = net.getLayerNames()

#получить индекса трех выходных слоев
outputLayers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

#цвета для выделения и текста
colorBox = (255, 0, 0)
colorText = (0, 255, 255)
colorTextBox = (0, 0, 0)


coordFile = open("../results3/coord.txt", "w+")
coordFile.write("Координаты распознаных штрихкодов:\n")
coordFile.write("\n")
coordFile.close()

countFiles = 0
	
# loop through all the images
for file in files:
    # Loading image
    img = cv2.imread(file)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
	
	#опредление ширины, высоты и количества каналов
    height, width, channels = img.shape

    # преобразование в блоб формат. 1/255 - преобразование от 0 до 1. квадрат для yolo.  swap цветов так как CV используе BGR
	#Blob - это объект массива 4D numpy (изображения, каналы, ширина, высота). 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (832, 832), (0, 0, 0), True, crop=False)

	#подаем блоб изображение на вход сети. сетевой отклик. Массив вывода распознаных объектов
    net.setInput(blob)
    results = net.forward(outputLayers)

    # Showing informations on the screen
    allIdClass = []
    confidences = []
    rect = []
    conf = []
	#перебор каждого выхода слоя
    for result in results:
		#перебор каждого обнаруженого объекта
        for detection in result:
            scores = detection[5:]
			#идентификатор класа. индекс масимального значения
            idClass = np.argmax(scores)
			#достоверность
            confid = scores[idClass]
            if confid > 0.5:
                # определяем центр полученного кода. умножаем на шир и выс оригинала
                xCenter = int(detection[0] * width)
                yCenter = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                #print(f"x={xCenter}, y={yCenter}\n")
                #print(f"h={h}, w={w}")
                
                # определние левого угла
                x = int(xCenter - w / 2)
                y = int(yCenter - h / 2)
                # обновить наш список координат ограничивающего прямоугольника, достоверности,
                # и идентификаторы класса
                rect.append([x, y, w, h])
                confidences.append(float(confid))
                allIdClass.append(idClass)
    
    coordFile = open("../results3/coord.txt", "a+")
	#убрать повторы
    filter = cv2.dnn.NMSBoxes(rect, confidences, 0.1, 0.1)
    print(filter)
    countFiles = countFiles + 1
    coordFile.write(f"Изображение №{countFiles}\n")
    fontText = cv2.FONT_HERSHEY_PLAIN
    countCode = 0
    for i in range(len(rect)):
        if i in filter:
		    #извлекаются координаты прямоугольника
            x, y, w, h = rect[i]
            conf = confidences[i] * 100
            countCode = countCode + 1;
            fx=int(x + w / 2)
            fy=int(y + h / 2)
            #print(f"x={fx}, y={fy}\n")
            coordFile.write(f"Код №{countCode}: ")
            coordFile.write(f"x={str(fx)}, y={str(fy)}\n")
            
			#имя объекта, в будущем баркоды
            label = (f'{classes[allIdClass[i]]}')
            label2= (f'{toFixed(conf, 2)}%')
			#отрисовка объекта
            cv2.rectangle(img, (x, y), (x + w, y + h), colorBox, 4)
			
            (widthText, heightText) = cv2.getTextSize(label, fontText, fontScale=3, thickness=2)[0]
            (widthTextConf, heightTextConf) = cv2.getTextSize(label2, fontText, fontScale=3, thickness=2)[0]
            xOffsetText = x
            yOffsetText = y - 5
            rectCoords = ((xOffsetText, yOffsetText), (xOffsetText + widthText, yOffsetText - heightText - heightTextConf - 20))
            overlay = img.copy()
            cv2.rectangle(img, rectCoords[0], rectCoords[1], colorTextBox, thickness=-1)
            # добавить непрозрачность (прозрачность поля)
            output = img.copy()
            # теперь поместите текст (метка: доверие%)
            cv2.putText(img, label, (x, y - 45), fontText, 3, colorText, 4)
            cv2.putText(img, label2, (x, y - 5), fontText, 3, colorText, 4)
			
            #cv2.addWeighted(img, 0.3, output, 0.7, 0, output)
            #cv2.putText(img, label, (x, y - 10), fontText, 3, colorText, 4)
    coordFile.write("----------------------------------\n")
    
    #cv2.imshow("Image", img)
    coordFile.close()
    #plt.imshow(img)
    #plt.show()
    cv2.imwrite(f'../results3/first{countFiles}.jpg', img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()