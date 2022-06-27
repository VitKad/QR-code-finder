from pyzbar import pyzbar
import cv2
from matplotlib import pyplot as plt
from glob import glob

# функция обнаружения штрихкодов
def find(image):
    # расшифровка штрихкодов с изображения
    codes = pyzbar.decode(image)
    numCode = 0
    for code in codes:
        numCode = numCode + 1
        # запись номера штрихкода в файл
        coordFile.write(f"Координаты штрихкода №{numCode}\n")
        # выделение	штрихкода на изображении	
        image = select(code, image)
        print("Данные:\n", code.data.decode())
        print()
    numCode = 0
    return image


# функция выделения штрихкода на изображении
def select(code, image):
    length = len(code.polygon)
    for i in range(length):
        # отрисовка линий между углами штрихкода
        image = cv2.line(image, code.polygon[i], code.polygon[(i+1) % length], color=(255, 0, 0), thickness=5)
        print(code.polygon[i].x)
        # запись координат в файл
        coordFile.write(f"x={str(code.polygon[i].x)}, y={str(code.polygon[i].y)}\n")
      
    return image


if __name__ == "__main__":

    count = 0
    coordFile = open("coordinates/coord.txt", "w+")
    coordFile.write("Координаты распознанных штрихкодов:\n")
    coordFile.close()
    files = glob("images/*.jpg")
    for file in files:
        count = count + 1 
        coordFile = open("coordinates/coord.txt", "a+")
        coordFile.write(f"\nИзображение №{count}:\n")
        coordFile.write("----------------------------------\n")
        # преобразование изображения в массив значений
        img = cv2.imread(file)
        img = find(img)
        
        # отображение результата на экран
        plt.imshow(img)
        plt.show()
        coordFile.write("----------------------------------\n")
        coordFile.close()   
        cv2.imwrite(f'result/{count}.jpg', img)
        cv2.waitKey(0)
