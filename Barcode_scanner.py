from pyzbar import pyzbar
import numpy as np
import cv2
from matplotlib import pyplot as plt


def decode(image):
    # decodes all barcodes from an image
    decoded_objects = pyzbar.decode(image)
    numCode = 0
    for obj in decoded_objects:
        # draw the barcode
        #print(f"Обнаружен штрих-код:\n{obj}")
        numCode = numCode + 1
        #my_file = open("D:/QR-scanner/QR-scanner/coordinate.txt", "a+")
        my_file.write(f"Координаты штрихкода №{numCode}\n")
        #my_file.close()		
        image = draw_barcode(obj, image)

        # print barcode type & data
        print("Тип кода:", obj.type)
        print("Данные:\n", obj.data.decode())
        print()
    numCode = 0
    return image


def draw_barcode(decoded, image):
    n_points = len(decoded.polygon)
    for i in range(n_points):
        image = cv2.line(image, decoded.polygon[i], decoded.polygon[(i+1) % n_points], color=(255, 0, 0), thickness=5)
        print(decoded.polygon[i].x)
        
        #my_file = open("D:/QR-scanner/QR-scanner/coordinate.txt", "a+")
        my_file.write(f"x={str(decoded.polygon[i].x)}, y={str(decoded.polygon[i].y)}\n")
        #my_file.close()
		
        #cv2.show()
    # раскомментируйте выше и закомментируйте ниже, если хотите нарисовать многоугольник, а не прямоугольник
    #image = cv2.rectangle(image, (decoded.rect.left, decoded.rect.top), 
    #                        (decoded.rect.left + decoded.rect.width, decoded.rect.top + decoded.rect.height),
    #                        color=(0, 255, 0),
    #                        thickness=5)
    return image

	
def qr_reader(img):
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    if data:
        print(data)
    else:
        print('Ничего не нашлось!')

count = 0

if __name__ == "__main__":
    from glob import glob
    my_file = open("results2/coord.txt", "w+")
    my_file.write("Координаты распознаных штрихкодов:\n")
    my_file.close()
    barcodes = glob("images/1/*.jpg")
    for barcode_file in barcodes:
        count = count + 1 
        my_file = open("results2/coord.txt", "a+")
        my_file.write(f"\nИзображение №{count}:\n")
        my_file.write("----------------------------------\n")
        img = cv2.imread(barcode_file)
        # декодировать обнаруженные штрих-коды и получить изображение
        # нарисованный
        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #im = cv2.filter2D(img, -1, kernel)
		
        img = decode(img)
        #width = 100
        #height = 100
        #dsize=(width, height)
        #outpad = cv2.resize(img,dsize)
        # показать изображение
        plt.imshow(img)
        plt.show()
        my_file.write("----------------------------------\n")
        my_file.close()   
        cv2.imwrite(f'results2/first{count}.jpg', img)
        cv2.waitKey(0)
