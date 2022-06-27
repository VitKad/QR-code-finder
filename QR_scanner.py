from pyzbar import pyzbar
import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage
from skimage import exposure, io, measure, feature
from glob import glob


def find(image):
	codes = pyzbar.decode(image)
	for code in codes:
		image = select(code, image)
		print("Данные:\n", code.data.decode())
		print()
	return image


def select(code, image):
	length = len(code.polygon)
	for i in range(length):
		image = cv2.line(image, code.polygon[i], code.polygon[(i+1) % length], color=(255, 0, 0), thickness=5)
	return image

		
if __name__ == "__main__":
	count = 0	
	files = glob("images/*.jpg")
	for file in files:
		img = plt.imread(file)
		# преобразование в черно-белый цвет
		image = skimage.color.rgb2gray(img)
		plt.imshow(image, cmap='gray')
		plt.show()
		# коррекция гаммы
		gamma = exposure.adjust_gamma(image, 5)
		plt.imshow(gamma, cmap='gray')
		plt.show()
		# нормализация изображения
		thr = (gamma >= 0.3)
		plt.imshow(thr, cmap='gray')
		plt.show()
		# обнаружение границ
		edges = feature.canny(thr, sigma=10)
		plt.imshow(edges)
		plt.show()
		# поиск контуров
		contours = measure.find_contours(edges, 0, fully_connected='high', positive_orientation='high')
		
		coordFile = open("coordinates/coord.txt", "w+")
		coordFile.write("Координаты распознаных штрихкодов:\n")
		plt.imshow(edges)

		# отображение всех найденных контуров на изображение
		for contour in contours:
			plt.plot(contour[:,1], contour[:,0], linewidth=2, color = (0, 0, 1))
		
		io.imshow(img)
		plt.show()

		# отображение всех контуров после аппроксимации
		for contour in contours:
			coords = measure.approximate_polygon(contour, tolerance=20)
			plt.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
			coordFile.write(str(coords))
	
		coordFile.close()
	  
		# поиск точек экстремума
		positions = np.concatenate(contours, axis=0)
		min_pos_x = int(min(positions[:,1]))
		max_pos_x = int(max(positions[:,1]))
		min_pos_y = int(min(positions[:,0]))
		max_pos_y = int(max(positions[:,0]))

		# координаты крайних точек
		start = (min_pos_x, min_pos_y)
		end = (max_pos_x, max_pos_y)
		cv2.rectangle(img, start, end, (255, 0, 0), 5)
		io.imshow(img)

		# создание нового изображения, где остается только фрагмент со штрихкодами
		newImage=img[min_pos_y:max_pos_y, min_pos_x:max_pos_x]
		io.imshow(newImage)
		plt.show()
		# поиск штрихкодов
		img = find(newImage)
		count = count + 1
		cv2.imwrite(f'result/{count}.jpg', img)

cv2.waitKey(0)