from pyzbar import pyzbar
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from matplotlib import pyplot as plt
import skimage
from skimage import util, exposure, io, measure, feature
from scipy import ndimage as ndi
import numpy as np
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

count = 0	
		
if __name__ == "__main__":
	files = glob("images/1/6.jpg")
	for file in files:
	
		img = plt.imread(file)
		image = skimage.color.rgb2gray(img)
		plt.imshow(image, cmap='gray')
		plt.show()
		gamma = exposure.adjust_gamma(image, 5)
		plt.imshow(gamma, cmap='gray')
		plt.show()
		thr = (gamma >= 0.3)
		plt.imshow(thr, cmap='gray')
		plt.show()
		edges = feature.canny(thr, sigma=10)
		plt.imshow(edges)
		plt.show()
		contours = measure.find_contours(edges, 0, fully_connected='high', positive_orientation='high')
		
		
		my_file = open("results1/coord.txt", "w+")
		my_file.write("Координаты распознаных штрихкодов:\n")
		#fig, (ax2) = plt.subplots(ncols=1, figsize=(9, 4))
		plt.imshow(edges)

		
		for contour in contours:
			#print(contour)
			plt.plot(contour[:,1], contour[:,0], linewidth=2, color = (0, 0, 1))

		io.imshow(img)
		plt.show()
		
		for contour in contours:
			
			coords = measure.approximate_polygon(contour, tolerance=20)
			plt.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
			print(f"Результат усреднения: из {len(contour)} заданных точек получилось {len(coords)}")
			my_file.write(str(coords))
	
		my_file.close()
		positions = np.concatenate(contours, axis=0)
		min_pos_x = int(min(positions[:,1]))
		max_pos_x = int(max(positions[:,1]))
		min_pos_y = int(min(positions[:,0]))
		max_pos_y = int(max(positions[:,0]))

		start = (min_pos_x, min_pos_y)
		end = (max_pos_x, max_pos_y)
		cv2.rectangle(img, start, end, (255, 0, 0), 5)
		io.imshow(img)
		plt.show()
		new_img=img[min_pos_y:max_pos_y, min_pos_x:max_pos_x]
		io.imshow(new_img)
		plt.show()
		img = find(new_img)
		count = count + 1
		cv2.imwrite(f'results1/first{count}.jpg', img)

cv2.waitKey(0)