import numpy as np
from matplotlib import pyplot as plt

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3))


# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)

# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

def relu(x):
    if x < 0:
        return 255
    if x >= 0:
        return x

filtr_horizontal = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
filtr_vertical = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
filtr_45_deg = np.array([[0,-1,-2], [1,0,-1], [2,1,0]])

def filtr(img, filtr, is_relu = False):
    new_img = np.zeros((128, 128, 3))
    for k in range(3):
        for i in range(0, 126):
            for j in range(0, 126):
                new_img[i, j, k] = np.sum(img[i:i+3, j:j+3, k] * filtr)
                if is_relu:
                    new_img[i, j, k] = relu(new_img[i, j, k])
    return new_img

# konwersja macierzy na obrazek i wyświetlenie
plt.imshow(filtr(data, filtr_vertical))
plt.savefig("vertical.png")
plt.imshow(filtr(data, filtr_vertical, is_relu=True))
plt.savefig("vertical_relu.png")
plt.imshow(filtr(data, filtr_horizontal))
plt.savefig("horizontal.png")
plt.imshow(filtr(data, filtr_horizontal, is_relu=True))
plt.savefig("horizontal_relu.png")
plt.imshow(filtr(data, filtr_45_deg))
plt.savefig("45_deg.png")
plt.imshow(filtr(data, filtr_45_deg, is_relu=True))
plt.savefig("45_deg_relu.png")
plt.show()
