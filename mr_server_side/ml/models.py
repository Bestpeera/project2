from django.db import models

import cv2
import matplotlib.pyplot as plt

# Create your models here.
def show_image(image):
    try:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    except Exception as e:
        return str(e)

def write_image(image, file_name):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_path = r"C:\Users\James\Desktop\project2\project2\static\images\{}.png".format(file_name)
        cv2.imwrite(output_path, rgb_image)
    except Exception as e:
        return str(e)