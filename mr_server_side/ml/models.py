from django.db import models

import base64
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
        return output_path, None
    except Exception as e:
        return None, str(e)

def image_to_base64(file_path):
    try:
        with open(file_path, "rb") as image_file:
            # Encode the image file to base64
            base64_encoded = base64.b64encode(image_file.read())
            
            # Decode the bytes to a UTF-8 string
            base64_string = base64_encoded.decode("utf-8")
            
            return base64_string, None
    except Exception as e:
        return str(e), None