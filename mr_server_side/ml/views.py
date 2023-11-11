from django.shortcuts import render

from mr_server_side.settings import STATICFILES_DIRS, STATIC_ROOT

from ml.models import show_image, write_image

from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import numpy as np

@api_view(["GET"])
def compare_image(request):
    rf = Roboflow(api_key="FDHkG7bpmLQsiZlma4nV")
    image_path= r"C:\Users\James\Desktop\project2\project2\static\images\example.jpg"
    project = rf.workspace().project("thesis-mep-testing")
    model = project.version(3).model

    # # infer on a local image
    # print(model.predict("your_image.jpg").json())

    # infer on an image hosted elsewhere
    predictions = model.predict(image_path).json()
    predictions = predictions["predictions"]
    points = predictions[0]["points"]

    # Load an image
    image = cv2.imread(image_path)

    interest_point_list = []
    for point in points:
        x = point['x']
        y = point['y']
        interest_point_list.append([x,y])

    # Draw the polygon
    fill_color = (0, 255, 0)
    polygon_points = np.array(interest_point_list, dtype=np.int32)
    polygon_points = polygon_points.reshape((-1, 1, 2))
    overlay = image.copy()
    overlay = cv2.polylines(overlay, [polygon_points], isClosed=True, color=fill_color, thickness=10)
    
    # Fill color
    fill_color = (0, 255, 0)
    cv2.fillPoly(overlay, [polygon_points], fill_color)

    # Transparent
    alpha = 0.5  # Transparency factor.
    image_copy = image.copy()  # Make a copy to draw on
    image_copy = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    err = write_image(image_copy, "overlay")
    if err:
        return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)

    height, width, channels = image.shape
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill the polygon on the blank image
    white_fill_color = (255,255,255)
    cv2.fillPoly(blank_image, [polygon_points], white_fill_color)

    # Display the image with the filled polygon
    err = write_image(blank_image, "bw")
    if err:
        return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)

    return Response({"success": True}, status=status.HTTP_200_OK)
