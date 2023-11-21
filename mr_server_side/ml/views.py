from django.shortcuts import render

from mr_server_side.settings import STATICFILES_DIRS, STATIC_ROOT

from ml.models import calculate_simirality_of_two_image ,image_to_base64, save_image_from_base64, show_image, write_image
from ml.serializers import CompareImageSerializer, ImageClassificationSerializer

from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import numpy as np

@api_view(["POST"])
def compare_image(request):

    serializer = CompareImageSerializer(data=request.data)
    if not serializer.is_valid():
        return Response({"detail": "Invalid request"}, status=status.HTTP_400_BAD_REQUEST)
    unity_image_base64 = serializer.validated_data.get("unity_image_base64")
    hololens_image_base64 = serializer.validated_data.get("hololens_image_base64")

    err = save_image_from_base64(hololens_image_base64, "hololens_image")
    err2 = save_image_from_base64(unity_image_base64, "unity_image")
    if err:
        return Response({"error": err, "error2": err2}, status=status.HTTP_400_BAD_REQUEST)
    
    rf = Roboflow(api_key="FDHkG7bpmLQsiZlma4nV")
    image_path= r"C:\Users\James\Desktop\project2\project2\static\images\hololens_image.png"
    project = rf.workspace().project("thesis-mep-testing")
    model = project.version(3).model

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
    cv2.fillPoly(overlay, [polygon_points], fill_color)

    # Transparent
    alpha = 0.5  # Transparency factor.
    image_copy = image.copy()  # Make a copy to draw on
    image_copy = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    output_path, err = write_image(image_copy, "overlay")
    if err:
        return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)
    overlay_base64, err = image_to_base64(output_path)

    height, width, channels = image.shape
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill the polygon on the blank image
    white_fill_color = (255,255,255)
    cv2.fillPoly(blank_image, [polygon_points], white_fill_color)

    # Compare 2 image to get a score.
    unity_image_path = r"C:\Users\James\Desktop\project2\project2\static\images\unity_image.png"
    unity_image = cv2.imread(unity_image_path)
    similarity_percentage = calculate_simirality_of_two_image(blank_image, unity_image)

    # Display the image with the filled polygon
    output_path, err = write_image(blank_image, "bw")
    if err:
        return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)
    blank_base64, err = image_to_base64(output_path)

    response = {"overlay_base64":overlay_base64,
                "blank_base64":blank_base64,
                "similarity_percentage":similarity_percentage,
                }

    return Response(response, status=status.HTTP_200_OK)

@api_view(["POST"])
def object_classification(request):
    serializer = ImageClassificationSerializer(data=request.data)
    if not serializer.is_valid():
        return Response({"detail": "Invalid request"}, status=status.HTTP_400_BAD_REQUEST)
    hololens_image_base64 = serializer.validated_data.get("hololens_image_base64")

    err = save_image_from_base64(hololens_image_base64, "hololens_image_1")
    if err:
        return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)
    
    rf = Roboflow(api_key="FDHkG7bpmLQsiZlma4nV")
    image_path= r"C:\Users\James\Desktop\project2\project2\static\images\hololens_image_1.png"
    project = rf.workspace().project("thesis-mep-testing")
    model = project.version(3).model

    # infer on an image hosted elsewhere
    predictions = (model.predict(image_path).json())["predictions"][0]
    confidence = predictions["confidence"]
    object_class_name = predictions["class"]
    points = predictions["points"]

    # Load image
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
    cv2.fillPoly(overlay, [polygon_points], fill_color)

    # Transparent
    alpha = 0.5  # Transparency factor.
    image_copy = image.copy()  # Make a copy to draw on
    image_copy = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    output_path, err = write_image(image_copy, "overlay")
    if err:
        return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)
    overlay_base64, err = image_to_base64(output_path)

    response = {"overlay_base64":overlay_base64,
                "confidence":confidence,
                "object_class_name":object_class_name,
                }

    return Response(response, status=status.HTTP_200_OK)