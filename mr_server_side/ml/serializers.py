from rest_framework import serializers

import json


class CompareImageSerializer(serializers.Serializer):

    unity_image_base64 = serializers.CharField()
    hololens_image_base64 = serializers.CharField()