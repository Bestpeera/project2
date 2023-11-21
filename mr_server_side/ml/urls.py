from django.urls import path

# from django_rest_passwordreset.views import reset_password_request_token
from ml import views

app_name = "ml"
urlpatterns = [
    path("compare_image", views.compare_image, name="compare_image"),
    path("classification", views.object_classification, name="classification")
]