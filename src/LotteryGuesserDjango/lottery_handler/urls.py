from django.urls import path

from . import views

urlpatterns = [
    path('lottery_handle/lottery_numbers', views.LotteryNumbersApiView.as_view()),
    path('lottery_handle/lottery_alogrithm', views.LotteryAlgorithmsApiView.as_view()),

]
