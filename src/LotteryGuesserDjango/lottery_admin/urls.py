from django.urls import path

from . import views

urlpatterns = [
    path('lottery_admin/lottery_type', views.LotteryTypeApiView.as_view()),
    path('lottery_admin/lottery_type_fill', views.FillLotteryTypeApiView.as_view()),
]
