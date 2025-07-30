from django.urls import path

from . import views

urlpatterns = [
    path('lottery_handle/lottery_numbers', views.LotteryNumbersApiView.as_view()),
    path('lottery_handle/lottery_alogrithm', views.LotteryAlgorithmsApiView.as_view()),
    path('lottery_handle/get_algorithms_with_scores', views.CalculateLotteryNumbersView.as_view()),
    
    # Új statisztika végpontok
    path('lottery_handle/detailed_statistics', views.DetailedAlgorithmStatisticsView.as_view()),
    path('lottery_handle/quick_statistics', views.QuickStatisticsView.as_view()),
]
