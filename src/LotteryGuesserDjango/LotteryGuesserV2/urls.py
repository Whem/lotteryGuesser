
from unittest.mock import patch

from django.urls import path, include, re_path
from django.views.generic import TemplateView
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

from rest_framework import routers
from rest_framework.schemas import get_schema_view
from rest_framework_swagger.views import get_swagger_view



router = routers.DefaultRouter()

def patch_the_method(func):
    def inner(*args, **kwargs):
        with patch('rest_framework.permissions.IsAuthenticated.has_permission', return_value=True):
            response = func(*args, **kwargs)
        return response

    return inner


schema_view = patch_the_method(get_swagger_view(title='Some API'))
openapi_schema_view = patch_the_method(get_schema_view(title="Some API", description="API for all things …", ), )

urlpatterns = [
    re_path(r'^', include(router.urls)),
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    # Optional UI:
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    path('lottery_admin/', include('lottery_admin.urls')),
    path('lottery_handler/', include('lottery_handler.urls')),
]
