from django.utils import timezone
from rest_framework_simplejwt.tokens import AccessToken

from user.models import User


class LastActionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            token = request.META.get('HTTP_AUTHORIZATION', " ").split(' ')[1]
            if token:
                access_token_obj = AccessToken(token)
                user_id = access_token_obj['user_id']
                user = User.objects.get(id=user_id)
                if user:
                    user.last_action_time = timezone.now()
                    user.save(update_fields=['last_action_time'])
        except Exception as e:
            print(e)
        return self.get_response(request)



