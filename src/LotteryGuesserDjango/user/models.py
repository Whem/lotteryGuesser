from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils import timezone
from rest_framework_simplejwt.tokens import RefreshToken

from LotteryGuesserV2 import settings
from LotteryGuesserV2.models import TimestampedModel
from .managers import UserManager


class User(AbstractBaseUser, PermissionsMixin, TimestampedModel):
    username = models.CharField(db_index=True, max_length=255, unique=True)
    last_action_time = models.DateTimeField(null=True)
    email = models.EmailField(db_index=True, unique=True)
    first_name = models.CharField(max_length=255, blank=True)
    last_name = models.CharField(max_length=255, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_verified = models.BooleanField(default=False)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']


    objects = UserManager()

    class Meta:
        db_table = 'lg_user'

    def __str__(self):
        return self.email

    @property
    def is_online(self) -> bool:
        if not self.last_action_time:
            return False

        time_since_last_action = timezone.now() - self.last_action_time
        return time_since_last_action.total_seconds() < settings.ONLINE_TIMEOUT

    @property
    def token(self):
        """
        Allows us to get a user's token by calling `user.token` instead of
        `user.generate_jwt_token().

        The `@property` decorator above makes this possible. `token` is called
        a "dynamic property".
        """

        refresh = RefreshToken.for_user(self)

        return {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }

    @property
    def jwt_token(self):
        refresh = RefreshToken.for_user(self)

        return [str(refresh.access_token), str(refresh)]
