from rest_framework import serializers


class SuccessSerializer(serializers.Serializer):
    success = serializers.BooleanField(default=True)