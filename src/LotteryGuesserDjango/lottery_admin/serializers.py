from rest_framework import serializers


class PostLotteryTypeSerializers(serializers.Serializer):
    id = serializers.IntegerField(required=False)
    url = serializers.CharField(required=False, allow_null=True)
    lottery_type = serializers.CharField(required=False, allow_null=True)
    lottery_type_description = serializers.CharField(required=False, allow_null=True)
    image_url = serializers.CharField(required=False, allow_null=True)
    min_number = serializers.IntegerField(required=False, allow_null=True)
    max_number = serializers.IntegerField(required=False, allow_null=True)
    skip_items = serializers.IntegerField(required=False, allow_null=True)
    pieces_of_draw_numbers = serializers.IntegerField(required=False, allow_null=True)
    has_additional_numbers = serializers.BooleanField(default=False)
    additional_min_number = serializers.IntegerField(required=False, allow_null=True)
    additional_max_number = serializers.IntegerField(required=False, allow_null=True)
    additional_numbers_count = serializers.IntegerField(required=False, allow_null=True)


class FillLotteryTypeWithWinNumbers(serializers.Serializer):
    lottery_type_id = serializers.IntegerField()
