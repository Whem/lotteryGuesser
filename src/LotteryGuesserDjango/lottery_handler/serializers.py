from rest_framework import serializers




class GetLotteryNumbersWithAlgorithm(serializers.Serializer):
    lottery_type_id = serializers.IntegerField()
    algorithms = serializers.ListField(child=serializers.CharField(), required=False, allow_null=True)


class LotteryNumbers(serializers.Serializer):
    numbers = serializers.ListField(child=serializers.IntegerField())
    algorithms = serializers.CharField()


class LotteryAlgorithmSerializer(serializers.Serializer):
    algorithm_type = serializers.CharField()
