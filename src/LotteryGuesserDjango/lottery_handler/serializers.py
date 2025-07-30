from rest_framework import serializers




class GetLotteryNumbersWithAlgorithm(serializers.Serializer):
    lottery_type_id = serializers.IntegerField()
    algorithms = serializers.ListField(child=serializers.CharField(), required=False, allow_null=True)


class LotteryNumbers(serializers.Serializer):
    numbers = serializers.JSONField()
    algorithm = serializers.CharField()
    score = serializers.FloatField()


class LotteryAlgorithmSerializer(serializers.Serializer):
    algorithm_type = serializers.CharField()

class PostCalculateLotteryNumbersSerializer(serializers.Serializer):
    winning_numbers = serializers.ListField(child=serializers.IntegerField())
    lottery_type_id = serializers.IntegerField()


class AlgorithmPerformanceSerializer(serializers.Serializer):
    algorithm_name = serializers.CharField()
    current_score = serializers.FloatField()
    total_predictions = serializers.IntegerField()
    success_rate = serializers.FloatField()
    average_execution_time = serializers.FloatField()
    fastest_execution = serializers.FloatField()
    slowest_execution = serializers.FloatField()
    total_executions = serializers.IntegerField()
    last_execution_time = serializers.FloatField()
    score_trend = serializers.ListField(child=serializers.FloatField())
    recent_scores = serializers.ListField(child=serializers.FloatField())
    performance_rank = serializers.IntegerField()
    speed_rank = serializers.IntegerField()
    consistency_score = serializers.FloatField()
    improvement_trend = serializers.CharField()
    last_updated = serializers.DateTimeField()


class DetailedStatisticsSerializer(serializers.Serializer):
    total_algorithms = serializers.IntegerField()
    total_predictions = serializers.IntegerField()
    total_executions = serializers.IntegerField()
    average_score_all = serializers.FloatField()
    top_3_algorithms = serializers.ListField(child=serializers.CharField())
    fastest_algorithms = serializers.ListField(child=serializers.CharField())
    most_consistent_algorithms = serializers.ListField(child=serializers.CharField())
    algorithm_performance = AlgorithmPerformanceSerializer(many=True)
    performance_distribution = serializers.DictField()
    execution_time_distribution = serializers.DictField()
    improvement_trends = serializers.DictField()


class GetStatisticsQuerySerializer(serializers.Serializer):
    lottery_type_id = serializers.IntegerField(required=False)
    days = serializers.IntegerField(default=30, required=False)
    algorithm_filter = serializers.CharField(required=False)
    include_trends = serializers.BooleanField(default=True, required=False)


class QuickAlgorithmRankingSerializer(serializers.Serializer):
    algorithm_name = serializers.CharField()
    current_score = serializers.FloatField()
    rank = serializers.IntegerField()
    performance_category = serializers.CharField()
    average_execution_time = serializers.FloatField()
    success_rate = serializers.FloatField()
    trend = serializers.CharField()


class QuickStatisticsSerializer(serializers.Serializer):
    top_algorithms = QuickAlgorithmRankingSerializer(many=True)
    summary = serializers.DictField()
    last_updated = serializers.DateTimeField()
