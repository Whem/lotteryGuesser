from django.db import models
from django.utils import timezone
class lg_generated_lottery_draw(models.Model):
    lottery_type = models.ForeignKey('algorithms.lg_lottery_type', on_delete=models.CASCADE)
    lottery_type_number = models.JSONField()
    lottery_type_number_year = models.IntegerField(null=True)
    lottery_type_number_week = models.IntegerField(null=True)
    sum = models.IntegerField(null=True)
    average = models.IntegerField(null=True)
    median = models.IntegerField(null=True)
    mode = models.IntegerField(null=True)
    standard_deviation = models.IntegerField(null=True)
    lottery_algorithm = models.CharField(max_length=50, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    

    class Meta:
        db_table = "lg_generated_lottery"

class lg_algorithm_score(models.Model):
    algorithm_name = models.CharField(max_length=100, unique=True)
    current_score = models.FloatField(default=0.0)
    total_predictions = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.algorithm_name} - Score: {self.current_score}"

    class Meta:
        db_table = "lg_algorithm_score"

class lg_prediction_history(models.Model):
    algorithm_name = models.CharField(max_length=100)
    prediction_date = models.DateTimeField(default=timezone.now)
    predicted_numbers = models.JSONField()
    actual_numbers = models.JSONField()
    score = models.FloatField()

    def __str__(self):
        return f"{self.algorithm_name} - {self.prediction_date}"


    class Meta:
        indexes = [
            models.Index(fields=['algorithm_name', 'prediction_date']),
        ]
        db_table = "lg_prediction_history"
