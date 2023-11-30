from django.db import models

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
