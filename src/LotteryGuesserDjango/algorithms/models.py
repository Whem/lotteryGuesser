from django.db import models

# Create your models here.

class lg_lottery_type(models.Model):
    draw_day_choices = (
        ('Monday', 'Monday'),
        ('Tuesday', 'Tuesday'),
        ('Wednesday', 'Wednesday'),
        ('Thursday', 'Thursday'),
        ('Friday', 'Friday'),
        ('Saturday', 'Saturday'),
        ('Sunday', 'Sunday'),
    )

    url = models.CharField(max_length=50, null=True)
    lottery_type = models.CharField(max_length=50, null=True)
    lottery_type_description = models.CharField(max_length=50, null=True)
    image_url = models.CharField(max_length=50, null=True)
    min_number = models.IntegerField()
    max_number = models.IntegerField()
    skip_items = models.IntegerField(null=True)
    pieces_of_draw_numbers = models.IntegerField(null=True)
    draw_day = models.CharField(max_length=50, choices=draw_day_choices, null=True)
    draw_time = models.TimeField(null=True)
    has_additional_numbers = models.BooleanField(default=False)
    additional_min_number = models.IntegerField(null=True)
    additional_max_number = models.IntegerField(null=True)
    additional_numbers_count = models.IntegerField(null=True)

    class Meta:
        db_table = "lg_lottery_type"


class lg_lottery_winner_number(models.Model):
    lottery_type = models.ForeignKey(lg_lottery_type, on_delete=models.CASCADE)
    lottery_type_number = models.JSONField()
    lottery_type_number_year = models.IntegerField(null=True)
    lottery_type_number_week = models.IntegerField(null=True)
    additional_numbers = models.JSONField(null=True)
    sum = models.IntegerField(null=True)
    average = models.IntegerField(null=True)
    median = models.IntegerField(null=True)
    mode = models.IntegerField(null=True)
    standard_deviation = models.IntegerField(null=True)

    class Meta:
        db_table = "lg_lottery_winner_number"