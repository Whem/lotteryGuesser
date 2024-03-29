# Generated by Django 4.2.6 on 2023-11-12 17:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('algorithms', '0003_remove_lg_lottery_winner_number_lottery_type_number_date_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='lg_lottery_winner_number',
            name='average',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='lg_lottery_winner_number',
            name='median',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='lg_lottery_winner_number',
            name='mode',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='lg_lottery_winner_number',
            name='standard_deviation',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='lg_lottery_winner_number',
            name='sum',
            field=models.IntegerField(null=True),
        ),
    ]
