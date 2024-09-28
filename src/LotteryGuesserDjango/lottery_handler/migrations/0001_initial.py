# Generated by Django 4.2.6 on 2024-01-20 20:28

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('algorithms', '0006_delete_lg_lottery_algorithm_type_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='lg_generated_lottery_draw',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lottery_type_number', models.JSONField()),
                ('lottery_type_number_year', models.IntegerField(null=True)),
                ('lottery_type_number_week', models.IntegerField(null=True)),
                ('sum', models.IntegerField(null=True)),
                ('average', models.IntegerField(null=True)),
                ('median', models.IntegerField(null=True)),
                ('mode', models.IntegerField(null=True)),
                ('standard_deviation', models.IntegerField(null=True)),
                ('lottery_algorithm', models.CharField(max_length=50, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('lottery_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='algorithms.lg_lottery_type')),
            ],
            options={
                'db_table': 'lg_generated_lottery',
            },
        ),
    ]
