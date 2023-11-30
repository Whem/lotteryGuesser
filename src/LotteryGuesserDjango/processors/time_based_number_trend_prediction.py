from collections import Counter

from django.db.models.functions import ExtractMonth
from django.utils import timezone

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):

    current_month = timezone.now().month
    past_draws_with_months = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).annotate(
        month=ExtractMonth('draw_date')
    ).values_list('lottery_type_number', 'month', flat=True)

    month_counter = Counter()
    for draw, month in past_draws_with_months:
        if month == current_month:
            month_counter.update(draw)

    month_trends = [num for num, _ in month_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(month_trends)