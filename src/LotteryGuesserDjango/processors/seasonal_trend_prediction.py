from collections import Counter

from django.db.models.functions import Extract

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    current_season = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).latest('draw_date').draw_date.quarter

    past_draws_with_seasons = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).annotate(
        season=Extract('draw_date', 'quarter')
    ).values_list('lottery_type_number', 'season', flat=True)

    season_counter = Counter()
    for draw, season in past_draws_with_seasons:
        if season == current_season:
            season_counter.update(draw)

    seasonal_trends = [num for num, _ in season_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(seasonal_trends)