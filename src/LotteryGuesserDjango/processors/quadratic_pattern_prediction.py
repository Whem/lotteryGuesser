from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    quadratic_differences = Counter()

    for draw in past_draws:
        for i in range(len(draw) - 2):
            quadratic_difference = draw[i + 2] - 2 * draw[i + 1] + draw[i]
            quadratic_differences[quadratic_difference] += 1

    common_quadratic_difference = quadratic_differences.most_common(1)[0][0]
    predicted_numbers = set()

    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if number not in predicted_numbers and common_quadratic_difference in quadratic_differences:
            predicted_numbers.add(number)

    return sorted(predicted_numbers)