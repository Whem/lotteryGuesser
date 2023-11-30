from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    last_digit_counter = Counter()

    for draw in past_draws:
        for number in draw:
            last_digit = number % 10
            last_digit_counter[last_digit] += 1

    most_common_last_digits = [digit for digit, _ in last_digit_counter.most_common()]
    predicted_numbers = set()
    for digit in most_common_last_digits:
        for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
            if number % 10 == digit:
                predicted_numbers.add(number)
                if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                    break

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]
