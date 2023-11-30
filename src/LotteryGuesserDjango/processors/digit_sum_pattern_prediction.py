from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    def digit_sum(number):
        return sum(int(digit) for digit in str(number))

    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    digit_sum_counter = Counter()

    for draw in past_draws:
        for number in draw:
            digit_sum_counter[digit_sum(number)] += 1

    common_digit_sum = digit_sum_counter.most_common(1)[0][0]
    predicted_numbers = set()

    for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
        if digit_sum(number) == common_digit_sum:
            predicted_numbers.add(number)
            if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                break

    return sorted(predicted_numbers)
