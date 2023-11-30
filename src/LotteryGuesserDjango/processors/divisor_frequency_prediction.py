from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    def find_divisors(num):
        return [i for i in range(1, num + 1) if num % i == 0]

    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    divisor_counter = Counter()

    for draw in past_draws:
        for number in draw:
            divisors = find_divisors(number)
            divisor_counter.update(divisors)

    common_divisors = [div for div, _ in divisor_counter.most_common()]
    predicted_numbers = set()
    for div in common_divisors:
        for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
            if number % div == 0:
                predicted_numbers.add(number)
                if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                    break

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]
