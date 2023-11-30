import random

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number',
                                                                                              flat=True))
    min_number = lottery_type_instance.min_number
    max_number = lottery_type_instance.max_number
    num_draws = lottery_type_instance.pieces_of_draw_numbers

    if not past_draws:
        return [random.randint(min_number, max_number) for _ in range(num_draws)]

    last_draw = past_draws[-1]
    selected_numbers = set()

    while len(selected_numbers) < num_draws:
        for number in last_draw:
            # Randomly decide whether to move up, down, or stay
            move = random.choice([-1, 0, 1])
            next_number = number + move

            # Ensure the number is within bounds
            if min_number <= next_number <= max_number:
                selected_numbers.add(next_number)
                if len(selected_numbers) == num_draws:
                    break

    return sorted(selected_numbers)