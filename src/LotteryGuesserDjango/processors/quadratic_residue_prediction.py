#Selects numbers based on quadratic residues, a concept from number theory.
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    def is_quadratic_residue(number, modulo):
        for i in range(modulo):
            if (i * i) % modulo == number % modulo:
                return True
        return False

    modulo = 10  # Example modulo value
    quadratic_residue_counter = Counter()

    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    for draw in past_draws:
        for number in draw:
            if is_quadratic_residue(number, modulo):
                quadratic_residue_counter[number] += 1

    most_common_quadratic_residues = [num for num, _ in quadratic_residue_counter.most_common(lottery_type_instance.pieces_of_draw_numbers)]
    return sorted(most_common_quadratic_residues)
