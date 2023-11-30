import pandas as pd

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    transition_matrix = pd.DataFrame(0, index=range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
                                     columns=range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))

    # Build transition matrix
    for i in range(len(past_draws) - 1):
        for num in past_draws[i]:
            for next_num in past_draws[i + 1]:
                transition_matrix.loc[num, next_num] += 1

    # Normalize the transition probabilities
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

    # Predict the next numbers based on transition probabilities
    last_draw = past_draws[-1]
    predicted_numbers = transition_matrix.loc[last_draw].sum().nlargest(lottery_type_instance.pieces_of_draw_numbers).index

    return predicted_numbers.tolist()
