from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type):
    queryset = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).values_list('lottery_type_number',
                                                                                              flat=True)

    # Dictionary to hold intervals for each number
    intervals = {}
    last_occurrence = {}

    # Calculate intervals
    for draw_index, numbers in enumerate(queryset):
        for number in numbers:
            if number in last_occurrence:
                interval = draw_index - last_occurrence[number]
                if number in intervals:
                    intervals[number].append(interval)
                else:
                    intervals[number] = [interval]
            last_occurrence[number] = draw_index

    # Calculate average intervals
    average_intervals = {number: sum(interval_list) / len(interval_list) for number, interval_list in intervals.items()}

    # Predict numbers based on shortest average intervals
    # Assuming you want to pick numbers that are 'due' to appear
    predicted_numbers = sorted(average_intervals, key=average_intervals.get)[:lottery_type.pieces_of_draw_numbers]

    return predicted_numbers