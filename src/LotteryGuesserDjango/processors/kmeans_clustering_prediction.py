# kmeans_clustering_prediction.py

# Generates lottery numbers using K-Means clustering.
# Applies K-Means clustering to past draws to find clusters of numbers.
# Predicts numbers based on cluster centers.
# Supports both main and additional numbers.

import random
import numpy as np
from sklearn.cluster import KMeans
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using K-Means clustering.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists: (main_numbers, additional_numbers).
    """
    # Main numbers configuration
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id')

    # Process main numbers
    past_main_numbers = []
    for draw in past_draws_queryset:
        main_numbers = draw.lottery_type_number
        if isinstance(main_numbers, list) and len(main_numbers) == total_numbers:
            past_main_numbers.append(main_numbers)

    # Generate main numbers
    main_numbers = generate_numbers(
        past_numbers=past_main_numbers,
        min_num=min_num,
        max_num=max_num,
        total_numbers=total_numbers
    )

    # Handle additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_min_num = lottery_type_instance.additional_min_number
        additional_max_num = lottery_type_instance.additional_max_number
        additional_total_numbers = lottery_type_instance.additional_numbers_count

        # Process additional numbers
        past_additional_numbers = []
        for draw in past_draws_queryset:
            additional_nums = getattr(draw, 'additional_numbers', [])
            if isinstance(additional_nums, list) and len(additional_nums) == additional_total_numbers:
                past_additional_numbers.append(additional_nums)

        # Generate additional numbers
        additional_numbers = generate_numbers(
            past_numbers=past_additional_numbers,
            min_num=additional_min_num,
            max_num=additional_max_num,
            total_numbers=additional_total_numbers
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(past_numbers, min_num, max_num, total_numbers):
    """
    Generates a set of numbers using K-Means clustering.

    Parameters:
    - past_numbers: List of past draws (list of lists).
    - min_num: Minimum number in the lottery.
    - max_num: Maximum number in the lottery.
    - total_numbers: Number of numbers to predict.

    Returns:
    - A list of predicted numbers.
    """
    if len(past_numbers) < 10:
        # Not enough data to perform clustering
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        return sorted(selected_numbers)

    # Prepare data for clustering
    data = []
    for draw in past_numbers:
        data.extend(draw)
    data = np.array(data).reshape(-1, 1)

    # Perform K-Means clustering
    k = total_numbers  # Number of clusters equal to numbers to select
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(data)

    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()
    predicted_numbers = [int(round(center)) for center in centers]
    predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

    # Ensure unique numbers
    predicted_numbers = list(set(predicted_numbers))

    # If not enough numbers, fill with random numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    else:
        predicted_numbers = predicted_numbers[:total_numbers]

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    # Sort and return the numbers
    return sorted(predicted_numbers)
