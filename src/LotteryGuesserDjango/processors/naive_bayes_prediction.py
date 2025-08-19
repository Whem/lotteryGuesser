import random
import numpy as np
from typing import List
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using Multinomial Naive Bayes with multi-label classification.
    Handles both main numbers and additional numbers if present.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists: (main_numbers, additional_numbers).
    """
    # Main numbers prediction
    main_numbers = predict_numbers(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Additional numbers prediction if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = predict_numbers(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    # Return as tuple
    return main_numbers, additional_numbers


def predict_numbers(lottery_type_instance, min_num: int, max_num: int, total_numbers: int, is_main: bool) -> List[int]:
    """
    Predicts either main numbers or additional numbers using Naive Bayes.
    """
    num_classes = max_num - min_num + 1

    try:
        # Get past draws based on type
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id')

        # Process past draws based on type (main or additional numbers)
        past_draws = []
        for draw in past_draws_queryset:
            try:
                numbers = draw.lottery_type_number if is_main else draw.additional_numbers
                if isinstance(numbers, list) and len(numbers) == total_numbers:
                    past_draws.append(numbers)
            except (ValueError, TypeError, AttributeError):
                continue

        if len(past_draws) < 10:
            selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
            selected_numbers.sort()
            return selected_numbers

        # Prepare data
        X = []
        y = []
        for i in range(len(past_draws) - 1):
            # Create feature vector
            features = [0] * num_classes
            for num in past_draws[i]:
                if min_num <= num <= max_num:
                    features[num - min_num] = 1
            X.append(features)

            # Next draw as target
            target = [num - min_num for num in past_draws[i + 1] if min_num <= num <= max_num]
            y.append(target)

        X = np.array(X)
        y = np.array(y, dtype=object)

        # Use MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=range(num_classes))
        y_encoded = mlb.fit_transform(y)

        # Train model
        model = OneVsRestClassifier(BernoulliNB())
        model.fit(X, y_encoded)

        # Prepare last draw for prediction
        last_draw = past_draws[-1]
        last_features = [0] * num_classes
        for num in last_draw:
            if min_num <= num <= max_num:
                last_features[num - min_num] = 1

        last_features = np.array([last_features])

        # Make prediction
        predicted_probas = model.predict_proba(last_features)

        # Ensure correct format
        if isinstance(predicted_probas, list):
            predicted_probas = np.array(predicted_probas)
        if len(predicted_probas.shape) == 3:
            predicted_probas = predicted_probas.mean(axis=0)

        predicted_probas = predicted_probas[0]

        # Select numbers
        predicted_numbers_indices = np.argsort(predicted_probas)[-total_numbers:]
        predicted_numbers = [int(idx + min_num) for idx in predicted_numbers_indices]

        # Fill if needed
        if len(predicted_numbers) < total_numbers:
            remaining = total_numbers - len(predicted_numbers)
            all_possible = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            additional_numbers = random.sample(list(all_possible), remaining)
            predicted_numbers.extend(additional_numbers)

        # Trim if too many
        if len(predicted_numbers) > total_numbers:
            predicted_numbers = predicted_numbers[:total_numbers]

        # Convert to int and sort
        predicted_numbers = sorted([int(num) for num in predicted_numbers])

        return predicted_numbers

    except Exception as e:
        # Fallback to random numbers
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers