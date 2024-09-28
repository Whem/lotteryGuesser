import random
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using Multinomial Naive Bayes with multi-label classification.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Paraméterek beállítása
    min_num = lottery_type_instance.min_number  # Minimum szám a lottón
    max_num = lottery_type_instance.max_number  # Maximum szám a lottón
    total_numbers = lottery_type_instance.pieces_of_draw_numbers  # Húzandó számok száma
    num_classes = max_num - min_num + 1  # Lehetséges számok száma

    print(f"min_num: {min_num}, max_num: {max_num}, total_numbers: {total_numbers}, num_classes: {num_classes}")

    # Korábbi nyerőszámok lekérése
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    # Korábbi húzások szűrése és előkészítése
    past_draws = [
        draw for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    print(f"Number of past_draws: {len(past_draws)}")

    if len(past_draws) < 10:
        # Nem elég adat, véletlenszerű számok kiválasztása
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        print(f"Not enough past draws. Selected random numbers: {selected_numbers}")
        return selected_numbers

    # Adatok előkészítése
    X = []
    y = []
    for i in range(len(past_draws) - 1):
        # Jellemző vektor létrehozása az aktuális húzáshoz
        features = [0] * num_classes
        for num in past_draws[i]:
            if min_num <= num <= max_num:
                features[num - min_num] = 1
            else:
                print(f"Number {num} out of range.")
        X.append(features)

        # Cél a következő húzás
        target = [num - min_num for num in past_draws[i + 1] if min_num <= num <= max_num]
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y before binarization: {y.shape}")

    # MultiLabelBinarizer használata a y binarizálásához
    mlb = MultiLabelBinarizer(classes=range(num_classes))
    y_encoded = mlb.fit_transform(y)

    print(f"Shape of y_encoded: {y_encoded.shape}")
    print(f"Sample y_encoded: {y_encoded[:1]}")

    # Ellenőrizni, hogy a y_encoded oszlopainak száma megegyezik-e a num_classes-szal
    if y_encoded.shape[1] != num_classes:
        print(f"Warning: Number of classes in y_encoded ({y_encoded.shape[1]}) does not match num_classes ({num_classes}).")

    # Modell betanítása OneVsRestClassifier-rel és MultinomialNB-vel vagy BernoulliNB-vel
    # A BernoulliNB jobban működhet bináris jellemzők esetén
    model = OneVsRestClassifier(BernoulliNB())
    try:
        model.fit(X, y_encoded)
        print("Model fitting successful.")
    except Exception as e:
        print(f"Error during model.fit: {e}")
        raise

    # Utolsó húzás jellemzőinek előkészítése a predikcióhoz
    last_draw = past_draws[-1]
    last_features = [0] * num_classes
    for num in last_draw:
        if min_num <= num <= max_num:
            last_features[num - min_num] = 1
        else:
            print(f"Number {num} out of range in last_draw.")

    last_features = np.array([last_features])
    print(f"Shape of last_features: {last_features.shape}")

    try:
        # Valószínűségek predikciója
        predicted_probas = model.predict_proba(last_features)[0]
        print(f"Shape of predicted_probas: {predicted_probas.shape}")
        print(f"Sample predicted_probas: {predicted_probas[:5]}")
    except Exception as e:
        print(f"Error during model.predict_proba: {e}")
        raise

    # Számok kiválasztása a legmagasabb valószínűségek alapján
    predicted_numbers_indices = np.argsort(predicted_probas)[-total_numbers:]
    predicted_numbers = [int(idx + min_num) for idx in predicted_numbers_indices]

    print(f"Predicted numbers before padding: {predicted_numbers}")

    # Biztosítani, hogy pontosan a kívánt számú szám legyen kiválasztva
    if len(predicted_numbers) < total_numbers:
        # Hiányzó számok pótlása véletlenszerűen
        remaining = total_numbers - len(predicted_numbers)
        all_possible = set(range(min_num, max_num + 1)) - set(predicted_numbers)
        all_possible_list = sorted(all_possible)

        if len(all_possible_list) < remaining:
            # Nem elég lehetséges szám, minden számot visszaadunk
            additional_numbers = list(all_possible_list)
        else:
            additional_numbers = random.sample(all_possible_list, remaining)
        predicted_numbers.extend(additional_numbers)
        print(f"Additional numbers added: {additional_numbers}")

    if len(predicted_numbers) > total_numbers:
        # Ha túl sok szám van, akkor kiválogatjuk a legvalószínűbbeket
        predicted_numbers = predicted_numbers[:total_numbers]
        print(f"Trimmed predicted_numbers to {total_numbers}: {predicted_numbers}")

    # Konvertáljuk a számokat int típusra JSON serializálási problémák elkerülése végett
    predicted_numbers = [int(num) for num in predicted_numbers]

    # Rendezés
    predicted_numbers.sort()
    print(f"Final predicted_numbers: {predicted_numbers}")

    return predicted_numbers
