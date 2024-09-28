# neural_network_time_series_prediction.py

import sys
import os
import numpy as np
import random
from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import tensorflow as tf

# Állítsd be a TensorFlow logolási szintjét és letiltsd a oneDNN optimalizációkat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Csak FATAL hibák
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN optimalizációk kikapcsolása

# Reconfigure stdout to use UTF-8 (opcionális, ha még mindig vannak Unicode hibák)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# További TensorFlow logolási szint beállítása
tf.get_logger().setLevel('ERROR')


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    # Lekérdezzük a múltbeli húzásokat és listává alakítjuk
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    past_draws = list(past_draws_queryset)  # Konvertálás listává
    print(f"Total past_draws retrieved: {len(past_draws)}")

    # Ellenőrizzük, hogy minden húzás listája és a megfelelő számú számot tartalmazza
    past_draws = [
        draw for draw in past_draws
        if isinstance(draw, list) and len(draw) == lottery_type_instance.pieces_of_draw_numbers
    ]
    print(f"Valid past_draws after filtering: {len(past_draws)}")

    if len(past_draws) < 50:
        selected_numbers = random.sample(
            range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
            lottery_type_instance.pieces_of_draw_numbers
        )
        selected_numbers = [int(num) for num in selected_numbers]  # Konvertálás Python int-re
        selected_numbers.sort()
        print(f"Not enough past draws. Selected random numbers: {selected_numbers}")
        return selected_numbers

    # Adatok előkészítése
    data = np.array([list(draw) for draw in past_draws])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Szekvenciák létrehozása
    X, y = [], []
    window_size = 10
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])
    X, y = np.array(X), np.array(y)

    # Modell létrehozása és tanítása
    model = Sequential([
        Input(shape=(window_size, lottery_type_instance.pieces_of_draw_numbers)),  # Használj Input réteget
        LSTM(50, activation='relu'),
        Dense(lottery_type_instance.pieces_of_draw_numbers)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    # Előrejelzés
    last_sequence = scaled_data[-window_size:].reshape(1, window_size, lottery_type_instance.pieces_of_draw_numbers)
    predicted_scaled = model.predict(last_sequence)
    predicted_numbers = scaler.inverse_transform(predicted_scaled).round().astype(int)[0]

    # Korrekció a megengedett tartományra és konvertálás Python int-re
    predicted_numbers = np.clip(predicted_numbers, lottery_type_instance.min_number,
                                lottery_type_instance.max_number).astype(int)

    # Kiválasztjuk a számtartományon belül lévő egyedi számokat
    predicted_numbers = set(predicted_numbers)

    # Hiányzó számok pótlása véletlenszerűen
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    # Visszatérés rendezett listával és konvertálás Python int-re
    return sorted([int(num) for num in predicted_numbers])[:lottery_type_instance.pieces_of_draw_numbers]
