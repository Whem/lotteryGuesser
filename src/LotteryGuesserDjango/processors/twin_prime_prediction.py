import sympy


def get_numbers(lottery_type_instance):
    predicted_numbers = set()

    for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number):
        if sympy.isprime(number) and sympy.isprime(number + 2):
            predicted_numbers.add(number)
            predicted_numbers.add(number + 2)
            if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
                break

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]
