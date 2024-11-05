from django.http import JsonResponse
from drf_spectacular.utils import extend_schema
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView

from LotteryGuesserV2.pagination import CustomPagination
from algorithms.models import *
from lottery_handler.models import *
from lottery_handler.serializers import *
from lottery_handler.signals import list_processor_files, call_get_numbers_dynamically
import json
import importlib
import os
import random
from typing import List, Set
from collections import Counter
from itertools import combinations
import statistics
import numpy as np
import time

import traceback
from contextlib import contextmanager
from django.utils import timezone
class LotteryNumbersApiView(APIView, CustomPagination):
    permission_classes = (AllowAny,)
    pagination_class = CustomPagination

    @extend_schema(
        summary="Get Lottery Numbers",
        parameters=[GetLotteryNumbersWithAlgorithm],
        responses={
            200: LotteryNumbers(many=True), })
    @action(methods=['GET'], detail=True, pagination_class=CustomPagination)
    def get(self, request):
        serializer = GetLotteryNumbersWithAlgorithm(data=request.GET)
        serializer.is_valid(raise_exception=True)

        lottery_type_id = serializer.validated_data["lottery_type_id"]
        lottery_type = lg_lottery_type.objects.filter(id=lottery_type_id).first()
        if lottery_type is None:
            return JsonResponse({"error": "Item not found"}, status=404)

        response = []

        try:
            results = call_get_numbers_dynamically(lottery_type)

            # Lekérjük az összes algoritmus score-ját
            algorithm_scores = {score.algorithm_name: score.current_score for score in lg_algorithm_score.objects.all()}

            for result in results:
                data = {
                    "numbers": result.lottery_type_number,
                    'algorithm': result.lottery_algorithm,
                    'score': algorithm_scores.get(result.lottery_algorithm, 0)  # Ha nincs score, 0-t adunk
                }
                response_serializer = LotteryNumbers(data=data)
                response_serializer.is_valid(raise_exception=True)
                response.append(response_serializer.data)

            # Rendezzük az eredményeket score alapján csökkenő sorrendben
            response.sort(key=lambda x: x['score'], reverse=True)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

        results = self.paginate_queryset(response, request, view=self)
        return self.get_paginated_response(results)


class LotteryAlgorithmsApiView(APIView):
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Get Lottery Algorithms",
        responses={
            200: LotteryAlgorithmSerializer(many=True), })

    def get(self, request):
        response = []
        result = list_processor_files()

        for item in result:
            data = {
                "algorithm_type": item
            }

            response_serializer = LotteryAlgorithmSerializer(data=data)
            response_serializer.is_valid(raise_exception=True)
            response.append(response_serializer.data)
        return JsonResponse(response, safe=False)


from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from django.http import JsonResponse
from drf_spectacular.utils import extend_schema
from .serializers import PostCalculateLotteryNumbersSerializer, LotteryNumbers
from .models import  lg_algorithm_score, lg_prediction_history, lg_generated_lottery_draw
import json
import os
import importlib
from django.utils import timezone
import statistics


class CalculateLotteryNumbersView(APIView):
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Calculate Lottery Numbers",
        request=PostCalculateLotteryNumbersSerializer,
        responses={
            200: LotteryNumbers(many=True), })
    def post(self, request):
        data = json.loads(request.body)
        lottery_type_id = data.get('lottery_type_id')
        winning_numbers = data.get('winning_numbers')

        if not lottery_type_id or not winning_numbers:
            return JsonResponse({"error": "Missing lottery_type_id or winning_numbers"}, status=400)

        lottery_type = lg_lottery_type.objects.get(id=lottery_type_id)

        for x in range(20):
            algorithms = self.evaluate_algorithms(lottery_type, winning_numbers)

        return JsonResponse({"ranked_algorithms": algorithms})

    def evaluate_algorithms(self, lottery_type, winning_numbers):
        algorithms = []
        processors_dir = "processors"

        for filename in os.listdir(processors_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                print(f"Calling get_numbers function in {module_name} module...")
                module = importlib.import_module(f"processors.{module_name}")

                if hasattr(module, 'get_numbers'):
                    try:
                        # Start performance measurement
                        start_time = time.time()

                        # Execute prediction
                        predicted_numbers,additional_numbers = module.get_numbers(lottery_type)

                        # End performance measurement
                        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                        # Calculate score
                        score = self.calculate_score(predicted_numbers, winning_numbers)

                        # Save prediction history
                        lg_prediction_history.objects.create(
                            algorithm_name=module_name,
                            predicted_numbers=predicted_numbers,
                            actual_numbers=winning_numbers,
                            score=score
                        )

                        # Save performance history
                        lg_performance_history.objects.create(
                            algorithm_name=module_name,
                            execution_time=execution_time,
                            success=True
                        )

                        # Update performance metrics
                        performance, created = lg_algorithm_performance.objects.get_or_create(
                            algorithm_name=module_name,
                            defaults={
                                'average_execution_time': execution_time,
                                'fastest_execution': execution_time,
                                'slowest_execution': execution_time,
                                'total_executions': 1,
                                'last_execution_time': execution_time
                            }
                        )

                        if not created:
                            # Update average execution time
                            total_time = (
                                                     performance.average_execution_time * performance.total_executions) + execution_time
                            performance.total_executions += 1
                            performance.average_execution_time = total_time / performance.total_executions

                            # Update fastest/slowest times
                            performance.fastest_execution = min(performance.fastest_execution, execution_time)
                            performance.slowest_execution = max(performance.slowest_execution, execution_time)

                            # Update last execution time
                            performance.last_execution_time = execution_time
                            performance.save()

                        # Save generated lottery draw
                        self.save_generated_lottery_draw(lottery_type, predicted_numbers, module_name)

                        # Update algorithm score
                        self.update_algorithm_score(module_name)

                        current_score = lg_algorithm_score.objects.get(algorithm_name=module_name).current_score
                        algorithms.append({
                            "name": module_name,
                            "score": current_score,
                            "execution_time": execution_time
                        })

                    except Exception as e:
                        print(f"Error calling get_numbers function in {module_name} module: {e}")
                        # Log error in performance history
                        lg_performance_history.objects.create(
                            algorithm_name=module_name,
                            execution_time=0,
                            success=False,
                            error_message=str(e)
                        )

        # Rendezzük az algoritmusokat pontszám szerint csökkenő sorrendbe
        ranked_algorithms = sorted(algorithms, key=lambda x: x['score'], reverse=True)

        return ranked_algorithms

    def calculate_score(self, predicted_numbers, winning_numbers):
        correct_numbers = set(predicted_numbers) & set(winning_numbers)
        base_score = len(correct_numbers) * 10  # Minden helyes találat 10 pontot ér

        # Bónusz pontok a helyes pozíciókért
        position_bonus = sum(5 for i, num in enumerate(predicted_numbers) if num == winning_numbers[i])

        return base_score + position_bonus

    def update_algorithm_score(self, algorithm_name):
        new_score = self.calculate_moving_average_score(algorithm_name)
        algorithm_score, created = lg_algorithm_score.objects.get_or_create(algorithm_name=algorithm_name)
        algorithm_score.current_score = new_score
        algorithm_score.total_predictions += 1
        algorithm_score.save()

    def calculate_moving_average_score(self, algorithm_name, window_size=10):
        latest_predictions = lg_prediction_history.objects.filter(algorithm_name=algorithm_name).order_by(
            '-prediction_date')[:window_size]
        if not latest_predictions:
            return 0.0

        total_score = sum(pred.score for pred in latest_predictions)
        return total_score / len(latest_predictions)

    def save_generated_lottery_draw(self, lottery_type, predicted_numbers, algorithm_name):
        current_date = timezone.now()
        lg_generated_lottery_draw.objects.create(
            lottery_type=lottery_type,
            lottery_type_number=predicted_numbers,
            lottery_type_number_year=current_date.year,
            lottery_type_number_week=current_date.isocalendar()[1],
            sum=sum(predicted_numbers),
            average=statistics.mean(predicted_numbers),
            median=statistics.median(predicted_numbers),
            mode=statistics.mode(predicted_numbers),
            standard_deviation=statistics.stdev(predicted_numbers),
            lottery_algorithm=algorithm_name
        )


