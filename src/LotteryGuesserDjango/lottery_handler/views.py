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
import logging

logger = logging.getLogger(__name__)
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
        try:
            serializer = GetLotteryNumbersWithAlgorithm(data=request.GET)
            serializer.is_valid(raise_exception=True)

            lottery_type_id = serializer.validated_data["lottery_type_id"]
            lottery_type = lg_lottery_type.objects.select_related().filter(id=lottery_type_id).first()
            if lottery_type is None:
                return JsonResponse({"error": "Lottery type not found"}, status=404)

            response = []

            # Optimalizált predikció futtatás
            results = call_get_numbers_dynamically(lottery_type)

            # Batch lekérés az algoritmus score-okhoz
            algorithm_scores = {
                score.algorithm_name: score.current_score 
                for score in lg_algorithm_score.objects.only('algorithm_name', 'current_score')
            }

            # Eredmények feldolgozása
            for result in results:
                # Számok formázása
                numbers = result.lottery_type_number
                if isinstance(numbers, dict):
                    # Main és additional számok kezelése
                    formatted_numbers = {
                        "main": numbers.get("main", []),
                        "additional": numbers.get("additional", [])
                    }
                else:
                    # Legacy formátum
                    formatted_numbers = numbers

                data = {
                    "numbers": formatted_numbers,
                    'algorithm': result.lottery_algorithm,
                    'score': algorithm_scores.get(result.lottery_algorithm, 0),
                    'execution_time': getattr(result, 'execution_time', None),
                    'created_at': result.created_at.isoformat() if result.created_at else None,
                    'statistics': {
                        'sum': getattr(result, 'sum', None),
                        'average': getattr(result, 'average', None),
                        'median': getattr(result, 'median', None),
                        'std_dev': getattr(result, 'standard_deviation', None)
                    }
                }
                
                # Validáció optimalizálva
                response_serializer = LotteryNumbers(data={
                    'numbers': data['numbers'],
                    'algorithm': data['algorithm'],
                    'score': data['score']
                })
                if response_serializer.is_valid():
                    # Teljes adatokat hozzáadjuk
                    response_data = response_serializer.data
                    response_data.update({
                        'execution_time': data['execution_time'],
                        'created_at': data['created_at'],
                        'statistics': data['statistics']
                    })
                    response.append(response_data)

            # Rendezés score szerint
            response.sort(key=lambda x: x['score'], reverse=True)

            # Deduplikálás: azonos számsorok összevonása (a legmagasabb score marad)
            def _canonical_key(numbers):
                # Normalizált kulcs a számokhoz: rendezett tuple
                if isinstance(numbers, dict):
                    main = numbers.get('main', [])
                    additional = numbers.get('additional', [])
                    return ('dict', tuple(sorted(main)), tuple(sorted(additional)))
                else:
                    return ('list', tuple(sorted(numbers)))

            seen = set()
            deduped = []
            for item in response:
                key = _canonical_key(item.get('numbers', []))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            response = deduped

            # Teljesítménystatisztikák hozzáadása
            performance_summary = {
                'total_algorithms': len(response),
                'average_score': sum(r['score'] for r in response) / len(response) if response else 0,
                'top_score': response[0]['score'] if response else 0,
                'response_time': timezone.now().isoformat()
            }

            # Paginált válasz teljesítménystatisztikákkal
            paginated_results = self.paginate_queryset(response, request, view=self)
            paginated_response = self.get_paginated_response(paginated_results)
            
            # Statisztikák hozzáadása a válaszhoz
            paginated_response.data['performance_summary'] = performance_summary
            
            return paginated_response

        except Exception as e:
            logger.error(f"Hiba a lottery numbers endpoint-ban: {e}")
            return JsonResponse({
                "error": "Internal server error",
                "message": str(e) if hasattr(e, '__str__') else "Unknown error occurred"
            }, status=500)


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
from .serializers import (PostCalculateLotteryNumbersSerializer, LotteryNumbers, 
                         AlgorithmPerformanceSerializer, DetailedStatisticsSerializer,
                         GetStatisticsQuerySerializer, QuickStatisticsSerializer, 
                         QuickAlgorithmRankingSerializer)
from .performance_monitor import performance_monitor, DatabasePerformanceAnalyzer
from .models import  lg_algorithm_score, lg_prediction_history, lg_generated_lottery_draw, lg_algorithm_performance, lg_performance_history
import json
import os
import importlib
from django.utils import timezone
from django.db.models import Avg, Count, Max, Min
from datetime import datetime, timedelta
import statistics
import numpy as np


class CalculateLotteryNumbersView(APIView):
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Calculate Lottery Numbers",
        request=PostCalculateLotteryNumbersSerializer,
        responses={
            200: LotteryNumbers(many=True), })
    def post(self, request):
        logger.info("="*60)
        logger.info("LOTTERY ALGORITHM TESTING STARTED")
        logger.info("="*60)
        
        try:
            data = request.data
            lottery_type_id = data.get('lottery_type_id')
            winning_numbers = data.get('winning_numbers')
            additional_numbers = data.get('additional_numbers', [])
            test_iterations = data.get('test_iterations', 10)  # Configurable iterations

            logger.info(f"Lottery Type ID: {lottery_type_id}")
            logger.info(f"Winning Numbers: {winning_numbers}")
            logger.info(f"Additional Numbers: {additional_numbers}")
            logger.info(f"Test Iterations: {test_iterations}")

            # Validáció
            if not lottery_type_id:
                logger.error("VALIDATION ERROR: lottery_type_id kotelzo")
                return JsonResponse({"error": "Missing lottery_type_id"}, status=400)
            if not winning_numbers or not isinstance(winning_numbers, list):
                logger.error("VALIDATION ERROR: winning_numbers kotelzo es listanak kell lennie")
                return JsonResponse({"error": "Invalid winning_numbers format"}, status=400)

            lottery_type = lg_lottery_type.objects.select_related().filter(id=lottery_type_id).first()
            if not lottery_type:
                logger.error(f"LOTTERY TYPE NOT FOUND: {lottery_type_id}")
                return JsonResponse({"error": "Lottery type not found"}, status=404)
            
            logger.info(f"Lottery Type Found: {lottery_type.lottery_type}")

            # Optimalizált algoritmus tesztelés
            start_time = timezone.now()
            test_results = []
            
            for iteration in range(test_iterations):
                algorithms = self.evaluate_algorithms(lottery_type, winning_numbers, additional_numbers)
                test_results.append(algorithms)

            # Eredmények aggregálása
            aggregated_results = self.aggregate_test_results(test_results)
            
            end_time = timezone.now()
            execution_duration = (end_time - start_time).total_seconds()
            
            # Teszt befejezése
            logger.info("="*60)
            logger.info("LOTTERY ALGORITHM TESTING COMPLETED")
            logger.info(f"Total algorithms tested: {len(aggregated_results)}")
            logger.info(f"Total execution time: {execution_duration:.2f} seconds")
            logger.info("="*60)

            return JsonResponse({
                "status": "success",
                "test_summary": {
                    "lottery_type_id": lottery_type_id,
                    "winning_numbers": winning_numbers,
                    "additional_numbers": additional_numbers,
                    "test_iterations": test_iterations,
                    "execution_time_seconds": execution_duration,
                    "timestamp": end_time.isoformat()
                },
                "ranked_algorithms": aggregated_results,
                "performance_metrics": {
                    "total_algorithms_tested": len(aggregated_results),
                    "average_score": sum(a['average_score'] for a in aggregated_results) / len(aggregated_results) if aggregated_results else 0,
                    "best_performer": aggregated_results[0]['name'] if aggregated_results else None
                }
            })

        except json.JSONDecodeError as e:
            logger.error(f"JSON DECODE ERROR: {str(e)}")
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            logger.error(f"GENERAL ERROR in algorithm testing: {str(e)}")
            logger.error("GENERAL ERROR TRACEBACK:")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("="*60)
            logger.info("LOTTERY ALGORITHM TESTING FAILED")
            logger.info("="*60)
            return JsonResponse({"error": f"Internal server error: {str(e)}"}, status=500)

    def evaluate_algorithms(self, lottery_type, winning_numbers, additional_numbers=None):
        algorithms = []
        processors_dir = "processors"
        
        # Optimalizált fájllista
        try:
            processor_files = sorted([
                f[:-3] for f in os.listdir(processors_dir)
                if f.endswith(".py") and not f.startswith("__")
            ])
        except OSError as e:
            logger.error(f"Nem sikerült betölteni a processors könyvtárat: {e}")
            return []

        for module_name in processor_files:
            try:
                logger.info(f"START: {module_name}")
                
                # Dinamikus import optimalizálva
                module = importlib.import_module(f"processors.{module_name}")

                if hasattr(module, 'get_numbers'):
                    # Teljesítménymérés kezdete
                    start_time = time.time()

                    # Predikció végrehajtása
                    prediction_result = module.get_numbers(lottery_type)
                    
                    # Tuple format kezelése
                    if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                        predicted_numbers, predicted_additional = prediction_result
                    else:
                        predicted_numbers = prediction_result
                        predicted_additional = []

                    # Végrehajtási idő mérése
                    execution_time = (time.time() - start_time) * 1000

                    # Pontszám számítása (fő és kiegészítő számokat is figyelembe véve)
                    score = self.calculate_enhanced_score(
                        predicted_numbers, predicted_additional,
                        winning_numbers, additional_numbers or []
                    )

                    # Batch adatbázis műveletek - teljesítményoptimalizálás
                    try:
                        # Predikciótörténet mentése
                        lg_prediction_history.objects.create(
                            algorithm_name=module_name,
                            predicted_numbers=predicted_numbers,
                            actual_numbers=winning_numbers,
                            score=score
                        )

                        # Teljesítménytörténet mentése
                        lg_performance_history.objects.create(
                            algorithm_name=module_name,
                            execution_time=execution_time,
                            success=True
                        )

                        # Teljesítménymetrikák frissítése (optimalizált)
                        self.update_performance_metrics(module_name, execution_time)

                        # Generált számok mentése
                        self.save_generated_lottery_draw(lottery_type, predicted_numbers, module_name)

                        # Algoritmus pontszám frissítése
                        self.update_algorithm_score(module_name)

                        # Jelenlegi pontszám lekérése
                        current_score = lg_algorithm_score.objects.filter(
                            algorithm_name=module_name
                        ).values_list('current_score', flat=True).first() or 0

                        algorithms.append({
                            "name": module_name,
                            "score": current_score,
                            "execution_time": execution_time,
                            "predicted_main": predicted_numbers,
                            "predicted_additional": predicted_additional,
                            "success": True
                        })
                        
                        # Részletes logging
                        from .utils import log_algorithm_performance
                        log_algorithm_performance(
                            module_name, current_score, execution_time, 
                            predicted_numbers, predicted_additional
                        )
                        logger.info(f"SUCCESS: {module_name}")

                    except Exception as db_error:
                        logger.error(f"DATABASE ERROR in {module_name}: {str(db_error)}")
                        logger.error(f"DATABASE ERROR TRACEBACK:")
                        import traceback
                        logger.error(traceback.format_exc())
                        # Folytatás hibánál is
                        algorithms.append({
                            "name": module_name,
                            "score": 0,
                            "execution_time": execution_time if 'execution_time' in locals() else 0,
                            "success": False,
                            "error": str(db_error)
                        })

            except Exception as e:
                logger.error(f"ALGORITHM ERROR in {module_name}: {str(e)}")
                logger.error(f"ALGORITHM ERROR TRACEBACK:")
                import traceback
                logger.error(traceback.format_exc())
                logger.info(f"FAILED: {module_name}")
                # Hiba naplózása
                try:
                    lg_performance_history.objects.create(
                        algorithm_name=module_name,
                        execution_time=0,
                        success=False,
                        error_message=str(e)
                    )
                except:
                    pass  # Ha az adatbázis írás is hibázik

                algorithms.append({
                    "name": module_name,
                    "score": 0,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e)
                })

        # Rendezés pontszám szerint
        ranked_algorithms = sorted(algorithms, key=lambda x: x.get('score', 0), reverse=True)
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

    def calculate_enhanced_score(self, predicted_main, predicted_additional, winning_main, winning_additional):
        """Fejlett pontszám számítás fő és kiegészítő számokkal."""
        
        if not predicted_main:
            return 0
        
        # Fő számok pontszáma
        main_matches = len(set(predicted_main) & set(winning_main))
        main_score = main_matches * 10  # Alap pont
        
        # Pozíciós bónusz a fő számoknál
        position_bonus = 0
        for i, num in enumerate(predicted_main):
            if i < len(winning_main) and num == winning_main[i]:
                position_bonus += 5  # Helyes pozíció bónusz
        
        # Kiegészítő számok pontszáma
        additional_score = 0
        if predicted_additional and winning_additional:
            additional_matches = len(set(predicted_additional) & set(winning_additional))
            additional_score = additional_matches * 8  # Kiegészítő számok értéke
        
        # Kombinációs bónusz
        combination_bonus = 0
        if main_matches >= 3 and len(set(predicted_additional) & set(winning_additional)) >= 1:
            combination_bonus = 15  # Kombinációs bónusz
        
        # Teljes pontszám
        total_score = main_score + position_bonus + additional_score + combination_bonus
        
        return total_score
    
    def update_performance_metrics(self, algorithm_name, execution_time):
        """Optimalizált teljesítménymetrika frissítés."""
        
        performance, created = lg_algorithm_performance.objects.get_or_create(
            algorithm_name=algorithm_name,
            defaults={
                'average_execution_time': execution_time,
                'fastest_execution': execution_time,
                'slowest_execution': execution_time,
                'total_executions': 1,
                'last_execution_time': execution_time
            }
        )

        if not created:
            # Optimalizált frissítés bulk művelettel
            performance.total_executions += 1
            
            # Mozgó átlag számítás (teljesítményoptimalizálás)
            alpha = 0.1  # Súlyozási faktor az új értéknek
            performance.average_execution_time = (
                performance.average_execution_time * (1 - alpha) + 
                execution_time * alpha
            )
            
            # Min/max frissítése
            performance.fastest_execution = min(performance.fastest_execution, execution_time)
            performance.slowest_execution = max(performance.slowest_execution, execution_time)
            performance.last_execution_time = execution_time
            
            performance.save(update_fields=[
                'total_executions', 'average_execution_time', 
                'fastest_execution', 'slowest_execution', 'last_execution_time'
            ])
    
    def aggregate_test_results(self, test_results):
        """Teszt eredmények aggregálása több iterációból."""
        
        algorithm_stats = {}
        
        # Összes eredmény összegyűjtése
        for iteration_results in test_results:
            for result in iteration_results:
                alg_name = result['name']
                
                if alg_name not in algorithm_stats:
                    algorithm_stats[alg_name] = {
                        'scores': [],
                        'execution_times': [],
                        'success_count': 0,
                        'error_count': 0,
                        'errors': []
                    }
                
                if result.get('success', True):
                    algorithm_stats[alg_name]['scores'].append(result.get('score', 0))
                    algorithm_stats[alg_name]['execution_times'].append(result.get('execution_time', 0))
                    algorithm_stats[alg_name]['success_count'] += 1
                else:
                    algorithm_stats[alg_name]['error_count'] += 1
                    algorithm_stats[alg_name]['errors'].append(result.get('error', 'Unknown error'))
        
        # Aggregált statisztikák számítása
        aggregated = []
        for alg_name, stats in algorithm_stats.items():
            scores = stats['scores']
            exec_times = stats['execution_times']
            
            if scores:  # Van legalább egy sikeres futtatás
                aggregated.append({
                    'name': alg_name,
                    'average_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'score_std': np.std(scores) if len(scores) > 1 else 0,
                    'average_execution_time': sum(exec_times) / len(exec_times),
                    'min_execution_time': min(exec_times),
                    'max_execution_time': max(exec_times),
                    'success_rate': (stats['success_count'] / (stats['success_count'] + stats['error_count'])) * 100,
                    'total_runs': stats['success_count'] + stats['error_count'],
                    'successful_runs': stats['success_count'],
                    'failed_runs': stats['error_count'],
                    'consistency_rating': self.calculate_consistency_rating(scores, exec_times)
                })
            else:  # Csak hibás futtatások
                aggregated.append({
                    'name': alg_name,
                    'average_score': 0,
                    'success_rate': 0,
                    'total_runs': stats['error_count'],
                    'failed_runs': stats['error_count'],
                    'errors': stats['errors'][:3]  # Első 3 hiba
                })
        
        # Rendezés átlagos pontszám szerint
        return sorted(aggregated, key=lambda x: x.get('average_score', 0), reverse=True)
    
    def calculate_consistency_rating(self, scores, execution_times):
        """Konzisztencia értékelés számítása."""
        
        if len(scores) < 2:
            return 100.0  # Egyetlen futtatás = teljes konzisztencia
        
        # Pontszám konzisztencia (alacsony szórás = jó)
        score_cv = (np.std(scores) / np.mean(scores)) * 100 if np.mean(scores) > 0 else 100
        score_consistency = max(0, 100 - score_cv)
        
        # Végrehajtási idő konzisztencia
        time_cv = (np.std(execution_times) / np.mean(execution_times)) * 100 if np.mean(execution_times) > 0 else 100
        time_consistency = max(0, 100 - time_cv)
        
        # Kombinált konzisztencia (pontszám fontosabb)
        overall_consistency = score_consistency * 0.7 + time_consistency * 0.3
        
        return round(overall_consistency, 2)


class DetailedAlgorithmStatisticsView(APIView):
    """
    Fejlett statisztika végpont az algoritmusok teljesítményének részletes elemzésére
    """
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Részletes Algoritmus Statisztikák",
        description="Részletes teljesítmény statisztikák az összes algoritmusról",
        parameters=[GetStatisticsQuerySerializer],
        responses={200: DetailedStatisticsSerializer}
    )
    def get(self, request):
        """
        Részletes statisztikák lekérése az algoritmusok teljesítményéről
        """
        try:
            # Query paraméterek validálása
            query_serializer = GetStatisticsQuerySerializer(data=request.GET)
            query_serializer.is_valid(raise_exception=True)
            
            params = query_serializer.validated_data
            days = params.get('days', 30)
            algorithm_filter = params.get('algorithm_filter')
            include_trends = params.get('include_trends', True)
            lottery_type_id = params.get('lottery_type_id')

            # Dátum szűrő
            since_date = timezone.now() - timedelta(days=days)

            # Algoritmus teljesítmény adatok összegyűjtése
            algorithm_performances = self.get_algorithm_performances(
                since_date, algorithm_filter, lottery_type_id, include_trends
            )

            # Összefoglaló statisztikák
            summary_stats = self.calculate_summary_statistics(algorithm_performances)

            # Válasz összeállítása
            response_data = {
                **summary_stats,
                'algorithm_performance': algorithm_performances
            }

            # Serializer validáció
            response_serializer = DetailedStatisticsSerializer(data=response_data)
            response_serializer.is_valid(raise_exception=True)

            return JsonResponse(response_serializer.validated_data, safe=False)

        except Exception as e:
            return JsonResponse({
                "error": f"Hiba a statisztikák lekérése során: {str(e)}"
            }, status=500)

    def get_algorithm_performances(self, since_date, algorithm_filter, lottery_type_id, include_trends):
        """
        Algoritmus teljesítmény adatok lekérése és elemzése
        """
        performances = []
        
        # Algoritmus pontszámok lekérése
        scores_query = lg_algorithm_score.objects.all()
        if algorithm_filter:
            scores_query = scores_query.filter(algorithm_name__icontains=algorithm_filter)

        for score in scores_query:
            algorithm_name = score.algorithm_name
            
            # Teljesítmény adatok
            performance_data = lg_algorithm_performance.objects.filter(
                algorithm_name=algorithm_name
            ).first()

            # Predikció történet (trend számításhoz)
            predictions = lg_prediction_history.objects.filter(
                algorithm_name=algorithm_name,
                prediction_date__gte=since_date
            ).order_by('prediction_date')

            # Végrehajtási történet
            executions = lg_performance_history.objects.filter(
                algorithm_name=algorithm_name,
                execution_date__gte=since_date
            )

            # Teljesítmény metrikák számítása
            perf_metrics = self.calculate_performance_metrics(
                score, performance_data, predictions, executions, include_trends
            )

            performances.append(perf_metrics)

        # Rangsorolás
        performances = self.rank_algorithms(performances)

        return performances

    def calculate_performance_metrics(self, score, performance_data, predictions, executions, include_trends):
        """
        Egy algoritmus teljesítmény metrikáinak számítása
        """
        algorithm_name = score.algorithm_name
        
        # Alapmetrikák
        current_score = score.current_score
        total_predictions = score.total_predictions
        
        # Végrehajtási statisztikák
        execution_stats = {
            'average_execution_time': performance_data.average_execution_time if performance_data else 0.0,
            'fastest_execution': performance_data.fastest_execution if performance_data else 0.0,
            'slowest_execution': performance_data.slowest_execution if performance_data else 0.0,
            'total_executions': performance_data.total_executions if performance_data else 0,
            'last_execution_time': performance_data.last_execution_time if performance_data else 0.0,
        }

        # Sikeres végrehajtások aránya
        total_exec_count = executions.count()
        successful_exec_count = executions.filter(success=True).count()
        success_rate = (successful_exec_count / total_exec_count * 100) if total_exec_count > 0 else 0.0

        # Trend elemzés
        score_trend = []
        recent_scores = []
        improvement_trend = "stabil"
        
        if include_trends and predictions.exists():
            # Utolsó 10 predikció pontszámai
            recent_predictions = predictions.order_by('-prediction_date')[:10]
            recent_scores = [pred.score for pred in recent_predictions]
            
            # Score trend (utolsó 20 predikció 5-ös csoportokban)
            if predictions.count() >= 10:
                score_trend = self.calculate_score_trend(predictions)
                improvement_trend = self.determine_improvement_trend(score_trend)

        # Konzisztencia számítás (score szórás alapján)
        consistency_score = 0.0
        if recent_scores:
            score_std = np.std(recent_scores) if len(recent_scores) > 1 else 0
            # Minél kisebb a szórás, annál konzisztensebb (0-100 skála)
            max_possible_std = max(recent_scores) - min(recent_scores) if len(recent_scores) > 1 else 1
            consistency_score = max(0, 100 - (score_std / max_possible_std * 100)) if max_possible_std > 0 else 100

        return {
            'algorithm_name': algorithm_name,
            'current_score': current_score,
            'total_predictions': total_predictions,
            'success_rate': success_rate,
            'score_trend': score_trend,
            'recent_scores': recent_scores,
            'consistency_score': consistency_score,
            'improvement_trend': improvement_trend,
            'last_updated': score.last_updated,
            'performance_rank': 0,  # Will be set in ranking
            'speed_rank': 0,       # Will be set in ranking
            **execution_stats
        }

    def calculate_score_trend(self, predictions):
        """
        Pontszám trend számítása 5-ös csoportokban
        """
        scores = [pred.score for pred in predictions.order_by('prediction_date')]
        
        # 5-ös csoportok átlaga
        trend = []
        for i in range(0, len(scores), 5):
            group = scores[i:i+5]
            if group:
                trend.append(sum(group) / len(group))
        
        return trend[-10:]  # Utolsó 10 csoport

    def determine_improvement_trend(self, score_trend):
        """
        Javulási trend meghatározása
        """
        if len(score_trend) < 3:
            return "nem_elegendo_adat"
        
        # Linear regression slope
        x = np.arange(len(score_trend))
        y = np.array(score_trend)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.5:
                return "javulo"
            elif slope < -0.5:
                return "romlo"
            else:
                return "stabil"
        except:
            return "stabil"

    def rank_algorithms(self, performances):
        """
        Algoritmusok rangsorolása különböző metrikák szerint
        """
        # Teljesítmény rangsor
        performances.sort(key=lambda x: x['current_score'], reverse=True)
        for i, perf in enumerate(performances):
            perf['performance_rank'] = i + 1

        # Sebesség rangsor
        speed_sorted = sorted(performances, key=lambda x: x['average_execution_time'])
        for i, perf in enumerate(speed_sorted):
            perf['speed_rank'] = i + 1

        return performances

    def calculate_summary_statistics(self, algorithm_performances):
        """
        Összefoglaló statisztikák számítása
        """
        if not algorithm_performances:
            return {
                'total_algorithms': 0,
                'total_predictions': 0,
                'total_executions': 0,
                'average_score_all': 0.0,
                'top_3_algorithms': [],
                'fastest_algorithms': [],
                'most_consistent_algorithms': [],
                'performance_distribution': {},
                'execution_time_distribution': {},
                'improvement_trends': {}
            }

        # Alapstatisztikák
        total_algorithms = len(algorithm_performances)
        total_predictions = sum(p['total_predictions'] for p in algorithm_performances)
        total_executions = sum(p['total_executions'] for p in algorithm_performances)
        
        scores = [p['current_score'] for p in algorithm_performances if p['current_score'] > 0]
        average_score_all = sum(scores) / len(scores) if scores else 0.0

        # Top 3 teljesítmény szerint
        top_3_algorithms = [p['algorithm_name'] for p in 
                          sorted(algorithm_performances, key=lambda x: x['current_score'], reverse=True)[:3]]

        # Top 3 leggyorsabb
        fastest_algorithms = [p['algorithm_name'] for p in 
                            sorted(algorithm_performances, key=lambda x: x['average_execution_time'])[:3]
                            if p['average_execution_time'] > 0]

        # Top 3 legkozisztensebb
        most_consistent_algorithms = [p['algorithm_name'] for p in 
                                    sorted(algorithm_performances, key=lambda x: x['consistency_score'], reverse=True)[:3]]

        # Teljesítmény eloszlás
        performance_distribution = self.calculate_performance_distribution(algorithm_performances)
        
        # Végrehajtási idő eloszlás
        execution_time_distribution = self.calculate_execution_time_distribution(algorithm_performances)
        
        # Javulási trendek összesítése
        improvement_trends = self.calculate_improvement_trends_summary(algorithm_performances)

        return {
            'total_algorithms': total_algorithms,
            'total_predictions': total_predictions,
            'total_executions': total_executions,
            'average_score_all': average_score_all,
            'top_3_algorithms': top_3_algorithms,
            'fastest_algorithms': fastest_algorithms,
            'most_consistent_algorithms': most_consistent_algorithms,
            'performance_distribution': performance_distribution,
            'execution_time_distribution': execution_time_distribution,
            'improvement_trends': improvement_trends
        }

    def calculate_performance_distribution(self, algorithm_performances):
        """
        Teljesítmény eloszlás számítása kategóriákba
        """
        distribution = {'kiváló': 0, 'jó': 0, 'közepes': 0, 'gyenge': 0}
        
        for perf in algorithm_performances:
            score = perf['current_score']
            if score >= 30:
                distribution['kiváló'] += 1
            elif score >= 20:
                distribution['jó'] += 1
            elif score >= 10:
                distribution['közepes'] += 1
            else:
                distribution['gyenge'] += 1
        
        return distribution

    def calculate_execution_time_distribution(self, algorithm_performances):
        """
        Végrehajtási idő eloszlás számítása
        """
        distribution = {'gyors': 0, 'közepes': 0, 'lassú': 0}
        
        for perf in algorithm_performances:
            exec_time = perf['average_execution_time']
            if exec_time <= 100:  # 100ms alatt
                distribution['gyors'] += 1
            elif exec_time <= 1000:  # 1s alatt
                distribution['közepes'] += 1
            else:
                distribution['lassú'] += 1
        
        return distribution

    def calculate_improvement_trends_summary(self, algorithm_performances):
        """
        Javulási trendek összesítése
        """
        trends = {'javulo': 0, 'stabil': 0, 'romlo': 0, 'nem_elegendo_adat': 0}
        
        for perf in algorithm_performances:
            trend = perf['improvement_trend']
            if trend in trends:
                trends[trend] += 1
        
        return trends


class QuickStatisticsView(APIView):
    """
    Gyors statisztika végpont dashboard használatra
    """
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Gyors Algoritmus Rangsor",
        description="Gyors áttekintés a legjobb algoritmusokról",
        parameters=[GetStatisticsQuerySerializer],
        responses={200: QuickStatisticsSerializer}
    )
    def get(self, request):
        """
        Gyors statisztikák lekérése dashboard használatra
        """
        try:
            # Query paraméterek
            days = int(request.GET.get('days', 30))
            algorithm_filter = request.GET.get('algorithm_filter')
            
            # Algoritmus pontszámok lekérése
            scores_query = lg_algorithm_score.objects.all().order_by('-current_score')
            if algorithm_filter:
                scores_query = scores_query.filter(algorithm_name__icontains=algorithm_filter)

            top_algorithms = []
            
            for i, score in enumerate(scores_query[:20]):  # Top 20
                algorithm_name = score.algorithm_name
                
                # Teljesítmény adatok
                performance_data = lg_algorithm_performance.objects.filter(
                    algorithm_name=algorithm_name
                ).first()
                
                # Sikeres végrehajtások
                since_date = timezone.now() - timedelta(days=days)
                executions = lg_performance_history.objects.filter(
                    algorithm_name=algorithm_name,
                    execution_date__gte=since_date
                )
                
                total_exec = executions.count()
                successful_exec = executions.filter(success=True).count()
                success_rate = (successful_exec / total_exec * 100) if total_exec > 0 else 0.0
                
                # Kategorizálás
                performance_category = self.categorize_performance(score.current_score)
                
                # Trend
                trend = self.get_quick_trend(algorithm_name, since_date)
                
                top_algorithms.append({
                    'algorithm_name': algorithm_name,
                    'current_score': score.current_score,
                    'rank': i + 1,
                    'performance_category': performance_category,
                    'average_execution_time': performance_data.average_execution_time if performance_data else 0.0,
                    'success_rate': success_rate,
                    'trend': trend
                })

            # Összefoglaló
            summary = self.calculate_quick_summary(top_algorithms)
            
            response_data = {
                'top_algorithms': top_algorithms,
                'summary': summary,
                'last_updated': timezone.now()
            }

            response_serializer = QuickStatisticsSerializer(data=response_data)
            response_serializer.is_valid(raise_exception=True)

            return JsonResponse(response_serializer.validated_data, safe=False)

        except Exception as e:
            return JsonResponse({
                "error": f"Hiba a gyors statisztikák lekérése során: {str(e)}"
            }, status=500)

    def categorize_performance(self, score):
        """Teljesítmény kategorizálása"""
        if score >= 30:
            return "kiváló"
        elif score >= 20:
            return "jó"
        elif score >= 10:
            return "közepes"
        else:
            return "gyenge"

    def get_quick_trend(self, algorithm_name, since_date):
        """Gyors trend meghatározás"""
        predictions = lg_prediction_history.objects.filter(
            algorithm_name=algorithm_name,
            prediction_date__gte=since_date
        ).order_by('prediction_date')

        if predictions.count() < 5:
            return "nem_elegendo_adat"

        scores = [pred.score for pred in predictions]
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]

        if not first_half or not second_half:
            return "stabil"

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_second > avg_first + 1:
            return "javulo"
        elif avg_second < avg_first - 1:
            return "romlo"
        else:
            return "stabil"

    def calculate_quick_summary(self, top_algorithms):
        """Gyors összefoglaló számítás"""
        if not top_algorithms:
            return {
                'total_algorithms': 0,
                'best_algorithm': None,
                'average_score_top10': 0.0,
                'fastest_algorithm': None,
                'most_reliable': None
            }

        # Legjobb algoritmus
        best_algorithm = top_algorithms[0]['algorithm_name'] if top_algorithms else None
        
        # Top 10 átlagpontszám
        top10_scores = [algo['current_score'] for algo in top_algorithms[:10]]
        avg_score_top10 = sum(top10_scores) / len(top10_scores) if top10_scores else 0.0
        
        # Leggyorsabb
        fastest_algo = min(top_algorithms, 
                          key=lambda x: x['average_execution_time'] if x['average_execution_time'] > 0 else float('inf'),
                          default={'algorithm_name': None})
        
        # Legmegbízhatóbb (legmagasabb success_rate)
        most_reliable = max(top_algorithms,
                           key=lambda x: x['success_rate'],
                           default={'algorithm_name': None})

        return {
            'total_algorithms': len(top_algorithms),
            'best_algorithm': best_algorithm,
            'average_score_top10': round(avg_score_top10, 2),
            'fastest_algorithm': fastest_algo['algorithm_name'],
            'most_reliable': most_reliable['algorithm_name'],
            'excellent_count': len([a for a in top_algorithms if a['performance_category'] == 'kiváló']),
            'good_count': len([a for a in top_algorithms if a['performance_category'] == 'jó']),
            'improving_count': len([a for a in top_algorithms if a['trend'] == 'javulo']),
            'declining_count': len([a for a in top_algorithms if a['trend'] == 'romlo'])
        }


class RealTimeMonitoringView(APIView):
    """
    Real-time performance monitoring dashboard
    """
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Real-time Performance Monitoring",
        description="Get real-time performance data and alerts for all algorithms",
        responses={200: "Real-time monitoring data"}
    )
    def get(self, request):
        """Get real-time monitoring overview"""
        try:
            # Get system overview
            system_overview = performance_monitor.get_system_overview()
            
            # Get alerts
            alerts = performance_monitor.get_performance_alerts()
            
            # Get database optimization report
            optimization_report = DatabasePerformanceAnalyzer.get_optimization_report()
            
            return JsonResponse({
                'status': 'success',
                'system_overview': system_overview,
                'alerts': [
                    {
                        'type': alert['type'],
                        'algorithm': alert['algorithm'],
                        'message': alert['message'],
                        'severity': alert['severity'],
                        'timestamp': alert['timestamp'].isoformat()
                    } for alert in alerts
                ],
                'optimization_summary': optimization_report['summary'],
                'timestamp': timezone.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


class AlgorithmMonitoringView(APIView):
    """
    Individual algorithm monitoring
    """
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Algorithm Specific Monitoring",
        description="Get detailed monitoring data for a specific algorithm",
        responses={200: "Algorithm monitoring data"}
    )
    def get(self, request, algorithm_name):
        """Get detailed monitoring data for specific algorithm"""
        try:
            # Get real-time stats
            real_time_stats = performance_monitor.get_real_time_stats(algorithm_name)
            
            # Get database trends
            trends = DatabasePerformanceAnalyzer.get_algorithm_trends(days=7)
            algorithm_trends = trends.get(algorithm_name, {})
            
            # Export recent performance data
            performance_export = performance_monitor.export_performance_data(
                algorithm_name=algorithm_name, 
                hours=24
            )
            
            return JsonResponse({
                'status': 'success',
                'algorithm_name': algorithm_name,
                'real_time_stats': real_time_stats,
                'database_trends': algorithm_trends,
                'recent_performance': performance_export['data'].get(algorithm_name, {}),
                'timestamp': timezone.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


class OptimizationReportView(APIView):
    """
    Comprehensive optimization report
    """
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Optimization Report",
        description="Get comprehensive optimization report showing performance improvements",
        responses={200: "Optimization report data"}
    )
    def get(self, request):
        """Get comprehensive optimization report"""
        try:
            # Get full optimization report
            optimization_report = DatabasePerformanceAnalyzer.get_optimization_report()
            
            # Get algorithm trends for context
            recent_trends = DatabasePerformanceAnalyzer.get_algorithm_trends(days=7)
            older_trends = DatabasePerformanceAnalyzer.get_algorithm_trends(days=30)
            
            # Performance categories
            performance_categories = {
                'ultra_fast': [],     # < 1ms
                'very_fast': [],      # 1-10ms
                'fast': [],           # 10-100ms
                'moderate': [],       # 100-1000ms
                'slow': [],           # 1000-5000ms
                'very_slow': []       # > 5000ms
            }
            
            for alg_name, data in recent_trends.items():
                avg_time = data.get('performance', {}).get('avg_execution_time', 0)
                
                if avg_time < 1:
                    performance_categories['ultra_fast'].append(alg_name)
                elif avg_time < 10:
                    performance_categories['very_fast'].append(alg_name)
                elif avg_time < 100:
                    performance_categories['fast'].append(alg_name)
                elif avg_time < 1000:
                    performance_categories['moderate'].append(alg_name)
                elif avg_time < 5000:
                    performance_categories['slow'].append(alg_name)
                else:
                    performance_categories['very_slow'].append(alg_name)
            
            return JsonResponse({
                'status': 'success',
                'optimization_report': optimization_report,
                'performance_categories': performance_categories,
                'total_algorithms_analyzed': len(recent_trends),
                'timestamp': timezone.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


class AutomaticAlgorithmTestView(APIView):
    """
    Automatic algorithm testing endpoint - runs all algorithms 10 times 
    and saves top 5 to file
    """
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Automatic Algorithm Testing",
        description="Run all algorithms 10 times and save top 5 performers to file",
        responses={200: "Algorithm test results"}
    )
    def get(self, request):
        """Run automatic algorithm testing"""
        try:
            # EuroJackpot winning numbers (latest draw)
            main_numbers = [4, 14, 26, 29, 50]
            additional_numbers = [3, 12]
            
            # Get EuroJackpot lottery type
            eurojackpot = lg_lottery_type.objects.filter(lottery_type__icontains='euro').first()
            if not eurojackpot:
                return JsonResponse({
                    'status': 'error',
                    'message': 'EuroJackpot lottery type not found'
                }, status=404)
            
            print(f"🎯 Starting automatic algorithm test with EuroJackpot numbers:")
            print(f"   Main: {main_numbers}")
            print(f"   Additional: {additional_numbers}")
            
            # Run algorithms 10 times
            all_results = []
            processors_dir = "processors"
            
            for iteration in range(1, 11):
                print(f"\n[ITER] Iteration {iteration}/10")
                iteration_results = []
                
                for filename in os.listdir(processors_dir):
                    if filename.endswith(".py") and not filename.startswith("__"):
                        module_name = filename[:-3]
                        
                        try:
                            # Import and execute algorithm
                            module = importlib.import_module(f"processors.{module_name}")
                            
                            if hasattr(module, 'get_numbers'):
                                # Measure execution time
                                start_time = time.time()
                                predicted_main, predicted_additional = module.get_numbers(eurojackpot)
                                execution_time = (time.time() - start_time) * 1000
                                
                                # Calculate score
                                score = self.calculate_eurojackpot_score(
                                    predicted_main, predicted_additional,
                                    main_numbers, additional_numbers
                                )
                                
                                iteration_results.append({
                                    'algorithm': module_name,
                                    'iteration': iteration,
                                    'score': score,
                                    'execution_time': execution_time,
                                    'predicted_main': predicted_main,
                                    'predicted_additional': predicted_additional
                                })
                                
                                print(f"   [OK] {module_name}: {score} points ({execution_time:.2f}ms)")
                                
                        except Exception as e:
                            print(f"   ❌ {module_name}: Error - {str(e)}")
                            iteration_results.append({
                                'algorithm': module_name,
                                'iteration': iteration,
                                'score': 0,
                                'execution_time': 0,
                                'error': str(e)
                            })
                
                all_results.extend(iteration_results)
            
            # Calculate average scores per algorithm
            algorithm_averages = {}
            algorithm_details = {}
            
            for result in all_results:
                alg_name = result['algorithm']
                if alg_name not in algorithm_averages:
                    algorithm_averages[alg_name] = []
                    algorithm_details[alg_name] = {
                        'scores': [],
                        'execution_times': [],
                        'errors': [],
                        'iterations_run': 0
                    }
                
                if 'error' not in result:
                    algorithm_averages[alg_name].append(result['score'])
                    algorithm_details[alg_name]['scores'].append(result['score'])
                    algorithm_details[alg_name]['execution_times'].append(result['execution_time'])
                    algorithm_details[alg_name]['iterations_run'] += 1
                else:
                    algorithm_details[alg_name]['errors'].append(result['error'])
            
            # Calculate final rankings
            final_rankings = []
            for alg_name, scores in algorithm_averages.items():
                if scores:  # Only algorithms with successful runs
                    details = algorithm_details[alg_name]
                    avg_score = sum(scores) / len(scores)
                    avg_execution_time = sum(details['execution_times']) / len(details['execution_times'])
                    
                    final_rankings.append({
                        'algorithm': alg_name,
                        'average_score': avg_score,
                        'max_score': max(scores),
                        'min_score': min(scores),
                        'average_execution_time': avg_execution_time,
                        'successful_iterations': len(scores),
                        'failed_iterations': len(details['errors']),
                        'error_count': len(details['errors'])
                    })
            
            # Sort by average score (descending)
            final_rankings.sort(key=lambda x: x['average_score'], reverse=True)
            
            # Get top 5
            top_5 = final_rankings[:5]
            
            # Save to file
            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            filename = f"top_5_algorithms_{timestamp}.json"
            
            file_data = {
                'test_timestamp': timezone.now().isoformat(),
                'eurojackpot_numbers': {
                    'main': main_numbers,
                    'additional': additional_numbers
                },
                'test_parameters': {
                    'iterations_per_algorithm': 10,
                    'lottery_type': 'EuroJackpot',
                    'total_algorithms_tested': len(algorithm_averages)
                },
                'top_5_algorithms': top_5,
                'all_rankings': final_rankings
            }
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(file_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n🏆 TOP 5 ALGORITHMS SAVED TO: {filename}")
            for i, alg in enumerate(top_5, 1):
                print(f"   {i}. {alg['algorithm']}: {alg['average_score']:.2f} avg score ({alg['average_execution_time']:.2f}ms)")
            
            return JsonResponse({
                'status': 'success',
                'message': f'Algorithm testing complete. Results saved to {filename}',
                'results_file': filename,
                'test_summary': {
                    'total_algorithms_tested': len(algorithm_averages),
                    'successful_algorithms': len([a for a in final_rankings if a['successful_iterations'] > 0]),
                    'iterations_per_algorithm': 10,
                    'eurojackpot_numbers': {
                        'main': main_numbers,
                        'additional': additional_numbers
                    }
                },
                'top_5_algorithms': top_5,
                'timestamp': timezone.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error during algorithm testing: {str(e)}'
            }, status=500)
    
    def calculate_eurojackpot_score(self, predicted_main, predicted_additional, winning_main, winning_additional):
        """
        Calculate EuroJackpot score based on matches
        """
        if not predicted_main or not predicted_additional:
            return 0
        
        # Main numbers matches
        main_matches = len(set(predicted_main) & set(winning_main))
        
        # Additional numbers matches  
        additional_matches = len(set(predicted_additional) & set(winning_additional))
        
        # EuroJackpot scoring system (simplified)
        score = 0
        
        # Main number scoring (exponential for more matches)
        if main_matches == 5:
            score += 100  # Jackpot!
        elif main_matches == 4:
            score += 50
        elif main_matches == 3:
            score += 20
        elif main_matches == 2:
            score += 10
        elif main_matches == 1:
            score += 2
        
        # Additional number scoring
        if additional_matches == 2:
            score += 30
        elif additional_matches == 1:
            score += 10
        
        # Bonus for combination matches (realistic EuroJackpot prizes)
        if main_matches == 5 and additional_matches == 2:
            score += 200  # Jackpot + bonus
        elif main_matches == 5 and additional_matches == 1:
            score += 100
        elif main_matches == 4 and additional_matches == 2:
            score += 75
        elif main_matches == 4 and additional_matches == 1:
            score += 40
        elif main_matches == 3 and additional_matches == 2:
            score += 35
        
        return score


