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


