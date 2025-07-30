"""
Real-time Performance Monitoring Module
Provides comprehensive monitoring and analysis of algorithm performance
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import statistics
import json
from django.utils import timezone
from django.db.models import Avg, Count, Max, Min
from lottery_handler.models import lg_algorithm_score, lg_algorithm_performance, lg_prediction_history


class PerformanceMonitor:
    """
    Real-time performance monitoring system for lottery algorithms
    """
    
    def __init__(self, max_history_size=1000):
        self.max_history_size = max_history_size
        self.execution_history = defaultdict(lambda: deque(maxlen=max_history_size))
        self.score_history = defaultdict(lambda: deque(maxlen=max_history_size))
        self.alert_thresholds = {
            'slow_execution': 5000,  # ms
            'score_drop': 0.5,  # 50% score drop
            'error_rate': 0.1   # 10% error rate
        }
        self.lock = threading.Lock()
    
    def record_execution(self, algorithm_name: str, execution_time: float, 
                        success: bool = True, error_message: str = None):
        """Record algorithm execution performance"""
        with self.lock:
            timestamp = timezone.now()
            
            self.execution_history[algorithm_name].append({
                'timestamp': timestamp,
                'execution_time': execution_time,
                'success': success,
                'error_message': error_message
            })
    
    def record_score(self, algorithm_name: str, score: float):
        """Record algorithm score"""
        with self.lock:
            timestamp = timezone.now()
            
            self.score_history[algorithm_name].append({
                'timestamp': timestamp,
                'score': score
            })
    
    def get_real_time_stats(self, algorithm_name: str) -> Dict:
        """Get real-time statistics for an algorithm"""
        with self.lock:
            executions = list(self.execution_history[algorithm_name])
            scores = list(self.score_history[algorithm_name])
            
            if not executions:
                return {'status': 'no_data'}
            
            # Recent executions (last 10)
            recent_executions = executions[-10:]
            successful_executions = [e for e in recent_executions if e['success']]
            
            # Calculate metrics
            success_rate = len(successful_executions) / len(recent_executions) if recent_executions else 0
            
            if successful_executions:
                execution_times = [e['execution_time'] for e in successful_executions]
                avg_execution_time = statistics.mean(execution_times)
                min_execution_time = min(execution_times)
                max_execution_time = max(execution_times)
                execution_std = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            else:
                avg_execution_time = min_execution_time = max_execution_time = execution_std = 0
            
            # Score metrics
            recent_scores = scores[-10:] if scores else []
            if recent_scores:
                score_values = [s['score'] for s in recent_scores]
                avg_score = statistics.mean(score_values)
                score_trend = 'improving' if len(score_values) > 1 and score_values[-1] > score_values[0] else 'declining'
            else:
                avg_score = 0
                score_trend = 'unknown'
            
            return {
                'status': 'active',
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time,
                'min_execution_time': min_execution_time,
                'max_execution_time': max_execution_time,
                'execution_std': execution_std,
                'avg_score': avg_score,
                'score_trend': score_trend,
                'total_executions': len(executions),
                'recent_errors': [e['error_message'] for e in recent_executions if not e['success']]
            }
    
    def get_performance_alerts(self) -> List[Dict]:
        """Generate performance alerts based on thresholds"""
        alerts = []
        
        with self.lock:
            for algorithm_name in self.execution_history.keys():
                stats = self.get_real_time_stats(algorithm_name)
                
                if stats['status'] == 'no_data':
                    continue
                
                # Slow execution alert
                if stats['avg_execution_time'] > self.alert_thresholds['slow_execution']:
                    alerts.append({
                        'type': 'slow_execution',
                        'algorithm': algorithm_name,
                        'message': f"Algorithm is running slow: {stats['avg_execution_time']:.2f}ms",
                        'severity': 'warning',
                        'timestamp': timezone.now()
                    })
                
                # Low success rate alert
                if stats['success_rate'] < (1 - self.alert_thresholds['error_rate']):
                    alerts.append({
                        'type': 'high_error_rate',
                        'algorithm': algorithm_name,
                        'message': f"High error rate: {(1-stats['success_rate'])*100:.1f}%",
                        'severity': 'critical',
                        'timestamp': timezone.now()
                    })
                
                # Score declining alert
                if stats['score_trend'] == 'declining' and stats['avg_score'] > 0:
                    alerts.append({
                        'type': 'score_declining',
                        'algorithm': algorithm_name,
                        'message': f"Score trending downward: {stats['avg_score']:.2f}",
                        'severity': 'info',
                        'timestamp': timezone.now()
                    })
        
        return alerts
    
    def get_system_overview(self) -> Dict:
        """Get overall system performance overview"""
        with self.lock:
            total_algorithms = len(self.execution_history)
            active_algorithms = 0
            total_executions = 0
            total_errors = 0
            avg_system_performance = 0
            
            algorithm_stats = []
            
            for algorithm_name in self.execution_history.keys():
                stats = self.get_real_time_stats(algorithm_name)
                if stats['status'] == 'active':
                    active_algorithms += 1
                    total_executions += stats['total_executions']
                    total_errors += len(stats['recent_errors'])
                    avg_system_performance += stats['avg_execution_time']
                    
                    algorithm_stats.append({
                        'name': algorithm_name,
                        'performance': stats['avg_execution_time'],
                        'score': stats['avg_score'],
                        'success_rate': stats['success_rate']
                    })
            
            # Sort by performance (fastest first)
            algorithm_stats.sort(key=lambda x: x['performance'])
            
            return {
                'total_algorithms': total_algorithms,
                'active_algorithms': active_algorithms,
                'total_executions': total_executions,
                'system_error_rate': total_errors / total_executions if total_executions > 0 else 0,
                'avg_system_performance': avg_system_performance / active_algorithms if active_algorithms > 0 else 0,
                'top_performers': algorithm_stats[:5],
                'alerts_count': len(self.get_performance_alerts()),
                'timestamp': timezone.now()
            }
    
    def export_performance_data(self, algorithm_name: str = None, 
                               hours: int = 24) -> Dict:
        """Export performance data for analysis"""
        cutoff_time = timezone.now() - timedelta(hours=hours)
        
        with self.lock:
            if algorithm_name:
                algorithms_to_export = [algorithm_name]
            else:
                algorithms_to_export = list(self.execution_history.keys())
            
            export_data = {}
            
            for alg_name in algorithms_to_export:
                executions = [
                    e for e in self.execution_history[alg_name] 
                    if e['timestamp'] >= cutoff_time
                ]
                scores = [
                    s for s in self.score_history[alg_name] 
                    if s['timestamp'] >= cutoff_time
                ]
                
                export_data[alg_name] = {
                    'executions': [
                        {
                            'timestamp': e['timestamp'].isoformat(),
                            'execution_time': e['execution_time'],
                            'success': e['success'],
                            'error_message': e['error_message']
                        } for e in executions
                    ],
                    'scores': [
                        {
                            'timestamp': s['timestamp'].isoformat(),
                            'score': s['score']
                        } for s in scores
                    ]
                }
            
            return {
                'export_timestamp': timezone.now().isoformat(),
                'time_range_hours': hours,
                'data': export_data
            }


class DatabasePerformanceAnalyzer:
    """
    Analyzer for historical performance data from database
    """
    
    @staticmethod
    def get_algorithm_trends(days: int = 30) -> Dict:
        """Get performance trends from database"""
        cutoff_date = timezone.now() - timedelta(days=days)
        
        # Get performance data
        performance_data = lg_algorithm_performance.objects.filter(
            created_at__gte=cutoff_date
        ).values(
            'algorithm_name'
        ).annotate(
            avg_execution_time=Avg('average_execution_time'),
            total_executions=Count('id'),
            min_execution_time=Min('average_execution_time'),
            max_execution_time=Max('average_execution_time')
        )
        
        # Get score data
        score_data = lg_algorithm_score.objects.filter(
            updated_at__gte=cutoff_date
        ).values(
            'algorithm_name'
        ).annotate(
            avg_score=Avg('current_score'),
            max_score=Max('current_score'),
            min_score=Min('current_score')
        )
        
        # Combine data
        trends = {}
        
        for perf in performance_data:
            alg_name = perf['algorithm_name']
            trends[alg_name] = {
                'performance': perf,
                'scores': {}
            }
        
        for score in score_data:
            alg_name = score['algorithm_name']
            if alg_name in trends:
                trends[alg_name]['scores'] = score
            else:
                trends[alg_name] = {
                    'performance': {},
                    'scores': score
                }
        
        return trends
    
    @staticmethod
    def get_optimization_report() -> Dict:
        """Generate comprehensive optimization report"""
        # Get recent performance data
        recent_data = DatabasePerformanceAnalyzer.get_algorithm_trends(7)
        older_data = DatabasePerformanceAnalyzer.get_algorithm_trends(30)
        
        optimization_results = []
        
        for alg_name in recent_data.keys():
            recent_perf = recent_data[alg_name]['performance']
            older_perf = older_data.get(alg_name, {}).get('performance', {})
            
            if recent_perf.get('avg_execution_time') and older_perf.get('avg_execution_time'):
                improvement = ((older_perf['avg_execution_time'] - recent_perf['avg_execution_time']) / 
                              older_perf['avg_execution_time']) * 100
                
                optimization_results.append({
                    'algorithm': alg_name,
                    'recent_avg_time': recent_perf['avg_execution_time'],
                    'older_avg_time': older_perf['avg_execution_time'],
                    'improvement_percentage': improvement,
                    'recent_score': recent_data[alg_name]['scores'].get('avg_score', 0),
                    'executions': recent_perf.get('total_executions', 0)
                })
        
        # Sort by improvement
        optimization_results.sort(key=lambda x: x['improvement_percentage'], reverse=True)
        
        return {
            'report_timestamp': timezone.now().isoformat(),
            'total_algorithms_analyzed': len(optimization_results),
            'optimizations': optimization_results,
            'summary': {
                'best_improvement': optimization_results[0] if optimization_results else None,
                'avg_improvement': statistics.mean([r['improvement_percentage'] for r in optimization_results]) if optimization_results else 0
            }
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_algorithm_execution(algorithm_name: str):
    """Decorator to monitor algorithm execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                performance_monitor.record_execution(algorithm_name, execution_time, True)
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                performance_monitor.record_execution(algorithm_name, execution_time, False, str(e))
                raise
        return wrapper
    return decorator 