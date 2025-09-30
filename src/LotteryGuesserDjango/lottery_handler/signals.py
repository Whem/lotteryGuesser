import os
import importlib
import numpy as np
import statistics
import logging
import random
import hashlib
from datetime import timedelta

from django.utils import timezone
from django.http import JsonResponse

from lottery_handler.models import lg_generated_lottery_draw, lg_algorithm_score
from algorithms.models import lg_lottery_type, lg_lottery_winner_number

logger = logging.getLogger(__name__)


def list_processor_files():
    processor_files = []
    directory = "processors"  # Path to your processors folder

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            processor_files.append(filename[:-3])  # Remove '.py' from the filename

    # Ensure deterministic ordering
    return sorted(processor_files)


def get_top_algorithms(limit=10):
    # Get the top performing algorithms based on their scores
    return [
        score.algorithm_name
        for score in lg_algorithm_score.objects.all().order_by('-current_score', 'algorithm_name')[:limit]
    ]


def convert_numpy_to_python_types(obj):
    """Convert NumPy types to Python native types to ensure JSON serialization works"""
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python_types(item) for item in obj]
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python_types(value) for key, value in obj.items()}
    
    # Convert NumPy integer types to Python int
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    
    # Convert NumPy float types to Python float
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    
    # Convert NumPy boolean to Python bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Return original object if no conversion needed
    return obj


def _sanitize_numbers(numbers, min_num, max_num, required_count):
    """Ensure numbers are ints, unique, within [min,max], sorted, correct length.

    Deterministic fill if not enough values.
    """
    try:
        if numbers is None:
            numbers = []
        # Convert scalars to list
        if not isinstance(numbers, (list, tuple)):
            numbers = [numbers]
        # Cast to int and clamp to range
        cleaned = []
        for x in numbers:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi < min_num:
                xi = min_num
            elif xi > max_num:
                xi = max_num
            cleaned.append(xi)

        # Enforce uniqueness, keep order by value
        unique_sorted = sorted(set(cleaned))

        # Deterministic fill
        if required_count is None or required_count <= 0:
            return unique_sorted
        if len(unique_sorted) < required_count:
            seen = set(unique_sorted)
            for n in range(min_num, max_num + 1):
                if n not in seen:
                    unique_sorted.append(n)
                    seen.add(n)
                    if len(unique_sorted) == required_count:
                        break
        elif len(unique_sorted) > required_count:
            unique_sorted = unique_sorted[:required_count]

        return unique_sorted
    except Exception as e:
        logger.error(f"Sanitization error: {e}")
        # Fallback: deterministic first N
        if required_count and required_count > 0:
            return list(range(min_num, min(min_num + required_count, max_num + 1)))
        return []


def _too_similar(candidate_main, accepted_mains, required_count):
    """Return True if candidate is too similar to any accepted main set.

    Similarity rule: if overlap >= required_count - 1, treat as too similar (diff < 2).
    """
    try:
        cset = set(candidate_main)
        for aset in accepted_mains:
            if len(cset & aset) >= max(0, required_count - 1):
                return True
    except Exception as e:
        logger.debug(f"Similarity check error: {e}")
    return False


def call_get_numbers_dynamically(lottery_type_id):
    """Execute top algorithms for a lottery type with stable week-based diversity.

    - Processor discovery is sorted alphabetically.
    - Execution order follows ranked `lg_algorithm_score` descending; ties broken by `algorithm_name`.
    - Before each algorithm call, we set a deterministic per-week seed derived from (algorithm_name, ISO year, ISO week)
      for both Python's `random` and NumPy RNG. This yields reproducible results within a given week, and
      natural variation between weeks for algorithms that use randomness.
    """

    processor_files = list_processor_files()
    results = []
    seen_keys = set()
    accepted_main_sets = []  # store as set for fast overlap checks
    # Track keys we have already appended to results to avoid duplicates in the output list
    result_keys_added = set()

    lottery_type = lg_lottery_type.objects.filter(id=lottery_type_id.id).first()

    if lottery_type is None:
        return JsonResponse({"error": "Item not found"}, status=404)

    # Preload today's existing results to avoid intra-day duplicates
    try:
        now = timezone.now()
        start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = start_day + timedelta(days=1)
        existing_today = lg_generated_lottery_draw.objects.filter(
            lottery_type=lottery_type,
            created_at__gte=start_day,
            created_at__lt=end_day,
        )
        # Map from canonical key -> existing instance for quick reuse
        existing_key_to_instance = {}
        for row in existing_today:
            nums = row.lottery_type_number
            if isinstance(nums, dict):
                main = nums.get('main', [])
                add = nums.get('additional', [])
                key = ('dict', tuple(sorted(main)), tuple(sorted(add)))
            else:
                main = nums if isinstance(nums, list) else []
                key = ('list', tuple(sorted(main)))
            seen_keys.add(key)
            existing_key_to_instance[key] = row
            try:
                accepted_main_sets.append(set(int(x) for x in main))
            except Exception:
                pass
    except Exception as preload_err:
        logger.debug(f"Preload existing draws failed: {preload_err}")

    # Get the top performing algorithms deterministically
    top_algorithms = get_top_algorithms(limit=15)

    # Keep only files present and preserve ranking order; for stability, tie-break by name
    ranked_set = set(top_algorithms)
    ranked_processors = [name for name in top_algorithms if name in processor_files]

    # If none of the ranked algorithms are present (edge case), fall back to all discovered (already sorted)
    execution_list = ranked_processors if ranked_processors else processor_files

    for file in execution_list:
        try:
            logger.info(f"EXECUTING: {file}")

            module = importlib.import_module(f"processors.{file}")

            if hasattr(module, 'get_numbers'):
                # Deterministic per-week seeding for reproducible but week-varying results
                current_time = timezone.now()
                iso_week = current_time.isocalendar()[1]
                seed_basis = f"{file}-{current_time.year}-{iso_week}-lt{lottery_type.id}"
                seed_int = int(hashlib.sha256(seed_basis.encode('utf-8')).hexdigest()[:8], 16)
                try:
                    random.seed(seed_int)
                except Exception:
                    pass
                try:
                    np.random.seed(seed_int % (2**32))
                except Exception:
                    pass

                # Call the get_numbers function if it exists in the module
                lottery_numbers = module.get_numbers(lottery_type)

                # Handle tuple return (main_numbers, additional_numbers)
                if isinstance(lottery_numbers, tuple) and len(lottery_numbers) == 2:
                    main_numbers, additional_numbers = lottery_numbers
                    # Convert NumPy types to Python native types
                    main_numbers = convert_numpy_to_python_types(main_numbers)
                    additional_numbers = convert_numpy_to_python_types(additional_numbers)

                    # Keep originals for diagnostics
                    _orig_main = list(main_numbers) if isinstance(main_numbers, (list, tuple)) else [main_numbers]
                    _orig_add = list(additional_numbers) if isinstance(additional_numbers, (list, tuple)) else [additional_numbers]

                    # Sanitize per configured ranges
                    _san_main = _sanitize_numbers(
                        main_numbers,
                        int(lottery_type.min_number),
                        int(lottery_type.max_number),
                        int(lottery_type.pieces_of_draw_numbers or 0),
                    )
                    if getattr(lottery_type, 'has_additional_numbers', False):
                        _san_add = _sanitize_numbers(
                            additional_numbers,
                            int(lottery_type.additional_min_number or lottery_type.min_number),
                            int(lottery_type.additional_max_number or lottery_type.max_number),
                            int(lottery_type.additional_numbers_count or 0),
                        )
                    else:
                        _san_add = []

                    # Log if sanitizer changed anything
                    try:
                        if sorted(set(_orig_main)) != _san_main:
                            logger.debug(f"SANITIZE main changed for {file}: orig={sorted(set(_orig_main))} -> san={_san_main}")
                        if getattr(lottery_type, 'has_additional_numbers', False) and sorted(set(_orig_add)) != _san_add:
                            logger.debug(f"SANITIZE additional changed for {file}: orig={sorted(set(_orig_add))} -> san={_san_add}")
                    except Exception:
                        pass

                    main_numbers = _san_main
                    additional_numbers = _san_add

                    # Combine numbers for storage
                    final_numbers = main_numbers if not additional_numbers else {"main": main_numbers, "additional": additional_numbers}
                else:
                    # Handle legacy format or single list
                    main_numbers = convert_numpy_to_python_types(lottery_numbers)
                    _orig_main = list(main_numbers) if isinstance(main_numbers, (list, tuple)) else [main_numbers]
                    _san_main = _sanitize_numbers(
                        main_numbers,
                        int(lottery_type.min_number),
                        int(lottery_type.max_number),
                        int(lottery_type.pieces_of_draw_numbers or 0),
                    )
                    try:
                        if sorted(set(_orig_main)) != _san_main:
                            logger.debug(f"SANITIZE main changed for {file}: orig={sorted(set(_orig_main))} -> san={_san_main}")
                    except Exception:
                        pass
                    main_numbers = _san_main
                    final_numbers = main_numbers

                # Create result with enhanced data
                current_time = timezone.now()

                # Prepare canonical main list for downstream checks
                main_nums = final_numbers.get('main', []) if isinstance(final_numbers, dict) else final_numbers

                # Deduplikáció ugyanazon futáson belül
                if isinstance(final_numbers, dict):
                    key = ('dict', tuple(main_nums), tuple(final_numbers.get('additional', [])))
                else:
                    key = ('list', tuple(main_nums))
                if key in seen_keys:
                    # Instead of returning empty results, reuse the existing instance for this key
                    try:
                        existing = existing_key_to_instance.get(key)
                    except Exception:
                        existing = None
                    if existing is not None and key not in result_keys_added:
                        logger.info(f"USING EXISTING RESULT for {file}")
                        results.append(existing)
                        result_keys_added.add(key)
                        # Ensure diversity filter state remains consistent
                        try:
                            accepted_main_sets.append(set(main_nums))
                        except Exception:
                            pass
                    else:
                        logger.info(f"SKIP DUPLICATE RESULT from {file}")
                    continue
                seen_keys.add(key)

                # Calculate statistics only for main numbers (list format)
                # Diversity filtering on main numbers
                required_count = int(lottery_type.pieces_of_draw_numbers or 0)
                if required_count > 0 and _too_similar(main_nums, accepted_main_sets, required_count):
                    logger.info(f"SKIP TOO SIMILAR RESULT from {file}")
                    continue
                # Mark this main set as accepted to filter near-duplicates next
                try:
                    accepted_main_sets.append(set(main_nums))
                except Exception:
                    pass

                if isinstance(main_nums, list) and all(isinstance(n, (int, float)) for n in main_nums):
                    sum_val = sum(main_nums)
                    average_val = statistics.mean(main_nums)
                    median_val = statistics.median(main_nums)
                    try:
                        mode_val = statistics.mode(main_nums)
                    except statistics.StatisticsError:
                        mode_val = main_nums[0] if main_nums else 0
                    std_val = statistics.stdev(main_nums) if len(main_nums) > 1 else 0.0
                else:
                    sum_val = average_val = median_val = mode_val = std_val = 0

                result = lg_generated_lottery_draw.objects.create(
                    lottery_type=lottery_type,
                    lottery_type_number=final_numbers,
                    lottery_type_number_year=current_time.year,
                    lottery_type_number_week=current_time.isocalendar()[1],
                    sum=sum_val,
                    average=average_val,
                    median=median_val,
                    mode=mode_val,
                    standard_deviation=std_val,
                    lottery_algorithm=file,
                    created_at=current_time)
                results.append(result)
                result_keys_added.add(key)
        except Exception as e:
            logger.error(f"Hiba a {file} algoritmus futtatásakor: {e}")
            # Hibák naplózása teljesítményfigyeléshez
            try:
                import traceback
                logger.debug(f"Teljes stacktrace: {traceback.format_exc()}")
            except:
                pass
            # Folytatás a következő algoritmussal
            continue

    return results