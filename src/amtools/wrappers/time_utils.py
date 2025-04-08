"""
time_utils.py
===============

This module provides a decorator to measure and display the execution time of functions.

Example usage::

    from timing_utils import timeit

    @timeit
    def slow_function():
        time.sleep(2)
"""

import time
from functools import wraps

def timeit(func):
    """
    Decorator that measures and prints the execution time of the decorated function.

    :param func: The function to be timed.
    :type func: Callable
    :return: The wrapped function with timing functionality.
    :rtype: Callable

    :Example:

    .. code-block:: python

        @timeit
        def my_function():
            time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed:.4f} seconds")
        return result
    return wrapper
