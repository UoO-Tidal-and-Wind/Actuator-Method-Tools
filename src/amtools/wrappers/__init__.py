"""
time_utils
==========

A lightweight utility module for measuring function execution time using decorators.

Modules:
--------

- :mod:`time_utils` – Contains the ``timeit`` decorator.

Usage example::

    from time_utils import timeit

    @timeit
    def do_work():
        time.sleep(1)

    do_work()
"""
