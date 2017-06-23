#!/usr/bin/env python

import datetime

class TimerBucket():

    def __init__(self, bucket_name, print_each):
        self.bucket_name = bucket_name
        self._count = 0
        self._print_each = print_each
        self._sum = datetime.timedelta(0)

    def get_average(self):
        return self._sum / self._count

    def print_statistics(self):
        print("Bucket {}, average time: {}".format(self.bucket_name, self.get_average()))
        print("Bucket {},   total time: {}".format(self.bucket_name, self._sum))

    def start_timer(self, timer_name):
        start_time = datetime.datetime.utcnow()
        def stop_timer():
            stop_time = datetime.datetime.utcnow()
            delta = stop_time - start_time
            self._count += 1
            self._sum += delta
            if self._print_each:
                suffix = "[SLOW]" if delta > 2 * self.get_average() else ""
                print("[TIMER] {} / {}: {}μs {}".format(self.bucket_name, timer_name, delta, suffix))
        return stop_timer
