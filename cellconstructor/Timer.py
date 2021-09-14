import time
import numpy as np


class Timer:

    def __init__(self, active = False, print_each = None):
        """
        This class is used to time functions over several repeats

        Parameters
        ----------
            active : bool
                If True the timing is executed, otherwise not.
            print_each: float
                Each time a subroutine overrun the following value in seconds print its status.
                By default None, do not print unless specific requested.
        """

        self.active = active
        self.timed_subroutines = {}
        self.print_each = print_each

    def add_timer(self, name, value):
        """
        Add a timer to the specific value
        """
        if name in self.timed_subroutines:
            self.timed_subroutines[name]["time"] += value 
            self.timed_subroutines[name]["counts"] += 1
        else:
            self.timed_subroutines[name] = {"counts": 1, "time" : value} 

        if self.print_each is not None:
            #print (self.timed_subroutines[name]["time"])
            if self.timed_subroutines[name]["time"] > self.print_each:
                self.print_report()
                
                # Reset
                for name in self.timed_subroutines:
                    self.timed_subroutines[name]["time"] = 0
                    self.timed_subroutines[name]["counts"] = 0

    def execute_timed_function(self, function, *args, **kwargs):
        """
        Execute the function with the given arguments and keyword arguments and time it.
        This method returns whatever is returned by the function
        """
        if self.active:
            t1 = time.time()
            ret = function(*args, **kwargs)
            t2 = time.time()

            self.add_timer(function.__name__, t2-t1)
            return ret
        else:
            return function(*args, **kwargs)



    def print_report(self):
        """
        Print the report on timing for each function.
        """

        print("")
        print("")
        print(" /----------------\ ")
        print(" |  TIMER REPORT  | ")
        print(" \----------------/  ")
        print()

        for name in self.timed_subroutines:
            tt = self.timed_subroutines[name]["time"]
            hours = int(tt) // 3600
            tt -= hours * 3600
            minutes = int(tt) // 60
            tt -= minutes * 60

            print("Function: {}".format(name))
            print("N = {} calls took: {} hours; {} minutes; {:.2f} seconds".format(self.timed_subroutines[name]["counts"], hours, minutes, tt))
            print("Average of {} s per call".format(self.timed_subroutines[name]["time"] / self.timed_subroutines[name]["counts"]))
            print()
        
        print()
        print(" END OF TIMER REPORT ")
        print("---------------------")
        print()


    