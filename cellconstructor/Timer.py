import time
import numpy as np
import cellconstructor as CC, cellconstructor.Settings
from cellconstructor.Settings import ParallelPrint as print


class Timer:

    def __init__(self, active=False, print_each=None, level=0):
        """
        This class is used to time functions over several repeats

        Parameters
        ----------
            active : bool
                If True the timing is executed, otherwise not.
            print_each: float
                Each time a subroutine overrun the following value in seconds print its status.
                By default None, do not print unless specific requested.
            level: int
                The level of subtimers. 
                If 0, This is the principal timer,
                if 1, this is a subtimer of the principal timer, etc.
        """

        self.active = active
        self.level = level
        self.timed_subroutines = {}
        self.print_each = print_each

    def add_timer(self, name, value, timer=None):
        """
        Add a timer to the specific value
        """
        if name in self.timed_subroutines:
            self.timed_subroutines[name]["time"] += value 
            self.timed_subroutines[name]["counts"] += 1
        else:
            self.timed_subroutines[name] = {"counts": 1, "time" : value, "timer" : timer} 

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

            # Check if the function accept the "timer" keyword argument
            new_timer = None
            if "timer" in function.__code__.co_varnames and not "timer" in kwargs:
                if function.__name__ in self.timed_subroutines:
                    new_timer = self.timed_subroutines[function.__name__]["timer"]
                else:
                    new_timer = Timer(active=True, print_each=self.print_each, level=self.level+1)
                ret = function(*args, timer=new_timer,**kwargs)
            else:
                ret = function(*args, **kwargs)
            t2 = time.time()

            self.add_timer(function.__name__, t2-t1, new_timer)
            return ret
        else:
            return function(*args, **kwargs)



    def print_report(self):
        """
        Print the report on timing for each function.
        """

        prefix = " "*4*self.level

        if self.level == 0:
            print("\n\n" + "="*24 + "\n" + " "*8 + "TIMER REPORT" + " "*8 + "\n" + "="*24 + "\n")

        for name in self.timed_subroutines:
            tt = self.timed_subroutines[name]["time"]
            hours = int(tt) // 3600
            tt -= hours * 3600
            minutes = int(tt) // 60
            tt -= minutes * 60

            print("{}Function: {}".format(prefix, name))
            print("{}N = {} calls took: {} hours; {} minutes; {:.2f} seconds".format(prefix, self.timed_subroutines[name]["counts"], hours, minutes, tt))
            print("{}Average of {} s per call".format(prefix, self.timed_subroutines[name]["time"] / self.timed_subroutines[name]["counts"]))
            if self.timed_subroutines[name]["timer"] is not None:
                print("{}Subroutine report:".format(prefix))
                self.timed_subroutines[name]["timer"].print_report()

            print()
        
        if self.level == 0:
            print()
            print(" END OF TIMER REPORT ")
            print("=====================")
            print()


        