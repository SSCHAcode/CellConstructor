import time
import numpy as np
import cellconstructor as CC, cellconstructor.Settings
from cellconstructor.Settings import ParallelPrint as print
import inspect
import json


# Recursively transform a class into a dictionary
def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return to_dict(obj.__dict__)
    elif isinstance(obj, list):
        return [to_dict(elem) for elem in obj]
    else:
        return obj
    
# Recursively transform a dictionary into the timer object
def to_timer(dict):
    if "timed_subroutines" in dict:
        for name in dict["timed_subroutines"]:
            if dict["timed_subroutines"][name]["timer"] is not None:
                dict["timed_subroutines"][name]["timer"] = to_timer(dict["timed_subroutines"][name]["timer"])
        
        timer = Timer()
        timer.__dict__.update(dict)
        return timer
    return dict


def load_json(filename):
    """
    Load a timer object from a json file
    """
    with open(filename, "r") as f:
        my_dyct = json.load(f)
        return to_timer(my_dyct)


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

    def spawn_child(self):
        """Spawn a child timer."""
        return Timer(active=self.active, print_each=self.print_each, level=self.level+1)

    def execute_timed_function(self, function, *args, override_name="", **kwargs):
        """
        Execute the function with the given arguments and keyword arguments and time it.
        This method returns whatever is returned by the function
        """
        if self.active:
            t1 = time.time()

            # Check if the function accept the "timer" keyword argument
            new_timer = None
            func_name = override_name
            isfunction = inspect.isfunction(function) or inspect.ismethod(function)
            if isfunction:
                if not func_name:
                    func_name = function.__name__

                sig = inspect.signature(function)
                tparam = sig.parameters.get("timer")
                if tparam is not None and not "timer" in kwargs:
                    if func_name in self.timed_subroutines:
                        new_timer = self.timed_subroutines[func_name]["timer"]
                    else:
                        new_timer = self.spawn_child()
                    ret = function(*args, timer=new_timer,**kwargs)
                else:
                    ret = function(*args, **kwargs)
            else:
                assert callable(function), "The function argument must be a function or a callable object, got: {}".format(type(function))
                ret = function(*args, **kwargs)
                if not func_name:
                    func_name = function.__class__.__name__
            t2 = time.time()


            self.add_timer(func_name, t2-t1, new_timer)
            return ret
        else:
            return function(*args, **kwargs)

    def save_json(self, filename):
        """
        Save the timing data to a file in json format
        """
        # save all the dictionary of this function as a json file
        with open(filename, "w") as f:
            my_dyct = to_dict(self)
            json.dump(my_dyct, f, indent=4)

    def print_report(self, master_level=0, is_master=False, verbosity_limit=0.05):
        """
        Print the report on timing for each timer and subtimer contained.

        Parameters
        ----------
            master_level: int
                The level of the master timer. This is used to print the correct indentation.
            is_master: bool
                If True, ignore the master_level keyword and assume this timer is the master.
                This is equivalent to setting master_level = self.level
            verbosity_limit: float
                The minimum percentage of time occupied by each function above which the function is printed.
                This is used to avoid printing too many functions that are not relevant.
                Put 0 to print all functions.
        """
        if is_master:
            master_level = self.level

        level = self.level - master_level

        prefix = " "*4*level

        if level == 0:
            print("\n\n" + "="*24 + "\n" + " "*6 + "TIMER REPORT" + " "*6 + "\n" + "="*24 + "\n")
            print("Threshold for printing: {:d} %".format(int(verbosity_limit*100)))
            print()

        total_time = sum([self.timed_subroutines[name]["time"] for name in self.timed_subroutines])

        for name in self.timed_subroutines:
            tt = self.timed_subroutines[name]["time"]
            if tt / total_time < verbosity_limit:
                continue
            hours = int(tt) // 3600
            tt -= hours * 3600
            minutes = int(tt) // 60
            tt -= minutes * 60

            print("{}Function: {}".format(prefix, name))
            print("{}N = {} calls took: {} hours; {} minutes; {:.2f} seconds".format(prefix, self.timed_subroutines[name]["counts"], hours, minutes, tt))
            print("{}Average of {} s per call".format(prefix, self.timed_subroutines[name]["time"] / self.timed_subroutines[name]["counts"]))
            if self.timed_subroutines[name]["timer"] is not None:
                print("{}Subroutine report:".format(prefix))
                self.timed_subroutines[name]["timer"].print_report(master_level=master_level, verbosity_limit=verbosity_limit)

            print()
        
        if level == 0:
            print()
            print(" END OF TIMER REPORT ")
            print("=====================")
            print()


        