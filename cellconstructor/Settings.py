"""
This file keeps in mind common settings that needs to be initialized once.
"""

try:
    __PARALLEL_TYPE__ = "mp"
    import multiprocessing as mp
except:
    __PARALLEL_TYPE__ = "serial"



__SUPPORTED_LIBS__ = ["mp", "serial"]
__NPROC__ = 1


def SetupParallel(n_processors):
    """
    SETUP THE MODULE FOR PARALLEL EXECUTION
    =======================================
    
    This method initialize the parallelization of the module.
    For serial execution use n_processors = 1.
    Note that this kind of parallelization is implemented in single nodes.
    
    Parameters
    ----------
        n_processors : int
            The number of processors to be used for the heavy parallel
            executions. If negative or zero, the system tries to determine 
            automatically the number of avea
    """
    
    global __NPROC__
    __NPROC__ = n_processors
    if n_processors > 1 and __PARALLEL_TYPE__ == "serial":
        raise ValueError("Error, trying to setup a parallel computation with a 'serial' library")
    if n_processors < 1:
        raise ValueError("Error, the number of processors must be 1 or higher")
    
def GetNProc():
    """
    GET THE PROCESSORS FOR THE PARALLEL COMPUTATION
    ===============================================
    
    This method returns the total number of processors currently setted up
    for a parallel execution. If a serial algorithm is chosen, then 1 is returned.
    You can modify it using the SetupParallel method.
    """
    
    return __NPROC__
    
def GoParallel(function, list_of_inputs, init_random_seed = False, args = None):
    """
    GO PARALLEL
    ===========
    
    Perform a parallel evaluation of the provided function with the spawned
    list of inputs, and returns a list of output
    
    Parameters
    ----------
        function : pointer to function
            The function to be executed in parallel
        list_of_inputs : list
            A list containing the inputs to be passed to the function.
            
    """
    if not __PARALLEL_TYPE__ in __SUPPORTED_LIBS__:
        raise ValueError("Error, wrong parallelization type: %s\nSupported types: %s" % (__PARALLEL_TYPE__, " ".join(__SUPPORTED_LIBS__)))
        
    
    if __PARALLEL_TYPE__ == "mp":
        p = mp.Pool(__NPROC__)
        if args != None:
            ret =  p.apply(function, )
        return p.map(function, list_of_inputs)
    elif __PARALLEL_TYPE__ == "serial":
        return map(function, list_of_inputs)
        
    