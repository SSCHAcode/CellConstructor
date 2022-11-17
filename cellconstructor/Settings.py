from __future__ import print_function
"""
This file keeps in mind common settings that needs to be initialized once.
"""
import numpy as np

# The parallelization setup
__PARALLEL_TYPE__ = "serial"
try:
    import mpi4py 
    import mpi4py.MPI
    __PARALLEL_TYPE__ = "mpi4py"
except:
    try:
        __PARALLEL_TYPE__ = "serial"
        import multiprocessing as mp
    except:
        __PARALLEL_TYPE__ = "serial"



__SUPPORTED_LIBS__ = ["mp", "serial", "mpi4py"]
__MPI_LIBRARIES__ = ["mpi4py"]
__NPROC__ = 1

def ParallelPrint(*args, **kwargs):
    """
    Print only if I am the master
    """
    if am_i_the_master():
        print(*args, **kwargs)

def all_print(*args, **kwargs):
    """
    Print for all the processors
    """
    print("[RANK {}] ".format(get_rank()), end = "")
    print(*args, **kwargs)


def am_i_the_master():
    if __PARALLEL_TYPE__ == "mpi4py":
        comm = mpi4py.MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            return True
        return False
    else:
        return True

def get_rank():
    """
    Get the rank of the process
    """
    if __PARALLEL_TYPE__ == "mpi4py":
        comm = mpi4py.MPI.COMM_WORLD
        return comm.Get_rank()
    elif __PARALLEL_TYPE__ == "serial":
        return 0
    else:
        raise NotImplementedError("Error, I do not know what is the rank with the {} parallelization".format(__PARALLEL_TYPE__))
        
def broadcast(list_of_values, enforce_double = False, other_type = None):
    """
    Broadcast the list to all the processors from the master.
    It returns a list equal for all the processors from the master.

    If you are broadcasting a numpy array, use enforce_double. If the array is not a C double
    type, specify the other_type (must be an MPI type).

    NOTE: Now enforce_double is just a dumb variable, as it is always overridded.
          It seems that Bcast does not work as expected
    """

    # STRONG OVERRIDE OVER ENFORCE DOUBLE THAT IS NOT WORKING
    enforce_double = False 

    if __PARALLEL_TYPE__ == "mpi4py":
        comm = mpi4py.MPI.COMM_WORLD
        if comm.Get_size() == 1:
            return list_of_values

        if not enforce_double:
            return comm.bcast(list_of_values, root = 0)
        else:
            total_shape = list_of_values.shape
            mpitype =  mpi4py.MPI.DOUBLE
            if other_type is not None:
                mpitype = other_type
            new_data = list_of_values.ravel()
            comm.Bcast([new_data, np.prod(total_shape), mpitype], root = 0)
            return new_data.reshape(total_shape)
    elif __PARALLEL_TYPE__ == "serial":
        return list_of_values
    else:
        raise NotImplementedError("broadcast not implemented for {} parallelization.".format(__PARALLEL_TYPE__))


def barrier():
    """
    This function force the MPI processes to sync:
    All the processes arrived at this function are stopped until all the others call the same method. 
    """

    if __PARALLEL_TYPE__ == "mpi4py":
        comm = mpi4py.MPI.COMM_WORLD
        comm.barrier()

def SetupParallel(n_processors=1):
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
            automatically the number of aveabile.
            It is useless if the parallelization is done with MPI
    """
    
    global __NPROC__
    global __PARALLEL_TYPE__

    if __PARALLEL_TYPE__ == "mpi4py":
        comm = mpi4py.MPI.COMM_WORLD
        __NPROC__ = comm.Get_size()
        if __NPROC__ == 1:
            __PARALLEL_TYPE__ = "serial"

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

    if __PARALLEL_TYPE__ == "mpi4py":
        return mpi4py.MPI.COMM_WORLD.Get_size()
    
    return __NPROC__
    
def GoParallel(function, list_of_inputs, reduce_op = None):
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
        reduce_op : string
            If a reduction must be performed on output, specify the operator, 
            accepted are "+", "*". For now this is implemented only with MPI
            
    """
    if not __PARALLEL_TYPE__ in __SUPPORTED_LIBS__:
        raise ValueError("Error, wrong parallelization type: %s\nSupported types: %s" % (__PARALLEL_TYPE__, " ".join(__SUPPORTED_LIBS__)))
        
    
    if __PARALLEL_TYPE__ in __MPI_LIBRARIES__ or __PARALLEL_TYPE__ == "serial":
        if not reduce_op is None:
            if not reduce_op in ["*", "+"]:
                raise NotImplementedError("Error, reduction '{}' not implemented.".format(reduce_op))

        # Here we create the poll manually
        n_proc = GetNProc()
        rank = get_rank()

        # broadcast the values
        list_of_inputs = broadcast(list_of_inputs)

        # Prepare the work for the current processor
        # TODO: Use a generator
        computing_list = []
        for i in range(rank, len(list_of_inputs), n_proc):
            computing_list.append(list_of_inputs[i])

        #print("Rank {} is computing {} elements".format(rank, len(computing_list)))
        #all_print("Computing:", computing_list)
        

        # Perform the reduction
        if reduce_op == "+":
            result = 0
            for x in computing_list:
                result += function(x) 

        elif reduce_op == "*":
            result = 1
            for x in computing_list:
                result *= function(x)

        # If a reduction must be done, return
        if not reduce_op is None:
            if __PARALLEL_TYPE__ == "mpi4py":
                comm = mpi4py.MPI.COMM_WORLD
                results = comm.allgather(result) 
            elif __PARALLEL_TYPE__ == "serial":
                return result
            else:
                raise NotImplementedError("Error, not implemented {}".format(__PARALLEL_TYPE__))


            #np.savetxt("result_{}.dat".format(rank), result)
            result = results[0]
            # Perform the last reduction
            if reduce_op == "+":
                for i in range(1,len(results)):
                    result+= results[i]
            elif reduce_op == "*":
                for i in range(1,len(results)):
                    result*= results[i]
            
            #np.savetxt("result_{}_total.dat".format(rank), result)

            return result 
        else:
            raise NotImplementedError("Error, for now parallelization with MPI implemented only with reduction")
    else:
        raise NotImplementedError("Something went wrong: {}".format(__PARALLEL_TYPE__))

    #elif __PARALLEL_TYPE__ == "mp":
        #p = mp.Pool(__NPROC__)
        #return p.map(function, list_of_inputs)
    #elif __PARALLEL_TYPE__ == "serial":
        #return map(function, list_of_inputs)
        
    
def GoParallelTuple(function, list_of_inputs, reduce_op = None):
    """
    GO PARALLEL TUPLE
    ==================
    
    Perform a parallel evaluation of the provided function with the spawned
    list of inputs, and returns a list of output. It works well if function returns more than one result
    
    Parameters
    ----------
        function : pointer to function
            The function to be executed in parallel. It must return a tuple, and each element of the tuple must be defined
        list_of_inputs : list
            A list containing the inputs to be passed to the function.
        reduce_op : string
            If a reduction must be performed on output, specify the operator, 
            accepted are "+", "*". For now this is implemented only with MPI
            
    """
    if not __PARALLEL_TYPE__ in __SUPPORTED_LIBS__:
        raise ValueError("Error, wrong parallelization type: %s\nSupported types: %s" % (__PARALLEL_TYPE__, " ".join(__SUPPORTED_LIBS__)))
        
    
    if __PARALLEL_TYPE__ in __MPI_LIBRARIES__ or __PARALLEL_TYPE__ == "serial":
        if not reduce_op is None:
            if not reduce_op in ["*", "+"]:
                raise NotImplementedError("Error, reduction '{}' not implemented.".format(reduce_op))

        # Here we create the poll manually
        n_proc = GetNProc()
        rank = get_rank()

        # broadcast the values
        list_of_inputs = broadcast(list_of_inputs)

        # Prepare the work for the current processor
        # TODO: Use a generator
        computing_list = []
        for i in range(rank, len(list_of_inputs), n_proc):
            computing_list.append(list_of_inputs[i])

        #print("Rank {} is computing {} elements".format(rank, len(computing_list)))
        
        # Work! TODO: THIS IS VERY MEMORY HEAVY
        results = [function(x) for x in computing_list]


        # Perform the reduction
        if reduce_op == "+":
            result = list(results[0])
            for i in range(1,len(results)):
                for j in range(len(results[i])):
                    result[j] += results[i][j]
            

        if reduce_op == "*":
            result = list(results[0])
            for i in range(1,len(results)):
                for j in range(len(results[i])):
                    result[j] *= results[i][j]


        # If a reduction must be done, return
        if not reduce_op is None:
            if __PARALLEL_TYPE__ == "mpi4py":
                comm = mpi4py.MPI.COMM_WORLD
                results = []
                for i in range(len(result)):
                    results.append(comm.allgather(result[i]))

            elif __PARALLEL_TYPE__ == "serial":
                return result
            else:
                raise NotImplementedError("Error, not implemented {}".format(__PARALLEL_TYPE__))


            result = results[0]
            for j in range(len(results)):
                # Perform the last reduction
                if reduce_op == "+":
                    for i in range(1,len(results[j])):
                        result[j]+= results[j][i]
                elif reduce_op == "*":
                    for i in range(1,len(results[j])):
                        result[j]*= results[j][i]

            return result 
        else:
            raise NotImplementedError("Error, for now parallelization with MPI implemented only with reduction")
    else:
        raise NotImplementedError("Something went wrong: {}".format(__PARALLEL_TYPE__))

    #elif __PARALLEL_TYPE__ == "mp":
        #p = mp.Pool(__NPROC__)
        #return p.map(function, list_of_inputs)
    #elif __PARALLEL_TYPE__ == "serial":
        #return map(function, list_of_inputs)
        
