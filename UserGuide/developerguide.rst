*****************
Developer's guide
*****************

In this chapter, I will introduce you to the code development.


The code management
===================

CellConstructor is distributed through the GIT control version system.

Get the last development code
-----------------------------

To download the last development distribution, use the command:

.. code:: bash

   $ git clone https://github.com/SSCHAcode/CellConstructor.git

In this way a new directory CellConstructor will be created.
After you created the directory, you can upload your local repository with the last
changes using the pull command.

.. code:: bash

   $ git pull

The pull command must be executed inside the CellConstructor directory.


What should I do BEFORE modify the code?
----------------------------------------

Note, the master branch of CellConstructor is holy: you should push your changes inside this branch
only after you are absolutely sure that they will not brake any function.
Moreover, unless you are an official developer of CellConstructor, you will not have the
permission to push your commits inside the official repository.

Then, how can you make your own changes available to other developers? The answer is **forking**.

Fork the CellConstructor repository, in this way you will create a new repository, synced with the
original one, where you are the owner.

You can do all the changes, commit and push them inside this repository.
When you think that everything is ok and ready to be merged inside the main repository, you can ask for a **pull request**.

CellConstructor uses GitHub (this may change in future). A detailed guide on how to manage forks and pull requests on
the github web site, please see:

https://help.github.com/en/github/getting-started-with-github/fork-a-repo



How do I report a bug?
----------------------

If you spot a bug, please open an issue on our GitHub page

https://github.com/mesonepigreco/CellConstructor

Before opening a new post, please, look if someone already spotted your same error (and maybe managed to find a workaround).
If not, open a new issue.

Describe in detail the problem and the calculation you are trying to run.
Please, **include an executable script with data** that triggers the bug.

We developers are not working 24h to answer issues, however, we will do our best to give you an answer as soon as we can.


Coding guidelines
=================

The CellConstructor module is written in mixed python, C and Fortran.
Python is a glue between the Fortran and C parts.
If you want to add a new utility to the code, consider in writing it directly in python, as interfacing between Fortran or C code could be very difficult.

In particular, when coding in python, keep in mind the following rules.

1. Keep the code compatible between Python 2 and Python 3. Always begin the source files with

   .. code:: python
	  
      from __future__ import print_function
      from __future__ import division

   and keep in mind to use a syntax that does not complain if executed by both python2 and python3 interprets.
2. Write a docstring for each function and classes implemented.
   Remember to use the numpy syntax for the docstring. In this way, the function can be automatically documented and included in the API reference guide. An example of numpy syntax docstrign for the my_new_function(x)

   .. code:: python

      def my_new_function(x):
         """
	 My New Function
	 ===============

	 This function takes a x parameter and returns its square.

	 .. math ::

	     y = x^2

	 The previous equation will become LaTeX code in the API.

	 Parameters
	 ----------
	    x : float
	       The input variable
	 Return
	 ------
	    y : float
	       The output (x*x)
	 """

	 return x*x

   This is an example of well documented code. Doing this each new function will simplify the life of the other developers, and also yourself when you will return to use this function some week after you wrote it.
   Please, remember to specify the types and dimensions of input arrays, as well as output. It is very upsetting having to look to the source to know what to pass to the function.
3. Name classes with capital letters for new words, for example::

     
     class MyNewClass(x):
        # Very good


   Name functions without capital letters, using underscores, for example::

      def my_function(x):
         pass
   
   NOTE: The code not always follows this convention, however, this is not a good excuse to avoid doing that yourself ;-)

4. Always use self-explaining names for the input variables. Avoid naming variables like ``pluto`` or ``goofy``

5. Comment each step of the algorithm. Always state what the algorithm does with comments, do not expect other people to understand what is in your mind when coding.
6. Avoid copy&paste. If you have to do an action twice, use a for loop, or define a function, do not copy&paste the code. If your code needs to be edited, it can be a pain to track all the positions of your copy&paste. Moreover, the smaller the code, the better.
7. Use numpy for math operation. Numpy is the numerical scientific library, it includes all the math libraries, linear algebra, fft, and so on. It can be compiled with many frontends, like Lapack, MKL. It is much faster than the native python math library.
8. Avoid unnecessary loop. Python is particularly slow when dealing with for loops. However, in math, most loop can be replaced by summation, products, and so on. Use the numpy functions sum, einsum, prod, dot, and so on to perform mathematical loop. Remember, a numpy sum call can be 1000 times faster than an explicit for loop to do a summation of an array.
9. Use exception handling and assertion. When you write a new function, do not expect the user to provide exactly the right data. If you need an input array of 3x4 elements, check it ussing the assert command::

     def MyFunction(x_array):

         # Check that x_array is a 2 rank tensor
         assert len(x_array.shape) == 2

	 # Check the shape of the x_array
         assert x_array.shape[0] == 3
	 assert x_array.shape[1] == 4

   Raise exceptions when the input is wrong::

       def sqrt(x):

          if x < 0:
	      raise ValueError("Error, x is lower than 0")


10. Always write a test inside the unittest suite to reproduce a known result. In this way, if bugs are introduced in future, they will be spotted immediately.
    To this purpose, see the next section.


Adding tests
------------

It is very important that each part of the code can be tested automatically each time a new feature is implemented.
The cellconstructor tests are based on unittest for now.
There is a script inside *scripts/cellconstructor_test.py*

Here, you will find a class that contains all the test runned when the command cellconstructor_test.py is executed. Remember, after editing each part of the code, no matter how small, always check that you did not break other parts by running the testsuite (after having reinstalled the software)

.. code:: bash

   $ cellconstructor_test.py

To add your own test, have a look inside that script. You just need to add a function to the class ``TestStructureMethods``. Your function must start with ``test_`` and take only ``self`` as argument.
To retrive some example dynamical matrix or strctures are inside ``self``.
For example, a ice XI structure is::

  # ICE XI structure
  self.struct_ice

  # Dynamical matrix of SnSe (a supercell)
  self.dynSnSe

  # Dynamical matrix of TiSe (a supercell)
  self.dynSky

You can also add your own file, by either expliciting coding it inside the ``__init__(self)`` method or by storing online and writing a download function. Remember that if you store them online, the file should be always be available.

Inside the testing function, you must check if the code is executed correctly by using ``self.assertTrue(cond)`` where ``cond`` is a bool condition that must be fullfilled, if not the test fails (it means a bug is present).

You can find online a more detailed guide on the ``unittest`` library.

You can also add custom tests inside the tests directory, naming a python script as test_my_function.py.
Look to other test you find in the same directory for examples.

	      
The Fortran interface
=====================

Sometimes, you have already written a code in Fortran, and you want to add it to CellConstructor.

If this is the case, and a complete python rewrite is impractical, then you can exploit the f2py utility provided by distutils to compile the fortran code into a shared library that will be read by Python.

This is done automatically by the setup.py installation script.
Please, give a look to the FModules directory and the setup.py.

Insert the fortran modules inside FModules directory, then, add them to the setup.py source file list.
In this way the fortran code will be automatically compiled when CellConstructor is installed.


How to include a new fortran source file
----------------------------------------

To include a new Fortran source file we must use the Extension class from distutils.
Let us take a look on how the symmetrization fortran module from quantum espresso has been imported into python. The fortran source files are contained inside the directory FModules. In the setup.py we have

.. code:: python

   from numpy.distutils.core import setup, Extension
   sources = [os.path.join("FModules", x) for x in os.listdir("FModules") if x.endswith(".f90")]
   
   symph_ext = Extension(name = "symph", sources = sources, libraries= ["lapack", "blas"], extra_f90_compile_args = ["-cpp"])

   setup(name = "CellConstructor", ext_modules = [symph_ext])

   

Here, I reduced only the lines we are interested in. First. I define a list that contains all the source files. In this case ``sources`` is a list of the paths to all the files that ends with ".f90" inside the FModules directory.
Then, I create an Extension object, named ``symph`` (the name of the package to be imported in python), linked to all the fortran soruce files listed inside the sources list, I specify the extra libraries needed for the link (if gfortran is used as default compiler, it will add -llapack -lblas to the compiling command). I can also specify extra flags or arguments for the fortran compiler. In this case, I use the "-cpp" flag. Then, the ``symph_ext`` object is added to the setup of the cellconstructor as an external module.

If you want to add a new function to the ``symph`` module, you just have to add it into the FModules directory and to the sources list (In this example, it will be recognized automatically, but in the actual setup.py all the files are manually listed, so remember to add it to the sources list).

Let us see a very simple example of a ``hello_world`` fotran module.

Create a new directory with the following ``hw.f90`` file::

   subroutine hello_world()
      print *, "Hello World"
   end subroutine hello_world
   
Then we can create our python extension. Make a ``setup.py`` file::

  from numpy.distutils.core import setup, Extension

  hw_f = Extension(name = "fort_hw", sources = ["hw.f90"])
  setup(name = "HW_IN_FORTRAN", ext_modules = [hw_f])

Now, you can try to install the module

.. code:: bash

   $ python setup.py install --user

To test if the module works, let us open an interactive python shell::

  >>> import fort_hw
  >>> fort_hw.hello_world()
   Hello World

Congratulations! You have your first Fortran module correctly compiled, installed, and working inside python.
For a more detailed guide on advanced features, refer to numpy fortran extension guide.

  

Fortran programming guidelines
-----------------------------

In the previous section we managed to make a very simple fortran extension to python. However, codes are always much more complicated.
**Remember: you are not writing a Fortran program, but a Fortran extension to a Python library**.
Keep your fortran code as simple as possible.

1. Always specify explicitly the intent and dimension of the input arrays::

      subroutine sum(a,b,c,n)
         double precision, intent(in), dimension(n) :: a,b
	 double precision, intent(out), dimension(n) :: c
	 integer n

	 c(:) = a(:) + b(:)
      end subroutine sum

   This code is the correct way to write a subroutine that sums two variables, avoid using ``dimension(:)`` in the declaration. Note that once your function is parsed in python, the fortran parser will recognized automatically that ``n`` is the dimension of the array. This means that **only in python** the dimension of the array can be omitted, as it will be inferred by the input variables.
   Note, array in fortran are numpy ndarray in python.
2. Pay attention to the typing. F2PY will automatically convert the type to match the input and output of your python functions, however, to get faster performances, it is better if you directly pass the correct type to the Fortran function. You can define a python type for the array using the dtype argument::

     import numpy as np
     a = np.zeros(10, dtype = np.double)

   This created the ``a`` array with 10 elements of type ``double precision``.
   You can find a detailed list of the python dtype and the corresponding fortran typing on the internet.

3. Multidimensional arrays. To preserve the readability of the code, f2py preserves the correct indexing of multidimensional arrays. If you have a python array like::

     mat = np.zeros((100, 10), dtype = np.double)

   It will be converted into a fortran array as::

      double precision, dimension(100, 10) :: mat

   However, by default, fortran stores in memory the multidimensional array in a different way than python. The fast index in fortran is the first one (in python it is the last one). This means that python needs to do a copy of the array before passing to (or retriving from) fortran to exchange the two indexes.
   If you know that a python array will be used extensively in fortran, you tell to python to create it directly in fortran order::

     mat = np.zeros((100, 10), dtype = np.double, order = "F")

   Now the ``mat`` array is stored in memory directly in fortran order, so no copy is needed to pass it to fortran.
   
4. Do not use custom types in fortran. Always pass to a subroutine or a function all the variable needed for that computations.

5. For better readability of the fortran code, it should be auspicable that you use a different source file for any different subroutine.

6. Avoid using any external library apart from blas and lapack. Remember that python is very good for linear algebra with numpy, so try to use Fortran only to perform critical computations that would require a slow massive for loop in python.

7. You can use openmp directives, but avoid importing the openmp library and use openmp subroutines, this breaks the compatibility if openmp is unavailable on the machine.

8. Always add a test file of the new function you implemented. In this way, if bugs are introduced in future, we will spot them immediately.

Fortran is very good to program fast tasks, however, the fortran converted subroutines are not documented and the input is uncontrolled. This means that passing an array with wrong size or typing can result in a Segmentation Fault error.
This is very annoying, as it can be very difficult to debug, especially if you are using a function written by someone else. **Each time you implement a fortran subroutine, write also the python parser**.

The parser is a python function that takes in input python arguments, converts them if necessary into the fortran types, verifies the size of the arrays to match exactly what the fortran function is expecting, calls the the fortran function, parses the output and return the output in python.
It is very important that the user of CellConstructor must **never** call directly a fortran function. This should also apply to other developers: try to make other peaple need only to call your final python functions and not directly the fortran ones.


