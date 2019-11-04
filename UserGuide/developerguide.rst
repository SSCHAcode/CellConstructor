*****************
Developer's guide
*****************

In this chapter, I will introduce you to the code development.


Coding guidelines
=================

The CellConstructor module is written in mixed python, C and Fortran.
Python is a glue between the Fortran and C parts.
If you want to add a new utility to the code, consider in writing it directly in python, as interfacing between Fortran or C code could be very difficult.

In particular, when coding in python, keep in mind the following rules.
1. Keep the code compatible between Python 2 and Python 3. Always begin the source files with::

     from __future__ import print_function
     from __future__ import division

   and keep in mind to use a syntax that does not complain if executed by both python2 and python3 interprets.
2. Write a docstring for each function and classes implemented.
   Remember to use the numpy syntax for the docstring. In this way, the function can be automatically documented and included in the API reference guide. An example of numpy syntax docstrign for the my_new_function(x)::

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
3. Name the function with capital letters for new words, for example::

     
     def HelloWorld(x):
        # Very good
	pass

   Not::

     def hello_world(x):
        # NO!
	pass
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

	      
The Fortran interface
=====================

Sometimes, you have already written a code in Fortran, and you want to add it to CellConstructor.

If this is the case, and a complete python rewrite is impractical, then you can exploit the f2py utility provided by distutils to compile the fortran code into a shared library that will be read by Python.

This is done automatically by the setup.py installation script.
Please, give a look to the FModules directory and the setup.py.

Insert the fortran modules inside FModules directory, then, add them to the setup.py source file list.
In this way the fortran code will be automatically compiled when CellConstructor is installed.



