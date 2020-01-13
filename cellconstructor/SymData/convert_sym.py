from __future__ import print_function

from numpy import *
import sys, os

if len(sys.argv) != 2:
    print ("Error, insert the file to parse as cmd")
    exit()
    
fname= sys.argv[1]

# The symbols
x = array([1,0,0,0])
y = array([0,1,0,0])
z = array([0,0,1,0])
hf= array([0,0,0,0.5])

if not os.path.exists(fname):
    print ("Error, the specified file does not exist")
    exit()

fdata = file(fname, "r")
lines = fdata.readlines()
fdata.close()

# Print the total number of symmetries
print (len(lines))

for line in lines:
    # Cut off the initial space
    new_line = line.strip()
    new_line = new_line[ new_line.find(" ") + 1 :]

    # Split the line into the symmetries
    vectors = new_line.split(",")

    matrix = zeros((3,4))
    for i, v in enumerate(vectors):
        new_v = v.replace("1/2", "hf")
        matrix[i, :] = eval(new_v)
        print   ("%.8f %.8f %.8f %.8f" % (matrix[i,0],
                                         matrix[i,1],
                                         matrix[i,2],
                                         matrix[i,3]))
        

    print ()

        
