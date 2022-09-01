#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#
#  Copyright 2022 Diego Martinez Gutierrez <diego.martinez@ehu.eus>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
# ---------------------------
# Importación de los módulos
# ---------------------------
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import math
# -------
# Clases
# -------
class Moro(object):
    def __init__(self):
        self.a0 =   2.50662823884
        self.a1 = -18.61500062529
        self.a2 =  41.39119773534
        self.a3 = -25.44106049637
        self.b0 =  -8.47351093090
        self.b1 =  23.08336743743
        self.b2 = -21.06224101826
        self.b3 =   3.13082909833
        self.c0 =   0.3374754822726147
        self.c1 =   0.9761690190917186
        self.c2 =   0.1607979714918209
        self.c3 =   0.0276438810333863
        self.c4 =   0.0038405729373609
        self.c5 =   0.0003951896511919
        self.c6 =   0.0000321767881768
        self.c7 =   0.0000002888167364
        self.c8 =   0.0000003960315187

    def gauss(self,u):
        y = u - 0.5
        if (abs(y) < 0.42):
            r = y * y
            #x = y * (((self.a3*r+self.a2)*r+self.a1)*r+self.a0)/((((self.b3*r+self.b2)*r+self.b1)*r+self.b0)*r+1)
            x = y * np.polyval([self.a3,self.a2,self.a1,self.a0],r)/np.polyval([self.b3,self.b2,self.b1,self.b0,1],r)
        else:
            r = u
            if (y > 0):
                r = 1-u
            r = np.log(-np.log(r))
            #x = self.c0+r*(self.c1+r*(self.c2+r*(self.c3+r*(self.c4+r*(self.c5+r*(self.c6+r*(self.c7+r*self.c8))))))
            x = np.polyval([self.c8,self.c7,self.c6,self.c5,self.c4,self.c3,self.c2,self.c1,self.c0],r)
            if (y < 0):
                x = -x
        return x

    def normalize(self,u):
        x = np.zeros(len(u))
        for i in range(len(u)):
            x[i] = self.gauss(u[i])
        return x

    def sobol(self,size,n_modes):
        sampler = qmc.Sobol(d=1, scramble=False)
        size_sobol = int(math.ceil(np.log(size)/np.log(2)))
        print ('size=',size,'size_sobol=',size_sobol,'ss=',2**size_sobol)
        # x = []
        # for i in range(n_modes):
        #     x.append(data)
        sample0 = sampler.random_base2(m=size_sobol)
        sample = sampler.random_base2(m=size_sobol)
#        print ('sample=',sample)
#        plt.hist(sample, bins=int(size/2))
#        plt.show()
#        plt.scatter(sample,range(len(sample)))
#        plt.show()
#        data = self.normalize((sample+0.01)%1)
        data = self.normalize(sample)
#        print ('data=',data)
#        plt.hist(data, bins=int(size/2))
#        plt.show()
#        plt.scatter(data,range(len(data)))
#        plt.show()
        m = len(data)-size      #****Diegom_test**** cut extra points (may work?)
        data1 = data[m:]
        x = np.resize(data1,(n_modes,size))
        print ('exit data:')
        print (x)
        return x

    def sobol_modes(self,size,n_modes,scramble):
        sampler = qmc.Sobol(d=n_modes, scramble=scramble)
        size_sobol = int(math.ceil(np.log(size)/np.log(2)))
        sample0 = sampler.random_base2(m=size_sobol)
        sample = sampler.random_base2(m=size_sobol)
        print ('size=',size,'size_sobol=',size_sobol,'ss=',2**size_sobol)
        #print (sample)
        data = np.zeros(shape=(size,n_modes))
        for i in range(size):
            for j in range(n_modes):
                data[i][j] = self.gauss(sample[i][j])
        m = len(data)-size      #****Diegom_test**** cut extra points (may work?)
        x = data[m:]
        #print ('exit data:')
        #print (x)

        return x

    def sobol_big(self,size,n_modes,scramble): # in case the number of vibrational modes excedes the max setting of the generator.
        if (n_modes>21201):
            number = n_modes-21201
            #if (number>21201): #Do it recursive???
            x1 = self.sobol_modes(size = size, n_modes = 21201 , scramble = scramble)
            x2 = self.sobol_big(size = size, n_modes = number, scramble = scramble)
            x = x1.append(x2)
        else:
            x = self.sobol_modes(size = size, n_modes = n_modes , scramble = scramble)
        return x

# ----------
# Funciones
# ----------

def main(args):
    size = 32
    n_modes = 3
    Sobol = Moro()
    # data = Sobol.sobol(size,n_modes)
    # for i in (range(n_modes)):
    #          plt.hist(data[i], bins=20)#int(size/2))
    #          plt.show()
    #          plt.scatter(data[i],range(len(data[i])))
    #          plt.show()

    data = Sobol.sobol_modes(size,n_modes)
    for i in (range(n_modes)):
             plt.hist(data.T[i], bins=20)#int(size/2))
             plt.show()
             plt.scatter(data.T[i],range(len(data.T[i])))
             plt.show()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
