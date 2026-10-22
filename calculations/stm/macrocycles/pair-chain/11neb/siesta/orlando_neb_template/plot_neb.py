#!/usr/bin/env python
# coding: utf-8

Number_of_images=5
NAME_OF_NEB_RESULTS='NEB.results'
#==========================================================================
# Written by Arsalan Akhtar 
# ICN2 31-August-2017 
#==========================================================================
# Libray imports
import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate


print ("*******************************************************************")
print ("Ploting NEB ...")
print ("Email:arsalan.akhtar@icn2.cat")
print ("*******************************************************************")

data=[]
with open(NAME_OF_NEB_RESULTS) as f:
    for ind, line in enumerate(f, 1):
        if ind>2:
            #print (line)
            data.append(line)     
while '\n' in data:
    data.remove('\n')


image_number=[]
reaction_coordinates=[]
Energy=[]
E_diff=[]
Curvature=[]
Max_Force=[]
for i in range(len(data)):
    image_number.append(float(data[i].split()[0]))
    reaction_coordinates.append(float(data[i].split()[1]))
    Energy.append(float(data[i].split()[2]))
    E_diff.append(float(data[i].split()[3]))
    Curvature.append(float(data[i].split()[4]))
    Max_Force.append(float(data[i].split()[5]))
#Last NEB

Total_Number=Number_of_images+2
shift=len(E_diff)-Total_Number


im=[]
x=[]
y=[]
y2=[]
for i in range(Total_Number):
    im.append(np.array(int(image_number[shift+i])))
    x.append(np.array(reaction_coordinates[shift+i]))
    y.append(np.array(E_diff[shift+i]))
    y2.append(np.array(Energy[shift+i]))
   
#Finding Barrier
Barrier=max(y)    


xnew = np.linspace(0, x[len(x)-1], num=1000, endpoint=True)
f1=interp1d(x, y,kind='linear')
f2=interp1d(x, y, kind='cubic')
f3=interp1d(x, y, kind='quadratic')


plt.plot(x,y,"o",xnew,f1(xnew),"-",xnew,f2(xnew),"--",xnew,f3(xnew),'r')
plt.title("Barrier Energy = "+str(Barrier)+" eV")
plt.legend(['data', 'linear', 'cubic','quadratic'], loc='best')


plt.savefig('NEB.png')
plt.savefig('NEB.pdf')
plt.savefig('NEB.jpeg',dpi=600)






