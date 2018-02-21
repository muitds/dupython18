# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:47:50 2018
Dhiraj
@author: du
"""
#map

Lst1 = [10, 20, 30, 40]
Lst2 = [50, 60, 70, 80]

sum_Lsts = map(lambda num1,num2: num1+num2, Lst1,Lst2)
print("The sum of two lists", list(sum_Lsts))
