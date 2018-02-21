# -*- coding: utf-8 -*-
"""
Mon Feb 19 15:34:01 2018: Dhiraj
"""
#lambda Functions

f = lambda x, y : x + y
f(1,1)


def fahrenheit(T):
    return ((float(9)/5)*T + 32)
def celsius(T):
    return (float(5)/9)*(T-32)
temp = (36.5, 37, 37.5,39)

F = map(fahrenheit, temp)
F
C = map(celsius, F)

Celsius = [39.2, 36.5, 37.3, 37.8]
Fahrenheit = map(lambda x: (float(9)/5)*x + 32, Celsius)
print(Fahrenheit)
C = map(lambda x: (float(5)/9)*(x-32), Fahrenheit)
print(C)



fib = [0,1,1,2,3,5,8,13,21,34,55]
result = filter(lambda x: x % 2, fib)
print(result)
result = filter(lambda x: x % 2 == 0, fib)
result

reduce(lambda x,y: x+y, [47,11,42,13])

f = lambda a,b: a if (a > b) else b
reduce(f, [47,11,42,102,13])
