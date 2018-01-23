# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 07:55:14 2018
Dhiraj
@author: du
"""

name = 'dhirajupadhyaya dhiraj'
key ='abcdefghijklmneopqrstuvwxyz'

msg = input('Input your message')
msg.lower()
lmsg = msg.lower()

'a'.isalpha()

for c in lmsg:
    print(c)
    if c.isalpha():
        print(key[name.index(c)], end='')
    
    
    if c.isalpha():
        printkey(name.index(c), end='')
    else:
        print(c, end='')