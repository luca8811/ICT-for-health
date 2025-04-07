# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:24:18 2023

@author: d001834
"""
import numpy as np
import minimization as mymin
Np=5
Nf=3
A=np.random.randn(Np,Nf)
w=np.random.randn(Nf,1)
eps=np.random.randn(Np,1)*0.1
y=A@w+eps
# Utilizza il metodo 'solve' anziché 'run'
m = mymin.SolverLLS(y, A)
m.solve()

# Stampa e plotta i risultati
m.plot_result()
