# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:48:04 2020

@author: aleft
"""

"""
Trabalho TCC
Modelagem reator CSTR
"""

#Bibliotecas
import numpy as np
import matplotlib.pyplot as plt


Cva=0.0824 #kmol/m3
Cbv=1.5324 #kmol/m3
Ccv=0 #kmol/m3
q=0.0720 #m3/min
qc=0.6307  #m3/min
ro=974.19 #kg/m3
roc=998 #kg/m3
Cp=3.7187 #kJ/(kg*K)
Cpc=4.182#kJ/(kg*K)
V=2.407 #m3
Vc=2 #m3
A=8.695 #m2
alpha=13.8 #kJ/(min*m2*K)
Tv=299.05 #K
Tvc=288.15 #K
g=10183 #K
DrH=-5.46e6 #kJ/kmol
kinf=2.8233e11 #min-1

#Inicialmente....
Ca=Cva
T=Tv
Tc=Tvc

dt=1

Ca_=[]
T_=[]
Tc_=[]
Tv_=[]
qc_=[]

for i in np.arange(0,1500,dt):
    Ca=Ca+((q*(Cva-Ca)/V)-(kinf*np.exp(-g/T)*Ca))*dt
    T=T+(((Tv-T)*q/V) -(alpha*A*(T-Tc)/(V*ro*Cp))+(kinf*np.exp(-g/T)*Ca*(-DrH)/(ro*Cp)))*dt
    Tc=Tc+((qc*(Tvc-Tc)/Vc)+(alpha*A*(T-Tc)/(Vc*roc*Cpc)))*dt
    
    #Perturbação degrau (temperatura de entrada)
    if i>600 and i <=1000:
        qc=0.10
    elif i>1000:
        qc=0.60
#    if i>500:
#        Tvc=289.5
        
    qc_.append(qc)
    Tv_.append(Tv)    
    Ca_.append(Ca)
    T_.append(T)
    Tc_.append(Tc)


plt.figure(figsize=(12,9))
plt.subplot(4,1,1)
plt.plot(qc_,'k--',label='Entrada Reator')
#plt.plot(Ca_)
plt.ylabel('q (m3/min)')



plt.subplot(4,1,2)
plt.plot(T_,label='Saída Reator')
plt.ylabel('T(K)')
plt.legend()

plt.subplot(4,1,3)
plt.plot(Tc_,label='Saída Jaqueta')
plt.ylabel('T(K)')
plt.legend()

plt.subplot(4,1,4)
plt.plot(Ca_,label='Ca')
plt.ylabel('kmol/m3')
plt.legend()

plt.xlabel('Pontos (tempo)')


plt.show()







