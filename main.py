# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:48:04 2020

@author: aleft
"""

"""
Código principal para simulação dos dados e comparação de técnicas de controle
"""

#Bibliotecas necessárias
import pyswarms as ps

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import time

from scipy.integrate import odeint,simps

#importando o arquivo com demais funções. Está disponivel na pasta TCC
import utils_tcc


#--------------------------------------------------------------------------------

#Modelo do sistema (Reator CSTR)
def plant(z,t,
          q,Cva,V,kinf,g,Tv,alpha,A,ro,Cp,DrH,qc,Tvc,Vc,roc,Cpc):
#    global q,Cva,V,kinf,g,Tv,alpha,A,ro,cp,DrH,qc,Tvc,Vc,roc,cpc
    Ca  = z[0]
    T   = z[1]
    Tc  = z[2]
    
    dCadt=(q*(Cva-Ca)/V)-(kinf*np.exp(-g/T)*Ca)
    dTdt=((Tv-T)*q/V) -(alpha*A*(T-Tc)/(V*ro*Cp))+(kinf*np.exp(-g/T)*Ca*(-DrH)/(ro*Cp))
    dTcdt=(qc*(Tvc-Tc)/Vc)+(alpha*A*(T-Tc)/(Vc*roc*Cpc))
    dzdt = [dCadt,dTdt,dTcdt]
    
    return dzdt

#--------------------------------------------------------------------------------

#Geração de dados
    
qc_,Tv_,Tvc_,q_,Ca_,T_,Tc_=utils_tcc.obtain_data(plant,Ca,T,Tc,
                                                 q,Cva,V,kinf,g,Tv,alpha,A,ro,Cp,DrH,qc,Tvc,Vc,roc,Cpc,
                                                 tempo_final=40000,plot=True)

#Estruturação de dados
data = np.vstack((qc_,Tv_,Tvc_,q_,Ca_,T_,Tc_)).T

data=pd.DataFrame(data)

#------------------------------------------------------------------------------------


"""
Treinamento das redes neurais
"""
#data=pd.read_csv('Dados_gerados_case_geral_v2', sep='\t', encoding='utf-8',header=None)

#data.columns=['qc_','Tv_','Tvc_','q_','Ca_','T_','Tc_']

#Lista de variáveis para reestruturação do DF 
lista=['Tvc_','T_']

               
#Função de reestruturação do DF data para entrada na rede
data=utils_tcc.sliding_window(data,lista,instantes_passados=2)

data=data.dropna()

data=data.loc[1000:,:]
#---------------------------------------------------------------------------------------

target=data.pop('T_')
_=data.pop('Tvc_')
#Padronização dos dados

scaler=StandardScaler()

entrada=scaler.fit_transform(data)


X_train=entrada[:32000,:]
y_train=target[:32000]

X_test=entrada[32000:,:]
y_test=target[32000:]


"""Quando treinar RNN"""

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


#Modelo MLP
model = tf.keras.Sequential([        
        tf.keras.layers.Dense(4,activation='relu',input_shape=(X_train.shape[1:])),
        keras.layers.Dense(1,activation='relu')
])

#Modelo Recorrente
model = tf.keras.Sequential([        
        tf.keras.layers.SimpleRNN(4,activation='relu',return_sequences=True,input_shape=(X_train.shape[1:])),
        keras.layers.Dense(1,activation='relu')
])
    
#Modelo LSTM
model = tf.keras.Sequential([        
        tf.keras.layers.LSTM(4,activation='relu',return_sequences=False,input_shape=(X_train.shape[1:])),
        keras.layers.Dense(1,activation='relu')
])
    

#Modelo GRU
model = tf.keras.Sequential([        
        tf.keras.layers.GRU(4,activation='relu',return_sequences=False,input_shape=(X_train.shape[1:])),
        keras.layers.Dense(1,activation='relu')
])
    

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse", optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])


"""
Treinamento
"""
history = model.fit(X_train, y_train, epochs=50)#,batch_size=1)
    


calc=model.predict(X_test).reshape(-1,1)
plt.title( 'Teste\nR2_score='+'%.3f'%r2_score(y_test,calc)+'  MAE='+'%.2f'% mean_absolute_error(y_test,calc))
plt.plot(y_test,calc,'.',label='Dados Teste',alpha=0.4)
plt.plot([min(calc),max(calc)],[min(calc),max(calc)],'-',label='Ideal')
plt.xlabel('T Real')
plt.ylabel('T Calculado')
plt.legend()
plt.show()


a=model.predict(X_train).reshape(-1,1)
plt.title( 'Treino\nR2_score='+'%.3f'%r2_score(y_train,a)+'  MAE='+'%.2f'% mean_absolute_error(y_train,a))
plt.plot(y_train,a,'.',label='Dados Teste',alpha=0.4)
plt.plot([min(a),max(a)],[min(a),max(a)],'-',label='Ideal')
plt.xlabel('T Real')
plt.ylabel('T Calculado')
plt.legend()
plt.show()



#model.save('Treino_LSTM_v3_.h5') 

#model = tf.keras.models.load_model('Treino_GRU_T.h5')
#definindo a função J do mpc (função objetivo)

#---------------------------------------------------------------------------

#Model predictive control


def J(u):   
    """
    u=Tvc[t-1,t,t+1]
    model = rede neural treinada
    h_pred = horizonte de predição
    h_contr = horizonte de controle
    SP = Set-point
    """
    global model,h_pred,SP,Tvc_t_1,T_t_1,Tvc_t_2,T_t_2,T_t_1_,T_,Tvc_t_2_,T_t_2_,T_t_1_
    #Parâmetro referente ao controlador (verificar artigos)
    wT=1000
    lambda_q=10

    #Informação do Tvc para ser utilizado no df 'd_Tvc'
    Tvc_t_2_=Tvc_t_2
    
    T_t_2_=T_t_2
    T_t_1_=T_t_1
    
    T_=[]
    
    for i in range(h_pred):
        
        if i<3:
          
            T_.append(float(model(np.asarray([np.ndarray.tolist(scaler.transform([[u[i],
                                                                                   T_t_1_,Tvc_t_2_,T_t_2_]]))]),training=False)))
            Tvc_t_2_=u[i]
            
            T_t_2_=T_t_1_
            T_t_1_=T_[i]
            
        else:
            T_.append(float(model(np.asarray([np.ndarray.tolist(scaler.transform([[u[3],
                                                                               T_t_1_,Tvc_t_2_,T_t_2_]]))]),training=False)))
            Tvc_t_2_=u[3]
            
            T_t_2_=T_t_1_
            T_t_1_=T_[i]
            
  
    #Array com os set points no decorrer da predição
    y_ref=[SP]*h_pred
    
    #Horizonte de controle (3)
    d_Tvc=[]
    d_Tvc.append((u[0]-Tvc_t_2_)**2)
    d_Tvc.append((u[1]-u[0])**2)
    d_Tvc.append((u[2]-u[1])**2)
    d_Tvc.append((u[3]-u[2])**2)
    
       
    #função objetivo
    obj=wT*sum((pd.Series(y_ref)-pd.Series(T_))**2)+lambda_q*sum(pd.Series(d_Tvc))
    
    return obj






#EDO do sistema (cstr)
def plant(z,t):
    global q,Cva,V,kinf,g,Tv,alpha,A,ro,Cp,DrH,qc,Tvc,Vc,roc,Cpc
    Ca  = z[0]
    T   = z[1]
    Tc  = z[2]
    
    dCadt=(q*(Cva-Ca)/V)-(kinf*np.exp(-g/T)*Ca)
    dTdt=((Tv-T)*q/V) -(alpha*A*(T-Tc)/(V*ro*Cp))+(kinf*np.exp(-g/T)*Ca*(-DrH)/(ro*Cp))
    dTcdt=(qc*(Tvc-Tc)/Vc)+(alpha*A*(T-Tc)/(Vc*roc*Cpc))
    dzdt = [dCadt,dTdt,dTcdt]
    
    return dzdt

#Condições Iniciais (Estado Estacionário)=-----------------

#FO
qc_t_1=0.6307
Tv_t_1=299.05
Tvc_t_1=288.15
q_t_1=0.072
T_t_1=296.7227
qc_t_2=0.6307
Tv_t_2=299.05
Tvc_t_2=288.15
q_t_2=0.072
T_t_2=296.7227

#Condições iniciais da EDO (partindo do estado estacionário)
Ca=0.0814416
T=296.7227
Tc=288.5237
Tvc=288.15

Tv=299.05 #K
#---------------------
#Entradas - Horizonte de predição e Set-point
h_pred=8
SP=298

z_=[]
z0 = [Ca,T,Tc]
Tvc_=[]
sp_=[]

u=[288.15,288.15,288.15,288.15]
					
def min_f(particles):
    #Otimização por enxame de partículas
        return [J(particle) for particle in particles]
    
    
constraints = (np.array([275,275,275,275]),
               np.array([310,310,310,310]))

start_time = time.time()
final_step=50
for i in range(final_step):
     
    if i==25:
        SP=297
    
    options={'c1':2,'c2':2,'w':0.5}
    optimizer=ps.single.GlobalBestPSO(n_particles=50,dimensions=4,options=options,bounds=constraints)
    popt=optimizer.optimize(min_f,iters=50)
    
    
    Tvc=popt[1][0]
    Tvc_.append(Tvc)
    #Intervalos de tempo para a integração
    t = np.linspace(0,1,2)
    
    z_.append(np.ndarray.tolist(odeint(plant,z0,t)[1]))
    
    z0=z_[-1]
    
#    Atualização do valor para a próxima iteração
    Tvc_t_2=Tvc_t_1
    Tvc_t_1=Tvc

    
    T_t_2=T_t_1
    T_t_1=z0[1]
    
    u=popt[1]
    
    print(z_)
    print('\n')
    print(T_)
    
    print('\n %.1f %%'%(i*100/final_step))
    print(70*'=')
    
    sp_.append(SP)


