# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:48:04 2020

@author: aleft
"""

"""
Trabalho TCC
Modelagem reator CSTR
Main file
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


#Valores das Entradas (com base no artigo do Fuzzy)
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



"""
-----------------------------------------------------------------------------------

***Possíveis Cases***
MV: [Case 1] qc - vazão de fluido refrigerante | [Case 2] Tvc - temperatura do fluido refrigerante |
    [Case 3] Os dois
CV: [Case 1.1/2.1/3.1] T - temperatura do reator |  [Case 1.1/2.1/3.1] Ca - Concentração de A

Possível perturbação: q, Tv


-----------------------------------------------------------------------------------
 
 Etapas:
    
1. Geração de dados
2. Modelo de predição (redes neurais)
3. Lei de controle

"""

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

data.columns=['qc_','Tv_','Tvc_','q_','Ca_','T_','Tc_']

data=data[['Tvc_','T_']]

#data.to_csv('Dados_gerados_case_geral_v2_', sep='\t', encoding='utf-8', header=None, index=False)

#------------------------------------------------------------------------------------


"""
Treinamento das redes neurais
"""
#data=pd.read_csv('Dados_gerados_case_geral_v2', sep='\t', encoding='utf-8',header=None)

data.columns=['Tvc_', 'T_']

#Lista de variáveis para reestruturação do DF 
lista=['Tvc_','T_']

#lista=['Tv_','Tvc_','T_']

               
#Função de reestruturação do DF data para entrada na rede
data=utils_tcc.sliding_window(data,lista,instantes_passados=2)
data.columns
#Salvando os dados
#data.to_csv('Dados_gerados_case_geral_t_2_v2_', sep='\t', encoding='utf-8', header=None, index=False)

#---------------------------------------

#Leitura dos dados
#data=pd.read_csv('Dados_gerados_case_geral_t_2_v2', sep='\t', encoding='utf-8',header=None)

#data.columns=['qc_', 'Tv_', 'Tvc_', 'q_', 'Ca_', 'T_', 'Tc_', 'qc_t-1', 'Tv_t-1',
#       'Tvc_t-1', 'q_t-1', 'Ca_t-1', 'T_t-1', 'Tc_t-1', 'qc_t-2', 'Tv_t-2',
#       'Tvc_t-2', 'q_t-2', 'Ca_t-2', 'T_t-2', 'Tc_t-2']


data.columns=['Tvc_','T_','Tvc_t-1','T_t-1','Tvc_t-2','T_t-2']


#data=data.drop(['qc_','Tv_','Tvc_','q_'],axis=1)

data=data.dropna()

#---------------------------------------------------------------------------------------

#data=data[ ['T_', 'qc_t-1', 'Tv_t-1',
#       'Tvc_t-1', 'q_t-1', 'T_t-1', 'qc_t-2', 'Tv_t-2',
#       'Tvc_t-2', 'q_t-2', 'T_t-2']]

#Definição de targets

#target_tc=data.pop('Tc_')
#target_ca=data.pop('Ca_')
target=data.pop('T_')
_=data.pop('Tvc_')
#_=data.pop('Tv_')

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
    

#def last_time_step_mse(Y_true, Y_pred):
#  return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

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



#model.save('Treino_MLP_T_v2_.h5') 

#model = tf.keras.models.load_model('Treino_GRU_T_.h5')
#definindo a função J do mpc (função objetivo)

#---------------------------------------------------------------------------

#Model predictive control




#def J(u):   
#    """
#    u=Tvc[t-1,t,t+1]
#    model = rede neural treinada
#    h_pred = horizonte de predição
#    h_contr = horizonte de controle
#    SP = Set-point
#    """
#    global model,h_pred,SP,qc_t_1, Tv_t_1,Tvc_t_1,q_t_1,T_t_1,qc_t_2,Tv_t_2,Tvc_t_2,q_t_2,T_t_2,T_t_1_,T_,Tvc_t_2_,T_t_2_,T_t_1_
#    #Parâmetro referente ao controlador (verificar artigos)
#    wT=100
#    lambda_q=10
#
##    start_time = time.time()
##    Loop para cálculo do horizonte de predição
#    
#    #Informação do Tvc para ser utilizado no df 'd_Tvc'
#    Tvc_t_2_=Tvc_t_2
#    
#    T_t_2_=T_t_2
#    T_t_1_=T_t_1
#    
#    T_=[]
#    
#    for i in range(h_pred):
#        
#        if i<3:
#          
#            T_.append(float(model(np.asarray([np.ndarray.tolist(scaler.transform([[qc_t_1, Tv_t_1,u[i],
#                                                                                   q_t_1,T_t_1_,qc_t_2,Tv_t_2,Tvc_t_2_,q_t_2,T_t_2_]]))]),training=False)))
#            Tvc_t_2_=u[i]
#            
#            T_t_2_=T_t_1_
#            T_t_1_=T_[i]
#            
#        else:
#            T_.append(float(model(np.asarray([np.ndarray.tolist(scaler.transform([[qc_t_1, Tv_t_1,u[3],
#                                                                               q_t_1,T_t_1_,qc_t_2,Tv_t_2,Tvc_t_2_,q_t_2,T_t_2_]]))]),training=False)))
#            Tvc_t_2_=u[3]
#            
#            T_t_2_=T_t_1_
#            T_t_1_=T_[i]
#            
##    print("--- %s seconds ---" % (time.time() - start_time))
#   
#    #Array com os set points no decorrer da predição
#    y_ref=[SP]*h_pred
#    
#    #Horizonte de controle (3)
#    d_Tvc=[]
#    d_Tvc.append((u[0]-Tvc_t_2_)**2)
#    d_Tvc.append((u[1]-u[0])**2)
#    d_Tvc.append((u[2]-u[1])**2)
#    d_Tvc.append((u[3]-u[2])**2)
#    
#       
#    #função objetivo
#    obj=wT*sum((pd.Series(y_ref)-pd.Series(T_))**2)+lambda_q*sum(pd.Series(d_Tvc))
#    
#    return obj

#
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
    wT=100
    lambda_q=10

#    start_time = time.time()
#    Loop para cálculo do horizonte de predição
    
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
            
#    print("--- %s seconds ---" % (time.time() - start_time))
   
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


#
#def J(u):   
#    """
#    u=Tvc[t-1,t,t+1]
#    model = rede neural treinada
#    h_pred = horizonte de predição
#    h_contr = horizonte de controle
#    SP = Set-point
#    """
#    global model,h_pred,SP, Tv_t_1,Tvc_t_1,T_t_1,Tv_t_2,Tvc_t_2,T_t_2,T_t_1_,T_,Tvc_t_2_,T_t_2_,T_t_1_
#    #Parâmetro referente ao controlador (verificar artigos)
#    wT=100
#    lambda_q=10
#
##    start_time = time.time()
##    Loop para cálculo do horizonte de predição
#    
#    #Informação do Tvc para ser utilizado no df 'd_Tvc'
#    Tvc_t_2_=Tvc_t_2
#    
#    T_t_2_=T_t_2
#    T_t_1_=T_t_1
#    
#    T_=[]
#    
#    for i in range(h_pred):
#        
#        if i<3:
#          
#            T_.append(float(model(np.asarray([np.ndarray.tolist(scaler.transform([[Tv_t_1,u[i],
#                                                                                   T_t_1_,Tv_t_2,Tvc_t_2_,T_t_2_]]))]),training=False)))
#            Tvc_t_2_=u[i]
#            
#            T_t_2_=T_t_1_
#            T_t_1_=T_[i]
#            
#        else:
#            T_.append(float(model(np.asarray([np.ndarray.tolist(scaler.transform([[Tv_t_1,u[3],
#                                                                               T_t_1_,Tv_t_2,Tvc_t_2_,T_t_2_]]))]),training=False)))
#            Tvc_t_2_=u[3]
#            
#            T_t_2_=T_t_1_
#            T_t_1_=T_[i]
#            
##    print("--- %s seconds ---" % (time.time() - start_time))
#   
#    #Array com os set points no decorrer da predição
#    y_ref=[SP]*h_pred
#    
#    #Horizonte de controle (3)
#    d_Tvc=[]
#    d_Tvc.append((u[0]-Tvc_t_2_)**2)
#    d_Tvc.append((u[1]-u[0])**2)
#    d_Tvc.append((u[2]-u[1])**2)
#    d_Tvc.append((u[3]-u[2])**2)
#    
#       
#    #função objetivo
#    obj=wT*sum((pd.Series(y_ref)-pd.Series(T_))**2)+lambda_q*sum(pd.Series(d_Tvc))
#    
#    return obj







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
        return [J(particle) for particle in particles]
    
    
constraints = (np.array([275,275,275,275]),
               np.array([310,310,310,310]))

start_time = time.time()
final_step=25
for i in range(final_step):
     
    #if i==30:
    #    SP=297.5
    #elif i==60:
    #    SP=298.5

    
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
    
print("--- %s seconds ---" % (time.time() - start_time))



#-----------------------------------------------------
"""
Plotagem dos resultados
"""
p_=pd.DataFrame(z_)

#plt.plot(p_[0])

#Métricas de erro
IAE=simps(abs(SP-p_[1]))
ISE=simps((SP-p_[1])**2)
ITAE=simps(np.linspace(0,final_step-1,final_step)*abs(SP-p_[1]))
ITSE=simps(np.linspace(0,final_step-1,final_step)*((SP-p_[1])**2))


plt.figure(figsize=(12,4))
plt.subplot(2,1,1)
plt.title("IAE= "+'%.3f'%IAE+ "   ISE= "+'%.3f'%ISE+"   ITAE= "+'%.3f'%ITAE+"   ITSE= "+'%.3f'%ITSE)
plt.step(np.linspace(0,final_step-1,final_step),Tvc_,'k',label='Ações controlador')
plt.ylabel('T_coolant - MV (K)')
#plt.xlabel('Tempo (s)')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend()


#plt.figure(figsize=(12,4))
plt.subplot(2,1,2)
plt.plot(p_[1],label='Temp Reator')
plt.plot(sp_,label='Set Point')
plt.ylabel('K')
plt.xlabel('Tempo (s)')

plt.xticks(np.linspace(0,final_step-1,final_step/5))
plt.legend()

plt.show()





























