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

#data.to_csv('Dados_gerados_case_geral_v2', sep='\t', encoding='utf-8', header=None, index=False)

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
#Salvando os dados
#data.to_csv('Dados_gerados_case_geral_t_2_v2_.csv', sep='\t', encoding='utf-8', header=None, index=False)

#---------------------------------------

#Leitura dos dados
#data=pd.read_csv('Dados_gerados_case_geral_t_2_v2.csv', sep='\t', encoding='utf-8',header=None)

#data.columns=['qc_', 'Tv_', 'Tvc_', 'q_', 'Ca_', 'T_', 'Tc_', 'qc_t-1', 'Tv_t-1',
#       'Tvc_t-1', 'q_t-1', 'Ca_t-1', 'T_t-1', 'Tc_t-1', 'qc_t-2', 'Tv_t-2',
#       'Tvc_t-2', 'q_t-2', 'Ca_t-2', 'T_t-2', 'Tc_t-2']


data.columns=['Tvc_','T_','Tvc_t-1','T_t-1','Tvc_t-2','T_t-2']


#data=data.drop(['qc_','Tv_','Tvc_','q_'],axis=1)

data=data.dropna()

data=data.loc[1000:,:]
#---------------------------------------------------------------------------------------

#data=data[ ['T_', 'qc_t-1', 'Tv_t-1',
#       'Tvc_t-1', 'q_t-1', 'T_t-1', 'qc_t-2', 'Tv_t-2',
#       'Tvc_t-2', 'q_t-2', 'T_t-2']]

#Definição de targets

#target_tc=data.pop('Tc_')
#target_ca=data.pop('Ca_')
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
    
print("--- %s seconds ---" % (time.time() - start_time))



#-----------------------------------------------------
"""
Plotagem dos resultados
"""
p_=pd.DataFrame(z_)


p_=pd.DataFrame(z_)

SAVE_RESULTS=p_.copy()
SAVE_RESULTS['sp']=sp_
SAVE_RESULTS['Tvc']=Tvc_

#SAVE_RESULTS.to_csv('Result_GRU_V1.csv', sep='\t', encoding='utf-8', header=None, index=False)
#SAVE_RESULTS.to_csv('Result_LSTM_V1.csv', sep='\t', encoding='utf-8', header=None, index=False)
#SAVE_RESULTS.to_csv('Result_RNN_V1.csv', sep='\t', encoding='utf-8', header=None, index=False)
#SAVE_RESULTS.to_csv('Result_MLP_V1.csv', sep='\t', encoding='utf-8', header=None, index=False)


#Métricas de erro
IAE=simps(abs(sp_-p_[1]))
ISE=simps((sp_-p_[1])**2)
ITAE=simps(np.linspace(0,final_step-1,final_step)*abs(sp_-p_[1]))
ITSE=simps(np.linspace(0,final_step-1,final_step)*((sp_-p_[1])**2))


#Plotagem
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





#=================================================================================

#Plotagem das versões

"""
MLP
"""
MLP_V1=pd.read_csv('Result_MLP_V1.csv', sep='\t', encoding='utf-8',header=None)
MLP_V2=pd.read_csv('Result_MLP_V2.csv', sep='\t', encoding='utf-8',header=None)
MLP_V3=pd.read_csv('Result_MLP_V3.csv', sep='\t', encoding='utf-8',header=None)
MLP_V4=pd.read_csv('Result_MLP_V4.csv', sep='\t', encoding='utf-8',header=None)
MLP_V5=pd.read_csv('Result_MLP_V5.csv', sep='\t', encoding='utf-8',header=None)
MLP_V6=pd.read_csv('Result_MLP_V6.csv', sep='\t', encoding='utf-8',header=None)
MLP_V7=pd.read_csv('Result_MLP_V7.csv', sep='\t', encoding='utf-8',header=None)

MLP_V1.loc[0,3]=296.727

plt.figure(figsize=(12,4))
plt.title("Rede Neural - MLP")
plt.plot(MLP_V1[1],label='wt=100 wtc=10   IAE=78.95    ITAE=2380.761' )
plt.plot(MLP_V2[1],label='wt=100 wtc=1    IAE=76.544    ITAE=2283.826')
plt.plot(MLP_V3[1],label='wt=100 wtc=0.1  IAE=76.75    ITAE=2107.683')
plt.plot(MLP_V4[1],label='wt=100 wtc=0.01 IAE=82.40    ITAE=2272.196')
plt.plot(MLP_V5[1],label='wt=10  wtc=10   IAE=57.77    ITAE=1482.844')
plt.plot(MLP_V6[1],label='wt=1   wtc=10   IAE=41.14    ITAE=1095.061')
plt.plot(MLP_V7[1],label='wt=0.1 wtc=10   IAE=44.002    ITAE=1244.954')

plt.plot(MLP_V1[3],'k--',label='Set Point')

plt.ylabel('Temp. Reator (K)')
plt.xlabel('Tempo (s)')
plt.legend(loc='best')

plt.show()



"""
RNN
"""
RNN_V1=pd.read_csv('Result_RNN_V1.csv', sep='\t', encoding='utf-8',header=None)
RNN_V2=pd.read_csv('Result_RNN_V2.csv', sep='\t', encoding='utf-8',header=None)
RNN_V3=pd.read_csv('Result_RNN_V3.csv', sep='\t', encoding='utf-8',header=None)
RNN_V4=pd.read_csv('Result_RNN_V4.csv', sep='\t', encoding='utf-8',header=None)
RNN_V5=pd.read_csv('Result_RNN_V5.csv', sep='\t', encoding='utf-8',header=None)
RNN_V6=pd.read_csv('Result_RNN_V6.csv', sep='\t', encoding='utf-8',header=None)
RNN_V7=pd.read_csv('Result_RNN_V7.csv', sep='\t', encoding='utf-8',header=None)

RNN_V1.loc[0,3]=296.727

plt.figure(figsize=(12,4))
plt.title("Rede Neural - RNN")
plt.plot(RNN_V1[1],label='wt=100 wtc=10   IAE= 42.92    ITAE=1250.885' )
plt.plot(RNN_V2[1],label='wt=100 wtc=1    IAE=20.45    ITAE=388.698')
plt.plot(RNN_V3[1],label='wt=100 wtc=0.1  IAE=18.39    ITAE=438.112')
plt.plot(RNN_V4[1],label='wt=100 wtc=0.01 IAE=19.23    ITAE=467.926')
plt.plot(RNN_V5[1],label='wt=10  wtc=10   IAE=46.35    ITAE=1332.922')
plt.plot(RNN_V6[1],label='wt=1   wtc=10   IAE=41.55    ITAE=1026.094')
plt.plot(RNN_V7[1],label='wt=0.1 wtc=10   IAE=35.58    ITAE=890.772')

plt.plot(RNN_V1[3],'k--',label='Set Point')

plt.ylabel('Temp. Reator (K)')
plt.xlabel('Tempo (s)')
plt.legend()

plt.show()



"""
LSTM
"""
LSTM_V1=pd.read_csv('Result_LSTM_V1.csv', sep='\t', encoding='utf-8',header=None)
LSTM_V2=pd.read_csv('Result_LSTM_V2.csv', sep='\t', encoding='utf-8',header=None)
LSTM_V3=pd.read_csv('Result_LSTM_V3.csv', sep='\t', encoding='utf-8',header=None)
LSTM_V4=pd.read_csv('Result_LSTM_V4.csv', sep='\t', encoding='utf-8',header=None)
LSTM_V5=pd.read_csv('Result_LSTM_V5.csv', sep='\t', encoding='utf-8',header=None)
LSTM_V6=pd.read_csv('Result_LSTM_V6.csv', sep='\t', encoding='utf-8',header=None)
LSTM_V7=pd.read_csv('Result_LSTM_V7.csv', sep='\t', encoding='utf-8',header=None)

LSTM_V1.loc[0,3]=296.727

plt.figure(figsize=(12,4))
plt.title("Rede Neural - LSTM")
plt.plot(LSTM_V1[1],label='wt=100 wtc=10   IAE=19.50    ITAE=438.779' )
plt.plot(LSTM_V2[1],label='wt=100 wtc=1    IAE=15.41    ITAE=318.853')
plt.plot(LSTM_V3[1],label='wt=100 wtc=0.1  IAE=15.40    ITAE=316.125')
plt.plot(LSTM_V4[1],label='wt=100 wtc=0.01 IAE=14.75    ITAE=306.174')
plt.plot(LSTM_V5[1],label='wt=10  wtc=10   IAE=37.49    ITAE=882.845')
plt.plot(LSTM_V6[1],label='wt=1   wtc=10   IAE=41.77    ITAE=1085.380')
plt.plot(LSTM_V7[1],label='wt=0.1 wtc=10   IAE=31.44    ITAE=550.823')

plt.plot(LSTM_V1[3],'k--',label='Set Point')

plt.ylabel('Temp. Reator (K)')
plt.xlabel('Tempo (s)')
plt.legend()

plt.show()



"""
GRU
"""
GRU_V1=pd.read_csv('Result_GRU_V1.csv', sep='\t', encoding='utf-8',header=None)
GRU_V2=pd.read_csv('Result_GRU_V2.csv', sep='\t', encoding='utf-8',header=None)
GRU_V3=pd.read_csv('Result_GRU_V3.csv', sep='\t', encoding='utf-8',header=None)
GRU_V4=pd.read_csv('Result_GRU_V4.csv', sep='\t', encoding='utf-8',header=None)
GRU_V5=pd.read_csv('Result_GRU_V5.csv', sep='\t', encoding='utf-8',header=None)
GRU_V6=pd.read_csv('Result_GRU_V6.csv', sep='\t', encoding='utf-8',header=None)
GRU_V7=pd.read_csv('Result_GRU_V7.csv', sep='\t', encoding='utf-8',header=None)

GRU_V1.loc[0,3]=296.727

plt.figure(figsize=(12,4))
plt.title("Rede Neural - GRU")
plt.plot(GRU_V1[1],label='wt=100 wtc=10   IAE=19.39    ITAE=437.120' )
plt.plot(GRU_V2[1],label='wt=100 wtc=1    IAE=18.50    ITAE=416.020')
plt.plot(GRU_V3[1],label='wt=100 wtc=0.1  IAE=15.21    ITAE=319.421')
plt.plot(GRU_V4[1],label='wt=100 wtc=0.01 IAE=12.95    ITAE=226.503')
plt.plot(GRU_V5[1],label='wt=10  wtc=10   IAE=25.83    ITAE=442.669')
plt.plot(GRU_V6[1],label='wt=1   wtc=10   IAE=38.22    ITAE=999.813')
plt.plot(GRU_V7[1],label='wt=0.1 wtc=10   IAE=40.74    ITAE=1131.692')

plt.plot(GRU_V1[3],'k--',label='Set Point')

plt.ylabel('Temp. Reator (K)')
plt.xlabel('Tempo (s)')
plt.legend()

plt.show()



"""
Melhores performances em relação às estruturas
"""

plt.figure(figsize=(12,4))
plt.title("Comparação Estruturas - Melhores pesos")
plt.plot(MLP_V6[1],label='MLP    wt=1   wtc=10   IAE=41.14    ITAE=1095.061')
plt.plot(RNN_V3[1],label='RNN    wt=100 wtc=0.1  IAE=18.39    ITAE=438.112')
plt.plot(LSTM_V4[1],label='LSTM  wt=100 wtc=0.01 IAE=14.75    ITAE=306.2')
plt.plot(GRU_V4[1],label='GRU    wt=100 wtc=0.01 IAE=12.95    ITAE=226.5')


plt.plot(GRU_V1[3],'k--',label='Set Point')
plt.ylabel('Temp. Reator (K)')
plt.xlabel('Tempo (s)')
plt.legend()

plt.show()



"""
Resumo das performances
"""
import seaborn as sns

Result=pd.read_csv('Plotagem_result.csv', sep=',', encoding='utf-8',header=0)

Result_wt=Result[Result['wt']==100]

Result_wtc=Result[Result['wtc']==10]
#Result_wtc=Result_wtc[Result['wt']<100]


"""
Case 1:
wt=100
"""
plt.figure(figsize=(9,8))
plt.subplot(4,1,1)
plt.title('Para wt=100')
ax = sns.barplot(x="wtc", y="IAE", hue="Estrutura",
                 data=Result_wt, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

plt.subplot(4,1,2)
ax = sns.barplot(x="wtc", y="ISE", hue="Estrutura",
                 data=Result_wt, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend([],[], frameon=False)

plt.subplot(4,1,3)
ax = sns.barplot(x="wtc", y="ITAE", hue="Estrutura",
                 data=Result_wt, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend([],[], frameon=False)

plt.subplot(4,1,4)
ax = sns.barplot(x="wtc", y="ITSE", hue="Estrutura",
                 data=Result_wt, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.legend([],[], frameon=False)
plt.show()







"""
Case 2:
wtc=10
"""
plt.figure(figsize=(9,8))
plt.subplot(4,1,1)
plt.title('Para wtc=10')
ax = sns.barplot(x="wt", y="IAE", hue="Estrutura",
                 data=Result_wtc, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

plt.subplot(4,1,2)
ax = sns.barplot(x="wt", y="ISE", hue="Estrutura",
                 data=Result_wtc, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend([],[], frameon=False)

plt.subplot(4,1,3)
ax = sns.barplot(x="wt", y="ITAE", hue="Estrutura",
                 data=Result_wtc, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.legend([],[], frameon=False)

plt.subplot(4,1,4)
ax = sns.barplot(x="wt", y="ITSE", hue="Estrutura",
                 data=Result_wtc, dodge=True,palette=sns.mpl_palette("GnBu_d"))
plt.legend([],[], frameon=False)
plt.show()


