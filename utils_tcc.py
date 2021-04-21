# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:42:49 2020

@author: aleft
"""


def obtain_data(plant,Ca,T,Tc,
                q,Cva,V,kinf,g,Tv,alpha,A,ro,Cp,DrH,qc,Tvc,Vc,roc,Cpc
                ,tempo_final=30000,plot=True):
    """
    Tempo final: é o horizonte de geração de dados.
    Plant: edo com a função de comportamento do sistema
    Ca,T e Tc são condições iniciais.
    A plotagem será feita a partir dos instantes entre 10
    """
    import numpy as np
    import random
    from scipy.integrate import odeint
    import pandas as pd
    import matplotlib.pyplot as plt
       
    
#    global Ca,T,Tc
    
    #armazenamento de dados
    Tvc_=[]
    Tv_=[]
    qc_=[]
    q_=[]
    
    #Condição inicial do sistema
    z0 = [Ca,T,Tc]
    
    #Armazenamento do resultado da EDO. Ordem: Ca,T,Tc
    z_=[]
    
    cont=0
    #Loop para geração de dados (em cada instate de tempo)
    for i in np.arange(0,tempo_final,1):
        
        
            
        #Intervalos de tempo para a integração
        t = np.linspace(0,1,2)
        
        z0=np.ndarray.tolist(odeint(plant,z0,t,
                                           (q,Cva,V,kinf,g,Tv,alpha,A,ro,Cp,DrH,qc,Tvc,Vc,roc,Cpc))[1])
        
        
        #if i%360==0:
            
        z_.append(z0)
        qc_.append(qc)
        Tv_.append(Tv)    
        Tvc_.append(Tvc)
        q_.append(q)
            
            
            #Variável manipulada - Vazão do fluido da jaqueta
            #qc=random.uniform(0.6307-0.6307,0.6307+1.6307)
                  
            #Variável manipulada - Temperatura do fluido da jaqueta
        Tvc=random.uniform(288.15-13, 288.25+20)
    #        
            #Perturbação - Vazão de alimentação
#            q=random.uniform(0.0720-0.0720,0.0720+0.0720)
              
            #Perturbação - Temperatura de alimentação
#        Tv=random.uniform(299.05-20,299+29.905)
        
        cont+=1
        
        print('Progresso: %.1f%%'%(cont*100/tempo_final))
        
    
    Ca_=pd.DataFrame(z_)[0]
    T_=pd.DataFrame(z_)[1]
    Tc_=pd.DataFrame(z_)[2]
    
    qc_=pd.Series(qc_)
    Tv_=pd.Series(Tv_)
    Tvc_=pd.Series(Tvc_)
    q_=pd.Series(q_)
    
    if plot==True:
        plt.figure(figsize=(13,10))
        plt.subplot(7,1,1)
        plt.plot(qc_[1000:1300],'k-',label='Entrada Jaqueta')
        plt.ylabel('qc (m3/min)')
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.legend()
        
        plt.subplot(7,1,2)
        plt.plot(Tv_[1000:1300],'k-',label='Entrada Reator')
        plt.ylabel('Tv(K)')
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.legend()
        
        plt.subplot(7,1,3)
        plt.plot(Tvc_[1000:1300],'k-',label='Entrada Jaqueta')
        plt.ylabel('Tvc(K)')
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.legend()
        
        plt.subplot(7,1,4)
        plt.plot(q_[1000:1300],'k-',label='Entrada Reator')
        plt.ylabel('q (m3/min)')
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.legend()
        
        plt.subplot(7,1,5)
        plt.plot(T_[1000:1300],label='Saída Reator')
        plt.ylabel('T(K)')
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.legend()
        
        plt.subplot(7,1,6)
        plt.plot(Tc_[1000:1300],label='Saída Jaqueta')
        plt.ylabel('Tj(K)')
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        plt.legend()
        
        plt.subplot(7,1,7)
        plt.plot(Ca_[1000:1300],label='Ca')
        plt.ylabel('kmol/m3')
        plt.legend()
        
        plt.xlabel('Pontos (tempo)')
        
        plt.show()
    
    return qc_,Tv_,Tvc_,q_,Ca_,T_,Tc_



def sliding_window(data,lista,instantes_passados=2):
    """
    Função para gerar dados com valores de um DF data atrasados em instantes_passados
    a partir de uma lista com as variáveis a serem "atrasadas"
    """    
    for k in range(1,1+instantes_passados):
        for i in lista:
            for j in range(len(data)):
                if j>=k:
                    data.loc[j,i+'t-'+str(k)]=data.loc[(j-k,i)]
                    
    return data  
                    
                    
                    
                    
                    
                    
