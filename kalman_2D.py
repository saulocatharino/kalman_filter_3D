#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pykalman import KalmanFilter
import random

fig = plt.figure(figsize=(16, 6))
ax = fig.gca()
plt.ioff()
apaga = open("kalman.csv","w") # apaga o histórico de valores aleatórios
apaga.write("valor\n") # cria coluna 'valor' em nosso csv
apaga.close()

def plota():
    rnd = random.randint(0,1000)
    grava = open("kalman.csv", "a")
    grava.write(str(rnd)+"\n")
    grava.close()
    df = pd.read_csv("kalman.csv")
    x = df.valor[-100:] # Seleciona apenas os últimos 100 valores do dataframe
    cm = x[0:1] 

    cm_seq = np.arange(1,cm, step=150)
    cm_lis = np.asarray(cm_seq)
    cm_com = cm_lis.tolist()
    cm_com = cm_com  + np.asarray(x).tolist()

    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=.1 * np.eye(2))

    states_pred = kf.em(cm_com).smooth(cm_com)[0]

    kf2 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=1 * np.eye(2))

    states_pred2 = kf2.em(cm_com).smooth(cm_com)[0]


    kf3 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance= 2 * np.eye(2))

    states_pred3 = kf3.em(cm_com).smooth(cm_com)[0]


    ax.clear()
    ax.plot(cm_com, ':', label="Random ") 
    ax.plot( 	states_pred[:, 0], label="Convariância =  0.1")
    ax.plot(states_pred2[:, 0], label="Convariância =  1")
    ax.plot(states_pred3[:, 0], label="Convariância = 2")
    ax.legend(loc=2)


while (True):
    plt.pause(0.001)
    plt.ion()
    plota()
