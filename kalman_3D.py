#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.spatial import distance


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = []
y = []
z = []


def init():
    recompensa = []

    x0 = 1
    y0 = 1
    z0 = 1
    history_x = []
    history_y = []
    history_z = []
    boas = []
    ruins = []
    reco = []
    actions = []
    recompensas = []
    t = 0

    def prediz(x0, y0, z0):
        alvo = [random.randint(0, 15), random.randint(0, 15), random.randint(0, 15)]
        x_a = alvo[0]
        y_a = alvo[1]
        z_a = alvo[2]
        ax.scatter(x_a, y_a, z_a, marker='o', color='black')
        volta = 0
        t = 0
        z0_ = z0
        y0_ = y0
        x0_ = x0
        while (True):

            action_ = random.randint(0, 5)
            if action_ == 0:
                z0_ = z0 + 1.
            if action_ == 1:
                y0_ = y0 + 1.
            if action_ == 2:
                x0_ = x0 + 1.
            if action_ == 3:
                x0_ = x0 - 1.
            if action_ == 4:
                y0_ = y0 - 1.
            if action_ == 5:
                z0_ = z0 - 1.

            pos_ = [x0_, y0_, z0_]
            reward_base = distance.euclidean(pos_, alvo)
            recompensa_valor = -reward_base
            recompensa.append(recompensa_valor)
            recompensa_ar = np.asarray(recompensa)

            try:

                if recompensa_ar[-1:] == np.mean(recompensa_ar[-2:]) and recompensa_ar[-1:] > -5:
                    history_x.append(x0)
                    history_y.append(y0)
                    history_z.append(z0)
                    history_x_ = np.asarray(history_x)[-20:]
                    history_y_ = np.asarray(history_y)[-20:]
                    history_z_ = np.asarray(history_z)[-20:]

                    j = len(history_y_) + 1  # valor do índice da predição

                    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                                      transition_covariance=.001 * np.eye(2))

                    kf2 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                                       transition_covariance=.001 * np.eye(2))

                    kf3 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                                       transition_covariance=.001 * np.eye(2))

                    x = history_x_
                    z = history_z_
                    y = history_y_

                    cm = np.asarray(x)[0]
                    cm2 = np.asarray(z)[0]
                    cm3 = np.asarray(y)[0]

                    cm_seq = np.arange(1, cm, step=3000)
                    cm_seq2 = np.arange(1, cm2, step=3000)
                    cm_seq3 = np.arange(1, cm3, step=3000)

                    cm_lis = np.asarray(cm_seq)
                    cm_lis2 = np.asarray(cm_seq2)
                    cm_lis3 = np.asarray(cm_seq3)

                    cm_com = cm_lis.tolist()
                    cm_com2 = cm_lis2.tolist()
                    cm_com3 = cm_lis3.tolist()

                    cm_com = cm_com + np.asarray(history_x_).tolist()
                    cm_com2 = cm_com2 + np.asarray(history_z_).tolist()
                    cm_com3 = cm_com3 + np.asarray(history_y_).tolist()

                    pos = [x0, y0, z0]  # posição inicial
                    states_pred_x = kf.em(cm_com).smooth(cm_com)[0]
                    states_pred_z = kf2.em(cm_com2).smooth(cm_com2)[0]
                    states_pred_y = kf3.em(cm_com3).smooth(cm_com3)[0]
                    ax.scatter(states_pred_x[:, 0].mean(), states_pred_y[:, 0].mean(), states_pred_z[:, 0].mean(),
                               marker='^', color='blue', alpha=0.5)

                    agora = (states_pred_x[:, 0].mean(), states_pred_y[:, 0].mean(), states_pred_z[:, 0].mean())

                    reward_base2 = distance.euclidean(agora, alvo)
                    recompensa2 = -reward_base2

                    if recompensa2 >= - 1.5:
                        prediz(x0,y0,z0)

                if recompensa_ar[-1:] > recompensa_ar[-2:-1]:
                    action = action_
                    if action == 0:
                        z0 = z0 + 1.
                    if action == 1:
                        y0 = y0 + 1.
                    if action == 2:
                        x0 = x0 + 1.
                    if action == 3:
                        x0 = x0 - 1.
                    if action == 4:
                        y0 = y0 - 1.
                    if action == 5:
                        z0 = z0 - 1.
                    history_x.append(x0)
                    history_y.append(y0)
                    history_z.append(z0)
                    reco.append(recompensa_ar[-1:])
                    history_x_ = np.asarray(history_x)[-20:]
                    history_y_ = np.asarray(history_y)[-20:]
                    history_z_ = np.asarray(history_z)[-20:]
                    boas_ = (x0, y0, z0)
                    boas.append(boas_)
                    actions.append(action)
                    recompensas.append(recompensa_ar[-1:])
                    grava()

                if recompensa_ar[-1:] < np.mean(recompensa_ar[-2:]):

                    ruins_ = (x0_, y0_, z0_)
                    ruins.append(ruins_)
                    ax.scatter(x0_, y0_, z0_, marker='o', color='red', alpha=0.4)
                    pass

            except:
                pass

            def grava():


                kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                                  transition_covariance=.009 * np.eye(2))

                kf2 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                                   transition_covariance=.009 * np.eye(2))

                kf3 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                                   transition_covariance=.009 * np.eye(2))

                x = history_x_
                z = history_z_
                y = history_y_

                cm = np.asarray(x)[0]
                cm2 = np.asarray(z)[0]
                cm3 = np.asarray(y)[0]

                cm_seq = np.arange(1, cm, step=150)
                cm_seq2 = np.arange(1, cm2, step=150)
                cm_seq3 = np.arange(1, cm3, step=150)

                cm_lis = np.asarray(cm_seq)
                cm_lis2 = np.asarray(cm_seq2)
                cm_lis3 = np.asarray(cm_seq3)

                cm_com = cm_lis.tolist()
                cm_com2 = cm_lis2.tolist()
                cm_com3 = cm_lis3.tolist()

                cm_com = cm_com + np.asarray(history_x_).tolist()
                cm_com2 = cm_com2 + np.asarray(history_z_).tolist()
                cm_com3 = cm_com3 + np.asarray(history_y_).tolist()

                pos = [x0, y0, z0]  # posição inicial
                states_pred_x = kf.em(cm_com).smooth(cm_com)[0]
                states_pred_z = kf2.em(cm_com2).smooth(cm_com2)[0]
                states_pred_y = kf3.em(cm_com3).smooth(cm_com3)[0]


                ax.scatter(pos[0], pos[1], pos[2], marker='o', color='green', label="Original", alpha=0.4)
                ax.scatter(states_pred_x[:, 0].mean(), states_pred_y[:, 0].mean(), states_pred_z[:, 0].mean(),
                           marker='^', color='blue', alpha=1)


            volta += 1
            if len(history_x) > 5000:
                ax.clear()
            plt.pause(0.001)
            plt.ion()

    prediz(x0, y0, z0)


while (True):
    init()
