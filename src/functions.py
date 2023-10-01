
# Importando bibliotecas
from scipy.integrate import odeint
from numpy import zeros, arange, pi, sin
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import numpy as np
import time
import os
import pickle


#Variáveis globais
theta_modelo = arange(0, 30.1, 0.1)
theta_modelo = list(map(lambda x: round(x, 1), theta_modelo))
recompensas = {}

#Setup referência
print("\nInsira ângulo inteiro de referência (em graus):")
ref_aux = int(input('Referência:'))

#Verifica se tabela de q-values já existe
file_path = f'tabelas/q_values-{ref_aux}-graus.pkl'
if os.path.exists(file_path):
    with open(file_path, 'rb') as file:
        q_values = pickle.load(file)
    print("Q_Values carregados com sucesso!!")
else:
    q_values = np.zeros((len(theta_modelo), 2))
    print("Criada nova tabela de Q_Values!!")

#Dinâmica do aeropêndulo
def din_aeropend(y, t, omega2):
    # Parametros da planta
    b = 0.006856*5
    m = 0.3182
    g = 9.81
    I = 0.0264
    kh = 2.12829e-5
    Lh = 0.32

    # Definindo estados
    x1, x2 = y
    # Dinamica do pendulo
    x1p = x2
    x2p = (Lh*kh/I)*omega2 - (Lh*m*g/I)*sin(x1) - (b/I)*x2
    return [x1p, x2p]

#Função para definir recompensas de acordo com a referência
def def_recompensas(ref):
    for i in theta_modelo:
        if i >= ref and i <= (ref+0.5):
                recompensas[i] = 1000
        elif i == (30):
                recompensas[i] = -100
        elif i == (0):
                recompensas[i] = -100
        else:
                recompensas[i] = 1

# Função que verifica se estado terminal foi alcançado (estados terminais = 0, 30, target)
def estado_terminal(theta_atual):
    if theta_atual > -1 or theta_atual < 30:
        return False
    else:
        return True

# Função que escolhe a ação a ser tomada utilizando uma política epsilon greedy
def e_greedy(theta_atual, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[theta_atual])
    else:
        return np.random.randint(2)

# Função que implementa o incremento na velocidade de acordo com a ação escolhida
def acao(acao, theta, aceleracao, ref):
    diff = abs(theta - ref)
    diff2 = theta - ref
    if theta == 0:
        aceleracao = aceleracao + 1000
    else:
        if diff > 5:
            if acao == 'A':
                aceleracao = aceleracao + 1500
            else:
                aceleracao = aceleracao - 1500
        elif 4 <= diff <= 5:
            if acao == 'A':
                aceleracao = aceleracao + 1200
            else:
                aceleracao = aceleracao - 1200
        elif 3 <= diff < 4:
            if acao == 'A':
                aceleracao = aceleracao + 950
            else:
                aceleracao = aceleracao - 950      
        elif 2.5 <= diff < 3:
            if acao == 'A':
                aceleracao = aceleracao + 800
            else:
                aceleracao = aceleracao - 800
        elif 2 <= diff < 2.5:
            if acao == 'A':
                aceleracao = aceleracao + 700
            else:
                aceleracao = aceleracao - 700
        elif 1.7 <= diff < 2:
            if acao == 'A':
                aceleracao = aceleracao + 600
            else:
                aceleracao = aceleracao - 600
        elif 1.5 <= diff < 1.7:
            if acao == 'A':
                aceleracao = aceleracao + 460
            else:
                aceleracao = aceleracao - 460
        elif 1.3 <= diff < 1.5:
            if acao == 'A':
                aceleracao = aceleracao + 380
            else:
                aceleracao = aceleracao - 380
        elif 1 <= diff < 1.3:
            if acao == 'A':
                aceleracao = aceleracao + 360
            else:
                aceleracao = aceleracao - 360
        elif 0.8 <= diff < 1:
            if acao == 'A':
                aceleracao = aceleracao + 340
            else:
                aceleracao = aceleracao - 340  
        elif 0.6 <= diff < 0.8:
            if acao == 'A':
                aceleracao = aceleracao + 320
            else:
                aceleracao = aceleracao - 320
        elif 0.3 <= diff < 0.6:
            if acao == 'A':
                aceleracao = aceleracao + 160
            else:
                aceleracao = aceleracao - 160
        elif 0 <= diff < 0.3:
            if acao == 'A':
                aceleracao = aceleracao + 1.5
            else:
                aceleracao = aceleracao - 1.5
    return aceleracao

#Função que implementa Qlearning para realizar incremento da aceleração
def controle(theta_t1, theta_t2, aceleracao, ref, p, k, i):
    #####Definição dos parâmetros
 

    epsilon = p # parâmetro greedy
    gamma = k # fator de desconto
    alpha = i  # taxa de aprendizagem
    aceleracoes = []
    acoes = ['A', 'D']
    
    start = time.time()
    # Se não for um estado terminal o incremento é realizado
    if not estado_terminal(theta_t1):
        # Escolhe uma ação utilizando epsilon greedy

        if theta_t2 >= 30:
            theta_t2 = 30.0
        if theta_t1 >= 30:
            theta_t1 = 30.0   

        theta_t1_idx = theta_modelo.index(theta_t1)
        
        theta_t2_idx = theta_modelo.index(theta_t2)

        action_index = e_greedy(theta_t1_idx, epsilon)
      
        # Salva os valores antigos de theta e omega
        
        # Incremento da velocidade de acordo com a ação escolhida
        aceleracao = acao(acoes[action_index], theta_t1, aceleracao, ref)
        
        # Afere recompensa para o estado e calcula o q_value
        recompensa = recompensas[theta_t2]
        q_value_antigo = q_values[theta_t1_idx, action_index]

        # Calcula a diferença temporal
        temporal_difference = recompensa + (gamma*np.max(q_values[theta_t2_idx]) - q_value_antigo)

        # Calcula novo q_value e o aloca na matriz de q_values
        q_value_novo = q_value_antigo + (alpha*temporal_difference)
        
        q_values[theta_t1_idx, action_index] = q_value_novo
        
        # print(f"Ângulo: {theta}")
        aceleracoes.append(aceleracao)
        
        end = time.time()
    
    return aceleracao

#Função que simula o comportamento da planta
def planta(ref, p, k, i, episodios, eps):

    # Parametros de simulacao
    Ta = 0.1  # Periodo de amostragem
    Tsim = 30.0  # tempo de simulacao
    kend = int(Tsim/Ta)
    

    omega2 = zeros(kend)  # rad/s
    theta = zeros(kend)  # posicao angular
    thetap = zeros(kend)  # velocidade angular

    rad2deg = 180/pi
    deg2rad = pi/180
    aceleracao = 0
 

    for z in range(kend-1):
        omega2[z] = aceleracao

        # # Evoluindo a din. da planta
        x0 = [theta[z], thetap[z]]   # condicao inicial
        sol = odeint(din_aeropend, x0, [0.0, Ta], args=(omega2[z],))
        theta[z + 1] = sol[:, 0][-1]
        thetap[z + 1] = sol[:, 1][-1]

        thetaux_t1 = round((theta[z]*rad2deg), 1)
        if thetaux_t1 < 0:
            thetaux_t1 = 0

        thetaux_t2 = round((theta[z+1]*rad2deg), 1)
        if thetaux_t2 < 0:
            thetaux_t2 = 0

        #incremento da aceleração utilizando Qlearning
        aceleracao = controle(thetaux_t1, thetaux_t2, aceleracao, ref, p, k, i)

    if episodios == eps:
        # Plotando resultados
        fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 7))

        ax1.plot(arange(0, Tsim, Ta), theta*180/3.14, lw=2, label=r'$\theta$ (deg)')
        ax1.set(xlabel='Tempo (s)')
        ax1.legend()
        ax1.grid(True)

        ax1.yaxis.set_major_locator(MultipleLocator(3))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))
        ax1.xaxis.set_major_locator(MultipleLocator(2))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))

        # fig = plt.figure()
        ax2.plot(arange(0, Tsim-Ta, Ta), omega2[0:-1], 'r--', lw=2, label=r'$\omega^2$ (rad$^2$/s$^2$)')
        ax2.set(xlabel='Tempo (s)')
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MultipleLocator(2))
        ax2.yaxis.set_minor_locator(MultipleLocator(10000))
        plt.savefig(f'img/theta-p-{ref_aux}-graus.png', dpi=300)
    
        fig3 = plt.figure()
        plt.plot(arange(0, Tsim-Ta, Ta),
                thetap[0:-1]*180/3.14, 'r--', lw=2, label=r'$\ddot{\theta} (rad/s^2)$')
        plt.xlabel('Tempo (s)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'img/omega2-{ref_aux}-graus.png', dpi=300)
        plot_q(ref_aux)
        plt.show(block=True)

    return round(theta[-2]*rad2deg, 1), round(aceleracao, 1)
    return theta, omega2

#Função para plotar q_values
def plot_q(ref_aux):
    
    plt.figure(figsize = (8, 6))

    plt.plot(arange(0, 30, 0.1),
             q_values[0:-1, 0])
    x = np.array(arange(0, 30, 0.1))
    y = np.array([
    q_values[0:-1, 0],
    q_values[0:-1, 1]])
    plt.title("Q_value x Ângulo")
    plt.xlabel("Ângulo em graus")
    plt.ylabel("Q value")

    # for i, array in enumerate(y):
    plt.plot(x, y[0], color = '#0d298f' , marker = ".", label = f"Ação = Acelerar")
    plt.plot(x, y[1], color = '#820101', marker = ".", label = f"Ação = Desacelerar")
        
    plt.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'img/q-values-{ref_aux}-eps.png', dpi=300)
    # plt.show(block=True)

#Função que define hiperparâmetros e inicia treinamento
def treino(eps):
    ref = ref_aux
    param1 = [0.99]
    param2 = [0.4]
    param3 = [0.99]
    Ta = 0.02  # Periodo de amostragem
    Tsim = 30.04  # tempo de simulacao
    kend = arange(0, Tsim-Ta, Ta)

    param1_angulo = np.zeros((len(kend), 4))
    param1_omega2 = np.zeros((len(kend), 4))
    # param2_angulo = np.zeros((len(kend), 4))
    # param2_omega2 = np.zeros((len(kend), 4))
    # param3_angulo = np.zeros((len(kend), 4))
    # param3_omega2 = np.zeros((len(kend), 4))
    index=0
    for p in param1:
        for k in param2:
            for i in param3:
                def_recompensas(ref)
                episodios = 0
                while episodios < eps:
                    angulo, omega2 = planta(ref, p, k, i, episodios, eps)
                    episodios += 1
                    print(episodios)
                param1_angulo[:-1, index] = angulo
                param1_omega2[:-1, index] = omega2
                index += 1
                print(f"epsilon:{p}, gamma:{k}, alpha:{i}")
    param1_angulo = param1_angulo[:, :1] 
    param1_omega2 = param1_omega2[:, :1]

    # plt.figure(figsize = (8, 6))

    # fig, (ax1, ax2) = plt.subplots(2)
    # x1 = np.array(kend[0:-1])
    # y1 = np.array([
    # param1_angulo[0:-1, 0]])
    # # param1_angulo[0:-1, 1]*180/3.14])
    # # param1_angulo[0:-1, 2]*180/3.14])
    # # param1_angulo[0:-1, 3]*180/3.14,]
    # # param1_angulo[0:-1, 4]*180/3.14,
    # # param1_angulo[0:-1, 5]*180/3.14])
    # # param1_angulo[0:-1, 6]*180/3.14,
    # # param1_angulo[0:-1, 7]*180/3.14,
    # # param1_angulo[0:-1, 8]*180/3.14])
    # ax1.set(xlabel='Tempo (s)')
    # ax1.set(ylabel=r'$\theta$')


    # # index2 = 0
    # # colors = ['#0d298f', '#ff0000', '#00ff00', '#0000ff', '#1bf7c8', '#820101', '#baad1a', '#eb7d34']
    # # for i in param1:
    # #     for k in param2:
    # #         ax1.plot(x1, y1[index2], color=colors[index2 % len(colors)], marker = ",", label = r'$\epsilon=%s, \gamma=%s$' % (str(i), str(k)))
    # # for i, array in enumerate(y):



    # ax1.plot(x1, y1[0], color = '#0d298f' , marker = ",", label = r'$\epsilon=0.9, \gamma=0.4, \alpha=0.9$') 
    # # ax1.plot(x1, y1[1], color = '#820101', marker = ",", label = r'$\epsilon=0.9, \gamma=0.4, \alpha=0.8$') 
    # # ax1.plot(x1, y1[2], color = '#baad1a', marker = ",", label = r'$\epsilon=0.9, \gamma=0.4, \alpha=0.9$') 
    # # ax1.plot(x1, y1[3], color = '#eb7d34', marker = ",", label = r'$\epsilon=0.9, \gamma=0.5, \alpha=0.5$') 
    # # ax1.plot(x1, y1[4], color = '#a3990b' , marker = ",", label = r'$\epsilon=0.5, \gamma=0.5$') 
    # # ax1.plot(x1, y1[5], color = '#ff0000', marker = ",", label = r'$\epsilon=0.5, \gamma=0.75$') 
    # # ax1.plot(x1, y1[6], color = '#0000ff', marker = ",", label = r'$\epsilon=0.75, \gamma=0.25$') 
    # # ax1.plot(x1, y1[7], color = '#00ff00', marker = ",", label = r'$\epsilon=0.75, \gamma=0.5$') 
    # # ax1.plot(x1, y1[8], color = '#eb7d34', marker = ",", label = r'$\epsilon=0.75, \gamma=0.75$') 
    # ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax1.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    # ax1.grid(True)

    # x2 = np.array(kend[0:-1])
    # y2 = np.array([
    # param1_omega2[0:-1, 0]*1/1.00])
    # # param1_omega2[0:-1, 1]*1.00,
    # # param1_omega2[0:-1, 2]*1.00])
    # # param1_omega2[0:-1, 3]*1.00])
    # # param1_omega2[0:-1, 4],
    # # param1_omega2[0:-1, 5]])
    # # param1_omega2[0:-1, 6],
    # # param1_omega2[0:-1, 7],
    # # param1_omega2[0:-1, 8],])
    # ax2.set(xlabel='Tempo (s)')
    # ax2.set(ylabel=r'$\omega^2$')

    # # for i, array in enumerate(y):
    # ax2.plot(x2, y2[0], color = '#0d298f' , marker = ",", label = r'$\epsilon=0.9, \gamma=0.4, \alpha=0.9$')
    # # ax2.plot(x2, y2[1], color = '#820101', marker = ",", label = r'$\epsilon=0.9, \gamma=0.4, \alpha=0.8$')
    # # ax2.plot(x2, y2[2], color = '#baad1a', marker = ",", label = r'$\epsilon=0.9, \gamma=0.4, \alpha=0.9$')
    # # ax2.plot(x2, y2[3], color = '#eb7d34', marker = ",", label = r'$\epsilon=0.9, \gamma=0.5, \alpha=0.9$')
    # # ax2.plot(x2, y2[4], color = '#a3990b' , marker = ",", label = r'$\epsilon=0.5, \gamma=0.5$')
    # # ax2.plot(x2, y2[5], color = '#ff0000', marker = ",", label = r'$\epsilon=0.5, \gamma=0.75$')
    # # ax2.plot(x2, y2[6], color = '#0000ff', marker = ",", label = r'$\epsilon=0.75, \gamma=0.25$')
    # # ax2.plot(x2, y2[7], color = '#00ff00', marker = ",", label = r'$\epsilon=0.75, \gamma=0.5$')
    # # ax2.plot(x2, y2[8], color = '#eb7d34', marker = ",", label = r'$\epsilon=0.75, \gamma=0.75$')
    # ax2.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    # ax2.grid(True)



    # plt.tight_layout()
    # plt.savefig(f'treino-final-{episodios-1}.png', dpi=300)

    # plt.show(block=True)


    with open(file_path, 'wb') as file:
        pickle.dump(q_values, file)

