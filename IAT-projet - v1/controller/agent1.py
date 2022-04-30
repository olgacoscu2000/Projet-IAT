from ctypes import sizeof
import pandas as pd
import pygame
import random
import math
from pygame import mixer
import numpy as np
import os
import gym

from game.SpaceInvaders import SpaceInvaders
from epsilon_profile import EpsilonProfile


class agent1():
    def __init__(self, space: SpaceInvaders, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        
        # Initialise la fonction de valeur Q
        #Numéro de l'alien le plus bas
        minInvY=space.get_indavers_Y().index(max(space.invader_Y))
        
        #self.dmax=1000 #math.sqrt(space.screen_height^2+space.screen_width^2)
        #1000/70 = 14.3 => on definit 15 carres.
        dnor=math.sqrt(pow(space.get_indavers_Y()[minInvY], 2)+pow((space.get_indavers_X()[minInvY]-space.get_player_X()), 2))
        self.distance=int(dnor/70)
        #print("paso horizontal", space.playerImage.get_width()+1.7)
        self.direction = int((space.invader_Xchange[minInvY])/abs(int(space.invader_Xchange[minInvY])))

        self.space = space
        self.na = space.na
        
        self.gamma =gamma
        self.alpha=alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial
    
        #distance discrétisée (on prend un peu plus large pour éviter les crash), direction (0 ou 1), et état (4)
        s = (30,2, 4)
        self.Q = np.zeros(s)

        self.qvalues = pd.DataFrame(data={'episode': [], 'value': []})
        self.values = pd.DataFrame(data={'dist': [self.distance], 'dir': [self.direction]})

    def learn(self, env, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.

        :param env: L'environnement 
        :type env: gym.Envselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int

        # Visualisation des données
        Elle doit proposer l'option de stockage de (i) la fonction de valeur & (ii) la Q-valeur 
        dans un fichier de log
        """
        n_steps = np.zeros(n_episodes) + max_steps
        
        # Execute N episodes 
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal, score = env.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                
                if terminal:
                    n_steps[episode] = step + 1  
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)

            # Sauvegarde et affiche les données d'apprentissage
            if n_episodes >= 0:
                state = env.reset()
                print(score)



    def updateQ(self, state, action, reward, next_state):
        """
        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        """print("etat: ",state)
        print("action: ",action)
        print("direction: ", self.direction)
        print("distance: ", self.distance)
        
        print("Q: ", self.Q.shape)
        print("Q[0]: ", self.Q[0])
        print("Q[state][action]: ", self.Q[state][action])"""

        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))


    def select_action(self, state : 'list[int, int]'):
        """
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).

        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state : 'Tuple[int, int]'):
        """
        Cette méthode retourne l'action gourmande.
        :param state: L'état courant
        :return: L'action gourmande
        """
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])
    