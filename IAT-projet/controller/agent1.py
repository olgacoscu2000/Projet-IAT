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
        self.Q = np.zeros([space.state, space.na])

        self.space = space
        self.na = space.na
        
        self.gamma =gamma
        self.alpha=alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        self.qvalues = pd.DataFrame(data={'episode': [], 'value': []})
        self.values = pd.DataFrame(data={'ix': [space.invader_X], 'iy': [space.invader_Y], 'px': [space.player_X]})

    def q_learn(self, env, n_episodes, max_steps):
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
                next_state, reward, terminal = env.step(action)
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
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)]), end =" ")
                self.save_log(env, episode)

        self.values.to_csv('partie_3/visualisation/logV.csv')
        self.qvalues.to_csv('partie_3/visualisation/logQ.csv')



    def updateQ(self, state, action, reward, next_state):
        """
        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
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
    
    def save_log(self, env, episode):
        """Sauvegarde les données d'apprentissage.
        :warning: Vous n'avez pas besoin de comprendre cette méthode
        """
        state = env.reset()
        # Construit la fonction de valeur d'état associée à Q
        V = np.zeros((int(self.space.invader_Y), int(self.space.invader_X-self.space.player_X)))
        for state in self.space.get_state():
            val = self.Q[state][self.select_action(state)]
            V[state] = val

        self.qvalues = self.qvalues.append({'episode': episode, 'value': self.Q[state][self.select_greedy_action(state)]}, ignore_index=True)
        self.values = self.values.append({'episode': episode, 'value': np.reshape(V,(1, self.maze.ny*self.maze.nx))[0]},ignore_index=True)