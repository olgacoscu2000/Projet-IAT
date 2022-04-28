#from curses.ascii import SP
from time import sleep
from controller.agent1 import agent1
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile

def main():

    game = SpaceInvaders(display=True)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)


    gamma= 1.
    alpha = 0.01
    eps_profile = EpsilonProfile(1.0, 0.1)
    n_episodes = 2000
    max_steps = 5000

    controller = agent1(game,eps_profile, gamma, alpha)
    print("On commence!")


    controller.learn(game, n_episodes, max_steps)

    print("on reset")
    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        #sleep(0.0001)

if __name__ == '__main__' :
    main()
