# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, game_state):
        def minimax(state, agent_index, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agent_index == 0:
                return max_value(state, agent_index, depth)
            else:
                return min_value(state, agent_index, depth)

        def max_value(state, agent_index, depth):
            best_score = float('-inf')
            best_action = None

            for action in state.getAvailableActions(agent_index):
                successor = state.generateNextState(agent_index, action)
                score = minimax(successor, 1, depth)
                if score > best_score:
                    best_score = score
                    best_action = action

            if depth == 0:
                return best_action
            return best_score

        def min_value(state, agent_index, depth):
            best_score = float('inf')
            next_agent = agent_index + 1
            next_depth = depth

            if next_agent == state.getNumAgents():
                next_agent = 0
                next_depth += 1

            for action in state.getAvailableActions(agent_index):
                successor = state.generateNextState(agent_index, action)
                score = minimax(successor, next_agent, next_depth)
                best_score = min(best_score, score)

            return best_score

        return max_value(game_state, 0, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, game_state):
        def minimax(state, agent_index, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent_index == 0:
                return max_value(state, agent_index, depth, alpha, beta)
            else:
                return min_value(state, agent_index, depth, alpha, beta)

        def max_value(state, agent_index, depth, alpha, beta):
            best_score = float('-inf')
            best_action = None

            for action in state.getAvailableActions(agent_index):
                successor = state.generateNextState(agent_index, action)
                score = minimax(successor, 1, depth, alpha, beta)

                if score > best_score:
                    best_score = score
                    best_action = action

                if best_score > beta:
                    return best_score
                alpha = max(alpha, best_score)

            if depth == 0:
                return best_action
            return best_score

        def min_value(state, agent_index, depth, alpha, beta):
            best_score = float('inf')
            next_agent = agent_index + 1
            next_depth = depth

            if next_agent == game_state.getNumAgents():
                next_agent = 0
                next_depth += 1

            for action in state.getAvailableActions(agent_index):
                successor = state.generateNextState(agent_index, action)
                score = minimax(successor, next_agent, next_depth, alpha, beta)

                if score < best_score:
                    best_score = score

                if best_score < alpha:
                    return best_score
                beta = min(beta, best_score)

            return best_score

        return max_value(game_state, 0, 0, float('-inf'), float('inf'))


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, game_state):
        def expectimax(state, agent_index, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent_index == 0:
                return max_value(state, agent_index, depth)
            else:
                return exp_value(state, agent_index, depth)

        def max_value(state, agent_index, depth):
            best_score = float('-inf')
            best_action = None

            for action in state.getAvailableActions(agent_index):
                successor = state.generateNextState(agent_index, action)
                score = expectimax(successor, 1, depth)
                if score > best_score:
                    best_score = score
                    best_action = action\

            if depth == 0:
                return best_action
            return best_score

        def exp_value(state, agent_index, depth):
            actions = state.getAvailableActions(agent_index)
            if not actions:
                return self.evaluationFunction(state)

            total = 0
            probability = 1.0 / len(actions)

            next_agent = agent_index + 1
            next_depth = depth
            if next_agent == game_state.getNumAgents():
                next_agent = 0
                next_depth += 1

            for action in actions:
                successor = state.generateNextState(agent_index, action)
                score = expectimax(successor, next_agent, next_depth)
                total += score * probability

            return total

        return max_value(game_state, 0, 0)


def betterEvaluationFunction(currentGameState):
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    scared_times = [ghost.scaredTimer for ghost in ghost_states]

    score = currentGameState.getScore()

    food_count = len(food_list)
    food_score = -5 * food_count

    capsule_score = -20 * len(capsules)

    if food_list:
        closest_food_dist = min(util.manhattanDistance(pacman_pos, food) for food in food_list)
        food_distance_score = -1.5 * closest_food_dist
    else:
        food_distance_score = 0

    ghost_penalty = 0
    for i, ghost in enumerate(ghost_states):
        ghost_pos = ghost.getPosition()
        dist = util.manhattanDistance(pacman_pos, ghost_pos)
        if scared_times[i] == 0:
            if dist < 2:
                ghost_penalty -= 200
            elif dist < 4:
                ghost_penalty -= 50
        else:
            if dist > 0:
                ghost_penalty += 30 / dist

    total = (
        score
        + food_score
        + capsule_score
        + food_distance_score
        + ghost_penalty
    )

    return total

# Abbreviation
better = betterEvaluationFunction
