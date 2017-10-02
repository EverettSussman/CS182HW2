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
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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
    """
      Your minimax agent (question 2)
    """

    # def __init__(self, index, depth):
    #   self.index = index
    #   self.depth = depth

    def getAction(self, gameState):
      """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game
      """
      return self.maxValue(gameState)[1]

    def value(self, gameState):
      # IF STATE IS TERMINAL STATE
      if gameState.isWin() or gameState.isLose() or self.depth == 0:
        return self.evaluationFunction(gameState)
      # IF NEXT STATE IS MAX
      elif self.index == 0:
        return self.maxValue(gameState)[0]
      # IF NEXT STATE IS MIN
      else:
        return self.minValue(gameState)[0]

    def maxValue(self, gameState):
      actions = gameState.getLegalActions(self.index)
      newStates = [gameState.generateSuccessor(self.index, action) \
                                                    for action in actions]
      if self.index + 1 == gameState.getNumAgents():
        newAgent = MinimaxAgent(depth=self.depth - 1)
        newAgent.index = 0
      else:
        newAgent = MinimaxAgent(depth=self.depth)
        newAgent.index = self.index + 1
      bestV = -sys.maxint
      for i, state in enumerate(newStates):
        newV = newAgent.value(state)
        if newV > bestV:
          bestV = newV
          bestAction = actions[i]
      return bestV, bestAction

    def minValue(self, gameState):
      actions = gameState.getLegalActions(self.index)
      newStates = [gameState.generateSuccessor(self.index, action) \
                                                    for action in actions]
      if self.index + 1 == gameState.getNumAgents():
        newAgent = MinimaxAgent(depth=self.depth - 1)
        newAgent.index = 0
      else:
        newAgent = MinimaxAgent(depth=self.depth)
        newAgent.index = self.index + 1
      worstV = sys.maxint
      for i, state in enumerate(newStates):
        newV = newAgent.value(state)
        if newV < worstV:
          worstV = newV
          worstAction = actions[i]
      return worstV, worstAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
      """
        Returns the minimax action using self.depth and self.evaluationFunction
      """
      alpha = -sys.maxint
      beta = sys.maxint
      return self.maxValue(gameState, alpha, beta)[1]

    def value(self, gameState, alpha, beta):
      # IF STATE IS TERMINAL STATE
      if gameState.isWin() or gameState.isLose() or self.depth == 0:
        return self.evaluationFunction(gameState)
      # IF NEXT STATE IS MAX
      elif self.index == 0:
        return self.maxValue(gameState, alpha, beta)[0]
      # IF NEXT STATE IS MIN
      else:
        return self.minValue(gameState, alpha, beta)[0]

    def maxValue(self, gameState, alpha, beta):
      actions = gameState.getLegalActions(self.index)
      if self.index + 1 == gameState.getNumAgents():
        newAgent = AlphaBetaAgent(depth=self.depth - 1)
        newAgent.index = 0
      else:
        newAgent = AlphaBetaAgent(depth=self.depth)
        newAgent.index = self.index + 1
      bestV = -sys.maxint
      for i, action in enumerate(actions):
        state = gameState.generateSuccessor(self.index, action)
        newV = newAgent.value(state, alpha, beta)
        if newV > bestV:
          bestV = newV
          bestAction = actions[i]
        if bestV > beta:
          return bestV, bestAction
        alpha = max(alpha, bestV)
      return bestV, bestAction

    def minValue(self, gameState, alpha, beta):
      actions = gameState.getLegalActions(self.index)
      if self.index + 1 == gameState.getNumAgents():
        newAgent = AlphaBetaAgent(depth=self.depth - 1)
        newAgent.index = 0
      else:
        newAgent = AlphaBetaAgent(depth=self.depth)
        newAgent.index = self.index + 1
      worstV = sys.maxint
      for i, action in enumerate(actions):
        state = gameState.generateSuccessor(self.index, action)
        newV = newAgent.value(state, alpha, beta)
        if newV < worstV:
          worstV = newV
          worstAction = actions[i]
        if worstV < alpha:
          return worstV, worstAction
        beta = min(beta, worstV)
      return worstV, worstAction
      

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
      """
      return self.maxValue(gameState)[1]

    def value(self, gameState):
      # IF STATE IS TERMINAL STATE
      if gameState.isWin() or gameState.isLose() or self.depth == 0:
        return self.evaluationFunction(gameState)
      # IF NEXT STATE IS MAX
      elif self.index == 0:
        return self.maxValue(gameState)[0]
      # IF NEXT STATE IS MIN
      else:
        return self.expectMinValue(gameState)

    def maxValue(self, gameState):
      actions = gameState.getLegalActions(self.index)
      newStates = [gameState.generateSuccessor(self.index, action) \
                                                    for action in actions]
      if self.index + 1 == gameState.getNumAgents():
        newAgent = ExpectimaxAgent(depth=self.depth - 1)
        newAgent.index = 0
      else:
        newAgent = ExpectimaxAgent(depth=self.depth)
        newAgent.index = self.index + 1
      bestV = -sys.maxint
      for i, state in enumerate(newStates):
        newV = newAgent.value(state)
        if newV > bestV:
          bestV = newV
          bestAction = actions[i]
      return bestV, bestAction

    def expectMinValue(self, gameState):
      actions = gameState.getLegalActions(self.index)
      newStates = [gameState.generateSuccessor(self.index, action) \
                                                    for action in actions]
      if self.index + 1 == gameState.getNumAgents():
        newAgent = ExpectimaxAgent(depth=self.depth - 1)
        newAgent.index = 0
      else:
        newAgent = ExpectimaxAgent(depth=self.depth)
        newAgent.index = self.index + 1

      expectSum = 0
      for state in newStates:
        expectSum += newAgent.value(state)
      return expectSum/float(len(actions))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

