import gymnasium as gym
from collections import defaultdict
import numpy as np
import pickle as pk
import os
import queue
import random
import csv
env = gym.make("ALE/Tetris-v5", render_mode="human")
#rgb_array
fitness = 0

shapes = []
line = [[True, True, True, True]]
vertLine = list(zip(*line[::-1]))
tShape =   [[False,True, False],
            [True, True, True]]
tShape90 = list(zip(*tShape[::-1]))
tShape180 = list(zip(*tShape90[::-1]))
tShape270 = list(zip(*tShape180[::-1]))
lShape =   [[False, False, True],
            [ True, True, True]]
lShape90 = list(zip(*lShape[::-1]))
lShape180 = list(zip(*lShape90[::-1]))
lShape270 = list(zip(*lShape180[::-1]))
jShape =   [[ True, True, True],
            [ False, False, True]]
jShape90 = list(zip(*jShape[::-1]))
jShape180 = list(zip(*jShape90[::-1]))
jShape270 = list(zip(*jShape180[::-1]))
zShape =   [[True,True, False],
            [False, True, True]]
reverseZShape = list(zip(*zShape[::-1]))
sShape =   [[False,True, True],
            [True, True, False]]
reverseSShape = list(zip(*sShape[::-1]))
square = [[True, True], [True, True]]
shapes.append(line)
shapes.append(vertLine)
shapes.append(sShape)
shapes.append(reverseSShape)
shapes.append(zShape)
shapes.append(reverseZShape)
shapes.append(square)
shapes.append(tShape)
shapes.append(tShape90)
shapes.append(tShape180)
shapes.append(tShape270)
shapes.append(lShape)
shapes.append(lShape90)
shapes.append(lShape180)
shapes.append(lShape270)
shapes.append(jShape)
shapes.append(jShape90)
shapes.append(jShape180)
shapes.append(jShape270)

numBlocksForEpisode = 0
blockAtTop = False


def getFirstColAtTop(bs):
    farLeft = 100
    for i in range(0,10):
        for j in range(0,5):
            if bs[j][i] == True:
                farLeft = min(farLeft, i)
    return farLeft

def findPeaks(bs):
    peaks = [0,0,0,0,0,0,0,0,0,0]
    peaksWithoutTop = [0,0,0,0,0,0,0,0,0,0]

    for i in range(0,10):
        for j in range(0,22):
            if bs[j][i] == True:
                if peaks[i] == 0:
                    peaks[i] = 22 - j
                if j > 4 and peaksWithoutTop[i] == 0:
                    peaksWithoutTop[i] = 22 - j


    return peaks, peaksWithoutTop

def newBlock(bs):
    for shapeindex in range(0,19):
        s = np.array(shapes[shapeindex])
        for i in range(0, 10):
            for j in range(0, 5):
                tot = 0
                
                for x in range(0,(s.shape[0])):
                    for y in range(0,(s.shape[1])):
                        if j + x > 4 or i + y > 9:
                            continue
                        if bs[j+x][i+y] == s[x][y]:
                            tot += 1
                
                if tot == s.shape[0] * s.shape[1]:
                    # print(shapes[shapeindex])
                    return shapeindex
    return -1

def getProjections(bs, ShapeIndex, previousProjections, currCol, bsPeaks):
    projs = previousProjections
    c = [row[:] for row in bs]

    s = np.array(shapes[ShapeIndex])
    for j in range(21, 5, -1):
        tot = 0
        len = s.shape
        if j + s.shape[0] > 22 or currCol + s.shape[1] > 10:
            continue
        for x in range(0,(s.shape[0])):
            for y in range(0,(s.shape[1])):
                if s[x][y] == False:
                    tot += 1
                elif s[x][y] == True and bs[j+x][currCol+y] == False:
                    if(bsPeaks[currCol+y] < (22 - j + x)):
                        tot += 1
        if tot == s.shape[0] * s.shape[1]:
            for x in range(0,(s.shape[0])):
                for y in range(0,(s.shape[1])):
                    if s[x][y] == True:
                        c[j+x][currCol+y] = True
            projs.append(c)
            if currCol == 0:
                return projs
            return getProjections(bs, ShapeIndex, projs, currCol - 1, bsPeaks)
    if currCol == 0:
        return projs
    return getProjections(bs, ShapeIndex, projs, currCol - 1, bsPeaks)

def GetRewardFromBoard(bs, peaks, ls,  prevFitness):
    height = 0
    lines = ls
    holes = 0
    bumpiness = 0

    for j in range(4,22):
        lineTest = 0
        for i in range(0,10):
            if bs[j][i] == True:
                lineTest += 1
        if lineTest == 10: 
            lines += 1
    for i in range(0,10):
        # if i != 0 and peaks[i] < 21 and peaks[i-1] < 21:
        #     bumpiness += abs(min(peaks[i],15) - min(peaks[i-1],15))
        if i != 9 and peaks[i] < 21 and peaks[i+1] < 21:
            bumpiness += abs(min(peaks[i],15) - min(peaks[i+1],15))
        colHeight = 0
        for j in range(4,22):
            # if 22 - j < peaks[i] and peaks[i] < 15 and bs[j][i] == False:
            if 21 - peaks[i] < j and bs[j][i] == False:
                holes += 1 
            if colHeight == 0 and bs[j][i] == True:
                colHeight = 22 - j

        height = max(colHeight, height)

    newFitness = (-.51 * height) + (-0.36 * holes) + (0.76 * lines) + (.2  * numBlocksForEpisode) + (-0.18 * bumpiness)#+ (-0.36 * holes)) #+ (-0.18 * bumpiness)
    returnReward = newFitness - prevFitness
    return returnReward, newFitness

class TetrisAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        vstobeadded : any,
        discount_factor: float = 0.9,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        # self.q_values = defaultdict(lambda: np.zeros([10, 4]))
        self.v_values = dict()


        if vstobeadded:
            for k,v in vstobeadded.items():
                self.v_values[k] = v

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def loadVValues(self, v):
        self.v_values = v

    def get_action(self, input):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """

        s = input[0]
        s_primes_rots = input[1]
        rewards = input[2]

        sp_tuples_array = []
        for rot in s_primes_rots:
            grids = []
            for sp in rot:
                rows = []
                for j in sp:
                    rows.append(tuple(j))
                grids.append(tuple(rows))
            sp_tuples_array.append(tuple(grids))

        sp_tuples = tuple(sp_tuples_array)

        # if np.random.random() < self.epsilon:
        if 100 < self.epsilon:
            maxRot = 0
            maxColIndex = 0
            # if len(s_primes_rots[maxRot]) == 0 or len(sp_tuples[maxRot]) == 0:
            for mr in s_primes_rots:
                if len(mr) > 0:
                    maxColIndex = np.random.randint(len(mr))
                    maxRot = s_primes_rots.index(mr)
                    break
            
            # maxColIndex = np.random.randint(1, len(s_primes_rots[maxRot]))
            # if len(s_primes_rots[maxRot]) == 0 or len(sp_tuples[maxRot]) == 0:
            #     for mr in s_primes_rots:
            #         if len(mr) > 0:
            #             np.random.randint(len(mr) -1)
            #             break
        else:
            held_vs = []
            for rot in sp_tuples:
                held_vs_rot = []
                for sp in rot:
                    if sp in self.v_values:
                        held_vs_rot.append(self.v_values[sp])
                    else:
                        held_vs_rot.append(0)
                held_vs.append(held_vs_rot)

            final_vals = []
            for rot in range(len(rewards)):
                temp = []
                for j in range(len(rewards[rot])):
                    temp.append(rewards[rot][j] + held_vs[rot][j])
                final_vals.append(tuple(temp))

            maxRots = []
            indexesOfMaxRots = []
            for rot in final_vals:
                if not rot:
                    maxRots.append(-1000)
                    indexesOfMaxRots.append(0)
                    continue
                temp = max(rot)
                maxRots.append(temp)
                indexesOfMaxRots.append(rot.index(temp))
            maxRot = maxRots.index(max(maxRots))
            maxColIndex = indexesOfMaxRots[maxRot]

        
        maxState = sp_tuples[maxRot][maxColIndex]
        maxReward = rewards[maxRot][maxColIndex]
        return maxRot, maxColIndex, maxState, maxReward
            

    def update(
        self,
        state,
        chosenState,
        reward
    ):
        stateArrayofTuples = []
        for row in state:
            stateArrayofTuples.append(tuple(row))
        stateTuples = tuple(stateArrayofTuples)

        future_v_value = 1
        if chosenState in self.v_values:
            future_v_value = (not terminated) * np.max(self.v_values[chosenState])
        else:
            self.v_values[chosenState] = 0
        
        if stateTuples not in self.v_values:
            self.v_values[stateTuples] = 0

        temporal_difference = (
            reward + self.discount_factor * future_v_value - self.v_values[stateTuples]
        )

        self.v_values[chosenState] = (
            self.v_values[chosenState]  + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
        

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)



previousBlockState = 0



def getBlockSpace(obs):
    rows = 175
    cols = 41
    playSpace = []
    for i in range(rows):
        row = []
        for j in range(cols):
            # print(observation[27 + i][23 + j])
            row.append(observation[27 + i][23 + j])
        playSpace.append(row)

    bs = []

    for i in range(22):
        temp = ""
        blockRow = []
        for j in range(10):
            if playSpace[i * 8 + 1][j * 4 + 1][0] != 111:
                temp += "O"
                blockRow.append(True)
            else:
                temp += " "
                blockRow.append(False)
        bs.append(blockRow)
        # print(temp)
    # print("--------break---------")
    # print(bs)
    return bs

# hyperparameters
learning_rate = 0.04
n_episodes = 500
start_epsilon = .5
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

temp_q_values = 0

file_path = 'final.pkl'

v_values = []
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    # Load Q-values from file
    with open(file_path, 'rb') as f:
        v_values = pk.load(f)


agent = TetrisAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    vstobeadded=v_values
)



#MAIN
# MODE = "AI"
MODE = "PROJ"
for episode in range(n_episodes):
    observation, info = env.reset()
    
    episodelines = 0
    episodeFinalFitness = 0

    terminated = False
    truncated = False
    numBlocksForEpisode = 0

    blockSpace = getBlockSpace(observation)
    peaks, peaksWithoutTop = findPeaks(blockSpace)
    block = newBlock(blockSpace)

    peaksBeforeLastMoveQueue = [0,0,0,0,0,0,0,0,0,0]
    colAndRotFromLastMoveQueue = [0,0]
    rewardRightBeforeLastMoveQueue = 0
    
    moveQueue = queue.Queue(100)

    observation, reward, terminated, truncated, info = env.step(0) # initial render
    # observation, reward, terminated, truncated, info = env.step(0) # initial renders
    cooldown = 0
    while not terminated or truncated:
        # Get current action
        # action = agent.get_action([peaks, block])
        
        
        
        
        # run action and get results back
        for i in range(4): # four frame skip
            action = 0
            if not moveQueue.empty():
                action = moveQueue.get()
            observation, reward, terminated, truncated, info = env.step(0)
            episodelines += reward
            observation, reward, terminated, truncated, info = env.step(action)
            episodelines += reward

            # get raw data
            currBlockSpace = getBlockSpace(observation)
            currPeaks, currPeaksWithoutTop = findPeaks(currBlockSpace)
            currBlock = block

            # find block at the top
            if blockAtTop == False:
                if cooldown <= 0 and max(currPeaks) >= 20:
                    numBlocksForEpisode += 1
                    currBlock = newBlock(currBlockSpace)
                    blockAtTop = True
                    cooldown = 10
                    break
                else:
                    cooldown -= 1
            else:
                # if max(currPeaks) < 20:
                blockAtTop = False
            

        # if info["frame_number"] % 4 == 0:
        
        # if new block, set grouped actions
        if blockAtTop and not (terminated or truncated):
            if MODE == "PROJ":
                if not moveQueue.empty():
                    moveQueue.get()

                # select any needed rotations
                rots = []
                unrotatedBlock = -1
                if currBlock <= -1 or currBlock >= 19:
                    continue
                if currBlock == 0 or currBlock == 1: # line
                    unrotatedBlock = 0
                    rots.append(0)
                    rots.append(1)
                if currBlock == 2 or currBlock == 3: # s
                    unrotatedBlock = 2
                    rots.append(2)
                    rots.append(3)
                if currBlock == 4 or currBlock == 5 : # z
                    unrotatedBlock = 4
                    rots.append(4)
                    rots.append(5)
                if currBlock == 6 : # square
                    unrotatedBlock = 6
                    rots.append(6)
                if currBlock == 7 or currBlock == 8 or currBlock == 9 or currBlock == 10: # t
                    unrotatedBlock = 7
                    rots.append(7)
                    rots.append(8)
                    rots.append(9)
                    rots.append(10)
                if currBlock == 11 or currBlock == 12 or currBlock == 13 or currBlock == 14: # l
                    unrotatedBlock = 11
                    rots.append(11)
                    rots.append(12)
                    rots.append(13)
                    rots.append(14)
                if currBlock == 15 or currBlock == 16 or currBlock == 17 or currBlock == 18: # j
                    unrotatedBlock = 15
                    rots.append(15)
                    rots.append(16)
                    rots.append(17)
                    rots.append(18)

                # will hold the projections, rewards, and fitnesses for each rot
                eachRotation = []

                for rotation in rots:
                    # get projections for each column
                    nextProjections = getProjections(currBlockSpace, rotation, [], 9, currPeaksWithoutTop)
                    
                    # run reward alg on each
                    projPeaks = []
                    projPeaksWithoutTops = []
                    projRewards = []
                    projFitnesses = []
                    for proj in nextProjections:
                        tempPeaks, tempPeaksWithoutTops = findPeaks(proj)
                        projPeaks.append(tempPeaks)
                        projPeaksWithoutTops.append(tempPeaksWithoutTops)
                        tempReward, tempFitness = GetRewardFromBoard(proj, tempPeaksWithoutTops, reward, fitness)
                        projRewards.append(tempReward)
                        projFitnesses.append(tempFitness)
                    
                    eachRotation.append([nextProjections, projRewards, projFitnesses])

                
                # AI SECTION:
                state = currBlockSpace
                allPossibleStates = []
                allPossibleStatesRewards = []

                for s in range(len(eachRotation)):
                    allPossibleStates.append(eachRotation[s][0])
                    allPossibleStatesRewards.append(eachRotation[s][2])

                bugCheck = True
                for ps in allPossibleStates:
                    if len(ps) != 0:
                        bugCheck = False
                
                if bugCheck:
                    continue


                # maxRot, maxColIndex, maxState, maxReward
                rotationNumber, chosenIndex, chosenProjectedState, chosenReward = agent.get_action([state, allPossibleStates, allPossibleStatesRewards])
                episodeFinalFitness += chosenReward

                agent.update(state, chosenProjectedState, chosenReward)

                # firstColOfChosenBlock = chosenIndex
                firstColOfChosenBlock = len(eachRotation[rotationNumber][0]) - chosenIndex - 1

                # data fixing for actions
                # finalChosenProjection = eachRotation[rotationNumber][0][rotMaxesChosenIndex[rotationNumber]]
                firstColWithBlock = getFirstColAtTop(chosenProjectedState)
                left = False
                if(firstColWithBlock > firstColOfChosenBlock):
                    left = True

                # setting number of rotations
                indexofRotatedShape = unrotatedBlock + rotationNumber

                currRotation = currBlock - unrotatedBlock
                goalRotation = rotationNumber
                rotationSum = 0
                rotationDiff = currRotation - goalRotation
                if currRotation > goalRotation:
                    if unrotatedBlock == 7 or unrotatedBlock == 11 or unrotatedBlock == 15:
                        rotationSum = 4 - currRotation # gets back to start
                        rotationSum += goalRotation
                    else:
                        rotationSum = 1
                elif currRotation != goalRotation:
                    rotationSum = 1

                for i in range(rotationSum):
                    moveQueue.put(1)

                shapeHorizontalDiff = 0
                if rotationSum > 0:
                    shapeforrotationfix = len(shapes[currBlock][0])
                    rotatedforrotationfix = len(shapes[currBlock + rotationDiff][0])
                    shapeHorizontalDiff = shapeforrotationfix - rotatedforrotationfix
                    

                # assign actions to produce chosen projection
                numMovements = abs(firstColWithBlock - firstColOfChosenBlock)

                tr = unrotatedBlock + goalRotation
                if tr == 9 or tr == 17 or tr == 11 or tr == 0 or tr == 15:
                    for i in range(numMovements):
                        if left: moveQueue.put(3)
                        else: moveQueue.put(2)
                else:
                    if left:
                        for i in range(numMovements  + shapeHorizontalDiff):
                            moveQueue.put(3) 
                    else:
                        for i in range(numMovements  - shapeHorizontalDiff):
                            moveQueue.put(2) 
                  
                moveQueue.put(0)
                moveQueue.put(0) # buffer
                
        
        # reset vars for next iteration
        blockSpace = currBlockSpace
        peaks = currPeaks
        peaksWithoutTop = currPeaksWithoutTop
        block = currBlock

    agent.decay_epsilon()
    print("-----------------")
    # print('Q values{}'.format(agent.q_values))
    with open("final.pkl", 'wb') as f:
        temp = agent.v_values
        pk.dump(temp, f)
    dataArray = [episode, episodeFinalFitness, episodelines]
    datafilename = "final.csv"
    print('Episode: {} score: {} lines: {}'.format(episode, episodeFinalFitness, episodelines), )
    with open(datafilename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
 
        # writing the fields
        csvwriter.writerow(dataArray)
    
    
env.close
