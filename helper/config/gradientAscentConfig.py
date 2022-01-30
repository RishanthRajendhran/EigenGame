import helper.config.mainConfig as config

numStepsPerIteration = 20
T = 100000
ascentVariant = "vanilla"
numIterations = T//config.L
gamma = 0.9
beta = 0.9
eps = 1e-8
beta1 = 0.9
beta2 = 0.999
learningRate = 0.0005
penaltyCoefficient = 10
extraPenaltyCoefficient = 100