import helper.config.mainConfig as config

numStepsPerIteration = 200
T = 100
ascentVariant = "vanilla"
numIterations = T//config.L
gamma = 0.9
beta = 0.9
eps = 1e-8
beta1 = 0.9
beta2 = 0.999
learningRate = 5e-6
