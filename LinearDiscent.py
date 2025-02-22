import numpy as np


#================Optimization Problem================

trainExamples = [
    (1,1),
    (2,3),
    (4,3),
]


def phi(x):
    return np.array([1,x])

def initialWeightVector():
    return np.zeros(2)


def trainLoss(w):
    return 1.0/len(trainExamples) * sum((w.dot(phi(x))-y)**2 for x,y in trainExamples )



def GradienttrainLoss(w):
    return 1.0/len(trainExamples) * sum(2 *(w.dot(phi(x))-y)* phi(x) for x,y in trainExamples )


#===============Optimization Algorithm================
def gradientDescent(F,gradientF,initialWeightVector):
   w=initialWeightVector()
   eta=0.1
   for t in range(500):
       value=F(w)
       gradient=gradientF(w)
       w=w-eta*gradient
       print(f'epach({t}):w={w},F(w)={value},gradientF={gradient} ') 
       
gradientDescent(trainLoss,GradienttrainLoss, initialWeightVector)              