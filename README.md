# Neural_network
For some given numbers, this neural network calculates the relation between inputs and outputs.

from numpy import exp, array, random, dot

class neural_network:
    def __init__(self):
        random.seed(1)
        
        self.weights = 2 * random.random((2,1)) - 1
        
    def train (self, inputs, outputs, num): #Num es el numero de veces que entrena
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = 0.01 * dot(inputs.T,error)
            self.weights += adjustment
            
    def think(self, inputs):
        return(dot(inputs, self.weights))
        
#Declaramos neural_network como objeto de clase neural_network()       
neural_network = neural_network()

#Aqui van los datos de entrenamiento
    #Entradas
inputs = array([[2,3], [1,1],[5,2],[12,3]])
    #Salidas
outputs = array([[10,4,14,30]]).T

#Se procede a entrenar la red neuroal:
neural_network.train(inputs, outputs, 10000)

print(neural_network.think(([12,3],[14,10],[12,19])))
