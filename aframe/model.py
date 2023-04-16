import numpy as np
import csdl




class Model(csdl.Model):
    def initialize(self):
        self.parameters.declare('dim')
    def define(self):
        dim = self.parameters['dim']
        

        K = self.declare_variable('K',shape=(dim,dim))
        U = self.declare_variable('U',shape=(dim))
        F = self.declare_variable('F',shape=(dim))

        R = csdl.matvec(K,U) - F

        self.register_output('R', R)