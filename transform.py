import numpy as np
import csdl
import python_csdl_backend




class Transform(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']



        nodes = self.declare_variable('nodes', shape=(3,n))
        L = self.declare_variable('L',shape=(n-1))



        x = nodes[0,:]
        y = nodes[1,:]
        z = nodes[2,:]



        for i in range(n-1):
            cxxp = (x[i+1] - x[i])/L[i]
            cyxp = 