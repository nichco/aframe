import numpy as np
import csdl
import python_csdl_backend



class GlobalStiffnessHelper(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('name')
        self.parameters.declare('dim')
        self.parameters.declare('node_list')
    def define(self):
        options = self.parameters['options']
        name = self.parameters['name']
        dim = self.parameters['dim']
        node_list = self.parameters['node_list']


        #element_nodes = options['nodes']
        node_1 =  options['nodes'][0]
        node_2 =  options['nodes'][1]



        # the transformed, local stiffness matrix:
        k = self.declare_variable(name+'k',shape=(12,12))
        # upper left k block
        k11 = k[0:6, 0:6]
        # upper right k block
        k12 = k[0:6, 6:11]
        # bottom left k block
        k21 = k[6:11, 0:6]
        # bottom right k block
        k22 = k[6:11, 6:11]


        # create global stiffness helper matrix:
        h = self.create_output(name+'h',shape=(dim,dim),val=0)

        for i, node in enumerate(node_list):
            if node == node_1:
                h[i*6 : i*6 + 6] = k11
            


        
