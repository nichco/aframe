import numpy as np
import csdl
import python_csdl_backend



"""
generates the required element dictionaries for a linear beam with 'n' nodes
e.g., an airplane wing

INPUTS:
start: the CSDL variable containing the start coordinate for the beam (shape: (3))
stop: the CSDL variable containing the stop coordinate for the beam (shape: (3))

OUTPUTS:
options[element_name]: the element entries in the global options dictionary
element_name+'node_a': the CSDL variable containing the elemental start coordinate (shape: (3))
element_name+'node_b': the CSDL variable containing the elemental stop coordinate (shape: (3))
"""



class GenBeam(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('beam_name')
        self.parameters.declare('E')
        self.parameters.declare('G')
        self.parameters.declare('rho')
        self.parameters.declare('nodes')
    def define(self):
        options = self.parameters['options']
        beam_name = self.parameters['beam_name']
        E = self.parameters['E']
        G = self.parameters['G']
        rho = self.parameters['rho']
        nodes = self.parameters['nodes']

        num_nodes = len(nodes)


        # declare the start and stop coordinates:
        start = self.declare_variable(beam_name+'start',shape=(3),val=[0,0,0])
        stop = self.declare_variable(beam_name+'stop',shape=(3),val=[0,10,0])

        # create a dictionary entry for every node:
        for i in range(num_nodes - 1):
            element_name = 'element_' + str(i)
            options[element_name] = {}

            # constant material properties:
            options[element_name]['E'] = E
            options[element_name]['G'] = G
            options[element_name]['rho'] = rho

            # define the elemental start node and stop node from the node list:
            options[element_name]['nodes'] = [nodes[i] , nodes[i+1]]

            # compute the elemental start and stop node coordinates:
            ds = (stop - start)/num_nodes

            node_a = start + ds*i
            node_b = start + ds*(i + 1)

            # register the outputs:
            self.register_output(element_name+'node_a', node_a)
            self.register_output(element_name+'node_b', node_b)





if __name__ == '__main__':
    options = {}

    nodes = [0,1,2,3,4,5,6,7,8,9]

    sim = python_csdl_backend.Simulator(GenBeam(options=options,
                                                beam_name='beam1',
                                                E=69E9,
                                                G=26E9,
                                                rho=2700,
                                                nodes=nodes))
    sim.run()