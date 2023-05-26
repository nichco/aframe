import numpy as np
import csdl
import python_csdl_backend




class Aframe(csdl.Model):

    def initialize(self):
        self.parameters.declare('beams', default={})


    def add_beam(self, name, n, cs, e, g, rho):

        mesh = self.declare_variable(name + '_mesh', shape=(n,3))
        self.create_input(name + '_E', e)
        self.create_input(name + '_G', g)
        
        # iterate over each element:
        for i in range(n - 1):
            element_name = name + '_element_' + str(i)
            node_a = csdl.reshape(mesh[i, :], (3))
            node_b = csdl.reshape(mesh[i + 1, :], (3))
            self.register_output(element_name + 'node_a', node_a)
            self.register_output(element_name + 'node_b', node_b)


    def define(self):
        beams = self.parameters['beams']


        for name in beams:
            self.add_beam(name=name, 
                          n=beams[name]['n'], 
                          cs=beams[name]['cs'], 
                          e=beams[name]['E'],
                          g=beams[name]['G'],
                          rho=beams[name]['rho'])








beams = {}
beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','n': 10}
sim = python_csdl_backend.Simulator(Aframe(beams=beams))
sim.run()