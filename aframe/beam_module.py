from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from aframe.group import Group
import numpy as np


class LinearBeam(MechanicsModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)
        self.num_nodes = None

    def _assemble_csdl(self):
        options = {}
        bcond = {}
        name = 'root'
        bcond[name] = {}
        bcond[name]['node'] = 0
        bcond[name]['fdim'] = [1, 1, 1, 1, 1, 1]

        csdl_model = LinearBeamCSDL(
            options=options, 
            bcond=bcond,
        )

        return csdl_model


class LinearBeamCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('bcond')
    
    def define(self):
        options = self.parameters['options']
        bcond = self.parameters['bcond']

        # generate the beam mesh:
        beam_name = 'beam1'
        start = self.register_module_input(beam_name+'start', shape=(6), val=[0, 0, 0, 0, 0, 0])
        stop = self.register_module_input(beam_name+'stop', shape=(6), val=[10, 0, 0, 0, 0, 0])
        nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_nodes = len(nodes) 

        for i in range(num_nodes - 1):
            element_name = 'element_' + str(i)
            options[element_name] = {}
            # constant material properties:
            options[element_name]['type'] = 'tube'
            options[element_name]['E'], options[element_name]['G'], options[element_name]['rho'] = 69E9, 26E9, 2700
            # define the elemental start node and stop node from the node list:
            options[element_name]['nodes'] = [nodes[i] , nodes[i+1]]
            # compute the elemental start and stop node coordinates:
            ds = (stop - start)/num_nodes
            node_a = start + ds*i
            node_b = start + ds*(i + 1)
            # register the outputs:
            self.register_output(element_name+'node_a', node_a)
            self.register_output(element_name+'node_b', node_b)

            # generate the loads vector:
        
        # pre-process the options dictionary to get dim:
        node_list = [options[name]['nodes'][0] for name in options] + [options[name]['nodes'][1] for name in options]
        node_list = [*set(node_list)]
        num_unique_nodes = len(node_list)
        dim = num_unique_nodes*6

        # create the global loads vector
        loads = np.zeros((dim))
        loads[dim-4] = -200
        F = self.create_input('F', shape=(dim), val=loads)

        # solve the beam group:
        self.add_module(Group(options=options,bcond=bcond), name='Group')