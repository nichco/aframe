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
        beams = {}
        bcond = {}
        name = 'root'
        bcond[name] = {}
        bcond[name]['node'] = 0
        bcond[name]['fdim'] = [1, 1, 1, 1, 1, 1]

        csdl_model = LinearBeamCSDL(
            options=options,
            beams=beams,  
            bcond=bcond,
        )

        return csdl_model


class LinearBeamCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('beams')
        self.parameters.declare('bcond')
    
    def define(self):
        options = self.parameters['options']
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']

        for beam_name in beams:
            nodes = beams[beam_name]['nodes']
            num_beam_nodes = len(nodes)

            mesh = self.register_module_input(beam_name+'mesh',shape=(num_beam_nodes,6))

            F = self.register_module_input(beam_name+'loads',shape=(num_beam_nodes,6))

        # solve the beam group:
        self.add_module(Group(options=options,beams=beams,bcond=bcond), name='Group')