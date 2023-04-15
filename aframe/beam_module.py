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

        self.parameters.declare('beams', default={})
        self.parameters.declare('bcond', default={})
        self.parameters.declare('connections', default={})
        self.num_nodes = None

    def _assemble_csdl(self):
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']
        connections = self.parameters['connections']

        """
        comp = self.parameters['component']
        comp_name = comp.parameters['name']
        beam_name = f'{comp_name}_beam'

        beams = dict()
        beams[beam_name] = dict(
            num_nodes=num_nodes,
            E=E,
            G=G,
            rho=rho,
            type=typ,
        )
        

        bcond = {}
        name = 'root'
        bcond[name] = {}
        bcond[name]['beam'] = 'b1'
        bcond[name]['fpos'] = 'a'
        bcond[name]['fdim'] = [1, 1, 1, 1, 1, 1]
        """

        csdl_model = LinearBeamCSDL(
            beams=beams,  
            bcond=bcond,
            connections=connections,
        )

        return csdl_model


class LinearBeamMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='m')



class LinearBeamCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('bcond')
        self.parameters.declare('connections')
    
    def define(self):
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']
        connections = self.parameters['connections']


        # solve the beam group:
        self.add_module(Group(beams=beams,connections=connections,bcond=bcond,mesh_units='ft'), name='Group')

        #self.register_module_output('mass')