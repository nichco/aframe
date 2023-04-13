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
        self.parameters.declare('E', default=69E9)
        self.parameters.declare('G', default=26E9)
        self.parameters.declare('rho', default=2700)
        self.parameters.declare('type', default='tube')
        self.parameters.declare('nodes', types=list)
        self.num_nodes = None

    def _assemble_csdl(self):
        E = self.parameters['E']
        G = self.parameters['G']
        rho = self.parameters['rho']
        typ = self.parameters['type']
        nodes = self.parameters['nodes']
        comp = self.parameters['component']
        comp_name = comp.parameters['name']
        beam_name = f'{comp_name}_beam'
        beams = dict()
        beams[beam_name] = dict(
            nodes=nodes,
            E=E,
            G=G,
            rho=rho,
            type=typ,
        )
        

        bcond = {}
        name = 'root'
        bcond[name] = {}
        bcond[name]['node'] = 0
        bcond[name]['fdim'] = [1, 1, 1, 1, 1, 1]

        """
        name = 'b1'
        beams[name] = {}
        beams[name]['nodes'] = [0,1,2,3,4,5,6,7,8,9]
        beams[name]['E'] = 69E9
        beams[name]['G'] = 26E9
        beams[name]['rho'] = 2700
        beams[name]['type'] = 'tube'
        """

        csdl_model = LinearBeamCSDL(
            beams=beams,  
            bcond=bcond,
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
    
    def define(self):
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']

        for beam_name in beams:

            nodes = beams[beam_name]['nodes']
            num_beam_nodes = len(nodes)

            mesh = self.register_module_input(beam_name+'mesh',shape=(num_beam_nodes,6))

            F = self.register_module_input(beam_name+'loads',shape=(num_beam_nodes,6))

        # solve the beam group:
        self.add_module(Group(beams=beams,bcond=bcond), name='Group')