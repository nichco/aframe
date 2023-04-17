from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from aframe.beamgroup import BeamGroup
import numpy as np


class LinearBeam(MechanicsModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)

        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.num_nodes = None

    def _assemble_csdl(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        """
        comp = self.parameters['component']
        comp_name = comp.parameters['name']
        beam_name = f'{comp_name}_beam'
        """

        csdl_model = LinearBeamCSDL(
            beams=beams,  
            bounds=bounds,
            joints=joints,
        )

        return csdl_model


class LinearBeamMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='m')



class LinearBeamCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('bounds')
        self.parameters.declare('joints')
    
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']


        # solve the beam group:
        self.add_module(BeamGroup(beams=beams,bounds=bounds,joints=joints,mesh_units='ft'), name='BeamGroup')