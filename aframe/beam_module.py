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
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('load_factor',default=1)
        self.num_nodes = None

    def _assemble_csdl(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']
        load_factor = self.parameters['load_factor']


        csdl_model = LinearBeamCSDL(
            module=self,
            beams=beams,  
            bounds=bounds,
            joints=joints,
            load_factor=load_factor,
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
        self.parameters.declare('load_factor')
    
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']
        load_factor = self.parameters['load_factor']



        for beam_name in beams:
            n = beams[beam_name]['n']
            xweb = self.register_module_input(beam_name+'t_web_in',shape=(n-1), computed_upstream=False)
            xcap = self.register_module_input(beam_name+'t_cap_in',shape=(n-1), computed_upstream=False)
            self.register_output(beam_name+'t_web',1*xweb)
            self.register_output(beam_name+'t_cap',1*xcap)

        # solve the beam group:
        self.add_module(BeamGroup(beams=beams,bounds=bounds,joints=joints,mesh_units='ft',load_factor=load_factor), name='BeamGroup')