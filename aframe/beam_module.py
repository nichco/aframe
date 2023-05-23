from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from aframe.beamgroup import BeamGroup
import numpy as np

from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl

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

    def construct_map_in(self, nodal_forces):
        # Temporary dummy implementation
        num_nodes = np.cumprod(nodal_forces.shape[:-1])[-1]
        self.map_in = np.eye(num_nodes)
        map_in_csdl = ModuleCSDL()
        num_forces = np.cumprod(nodal_forces.shape[:-1])[-1]
        nodal_forces_flattened_shape = tuple((num_forces, nodal_forces.shape[-1]))
        nodal_forces_csdl = map_in_csdl.declare_variable('nodal_extrinsic_cruise_wing_pressure', shape=nodal_forces_flattened_shape)
        map_in = map_in_csdl.create_input('map_in', self.map_in.copy())
        forces_and_moments_on_beam_mesh = csdl.matmat(map_in, nodal_forces_csdl)
        left_wing_beam_forces = map_in_csdl.create_output('left_wing_beam_forces', shape=(int(num_nodes/2),3))
        right_wing_beam_forces = map_in_csdl.create_output('right_wing_beam_forces', shape=(int(num_nodes/2),3))
        left_wing_beam_forces[:,:] = forces_and_moments_on_beam_mesh[:int(num_nodes/2),:]
        right_wing_beam_forces[:,:] = forces_and_moments_on_beam_mesh[int(num_nodes/2):,:]

        self.map_in_csdl = map_in_csdl

    def construct_map_out(self, nodal_displacements_mesh):
        # Temporary dummy implementation
        num_nodes = np.cumprod(nodal_displacements_mesh.shape[:-1])[-1]
        # self.map_out = np.eye(num_nodes)
        map_out_csdl = ModuleCSDL()
        # num_displacements = np.cumprod(nodal_displacements_mesh.shape[:-1])[-1]
        # # nodal_displacements_flattened_shape = tuple((num_displacements, nodal_displacements_mesh.shape[-1]))
        # left_wing_beam_displacements = map_out_csdl.declare_variable('left_wing_beam_displacement', shape=(int(num_nodes/2),3))
        # right_wing_beam_displacements = map_out_csdl.declare_variable('right_wing_beam_displacement', shape=(int(num_nodes/2),3))
        # solver_displacements_csdl = map_out_csdl.create_output('solver_displacements_csdl', shape=(num_nodes,3))
        # solver_displacements_csdl[:int(num_nodes/2),:] = left_wing_beam_displacements
        # solver_displacements_csdl[int(num_nodes/2):,:] = right_wing_beam_displacements
        # # NOTE: The shape will be wrong in the near future.
        # map_out = map_out_csdl.create_input('map_out', self.map_out.copy())
        # nodal_displacements_csdl = csdl.matmat(map_out, solver_displacements_csdl)
        # map_out_csdl.register_output('solver_output', nodal_displacements_csdl)
        # self.map_out_csdl = map_out_csdl

        map_out_csdl = ModuleCSDL()
        left_wing_beam = self.parameters['mesh'].parameters['meshes']['left_wing_beam']
        right_wing_beam = self.parameters['mesh'].parameters['meshes']['right_wing_beam']
        left_wing_beam_mesh = nodal_displacements_mesh.value[:,:11,:].reshape((-1, 3))
        right_wing_beam_mesh = nodal_displacements_mesh.value[:,11:,:].reshape((-1, 3))
        left_wing_map = self.sisr(left_wing_beam.value.reshape((17,3)), oml=left_wing_beam_mesh)
        right_wing_map = self.sisr(right_wing_beam.value.reshape((17,3)), oml=right_wing_beam_mesh)
        output_map = np.block([
            [left_wing_map, np.zeros(left_wing_map.shape)],
            [np.zeros(left_wing_map.shape), right_wing_map]
        ])
        self.map_out = output_map
        left_wing_beam_displacements = map_out_csdl.declare_variable('left_wing_beam_displacement', shape=left_wing_beam.shape[1:])
        right_wing_beam_displacements = map_out_csdl.declare_variable('right_wing_beam_displacement', shape=right_wing_beam.shape[1:])
        solver_displacements_csdl = map_out_csdl.create_output('solver_displacements_csdl', shape=(left_wing_beam.shape[1] + right_wing_beam.shape[1], 3))
        solver_displacements_csdl[:left_wing_beam.shape[1],:] = left_wing_beam_displacements
        solver_displacements_csdl[left_wing_beam.shape[1]:,:] = right_wing_beam_displacements
        # NOTE: The shape will be wrong in the near future.
        map_out = map_out_csdl.create_input('map_out', self.map_out.copy())
        nodal_displacements_csdl = csdl.matmat(map_out, solver_displacements_csdl)
        map_out_csdl.register_output('solver_output', nodal_displacements_csdl)

        self.map_out_csdl = map_out_csdl

        


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



    def umap(self, mesh, oml):
        # Up = W*Us

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((m,n))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)
            
            weights[i, ia] = (l - ac)/length
            weights[i, ib] = (l - bc)/length

        return weights
    


    def fmap(self, mesh, oml):
        # Fs = W*Fp

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((n, m))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)
            
            weights[ia, i] = (l - ac)/length
            weights[ib, i] = (l - bc)/length

        return weights




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
            typ = beams[beam_name]['type']

            if typ == 'box':
                xweb = self.register_module_input(beam_name+'t_web_in',shape=(n-1), computed_upstream=False)
                xcap = self.register_module_input(beam_name+'t_cap_in',shape=(n-1), computed_upstream=False)
                self.print_var(xweb)
                self.register_output(beam_name+'_t_web',1*xweb)
                self.register_output(beam_name+'_t_cap',1*xcap)
                
            elif typ == 'tube':
                thickness = self.register_module_input(beam_name+'thickness_in',shape=(n-1), computed_upstream=False)
                radius = self.register_module_input(beam_name+'radius_in',shape=(n-1), computed_upstream=False)
                self.register_output(beam_name+'_thickness', 1*thickness)
                self.register_output(beam_name+'_radius', 1*radius)

        # solve the beam group:
        self.add_module(BeamGroup(beams=beams,bounds=bounds,joints=joints,load_factor=load_factor), name='BeamGroup')