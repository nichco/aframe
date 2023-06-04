from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from aframe.aframe import Aframe
import numpy as np

from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl
import m3l
import array_mapper as am

# class LinearBeam(m3l.Model):
class LinearBeam(MechanicsModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.num_nodes = None

    def construct_force_map(self, nodal_forces):
        num_nodes = np.prod(nodal_forces.mesh.shape[:-1])
        force_map = np.eye(num_nodes)
        return force_map
    
    def construct_moment_map(self, nodal_moments):
        num_nodes = np.prod(nodal_moments.mesh.shape[:-1])
        moment_map = np.eye(num_nodes)
        return moment_map

    def construct_displacement_map(self, nodal_outputs_mesh):
        left_wing_beam = self.parameters['mesh'].parameters['meshes']['left_wing_beam']
        right_wing_beam = self.parameters['mesh'].parameters['meshes']['right_wing_beam']
        left_wing_beam_mesh = nodal_outputs_mesh.value[:,:11,:].reshape((-1, 3))
        right_wing_beam_mesh = nodal_outputs_mesh.value[:,11:,:].reshape((-1, 3))
        left_wing_map = self.sisr(left_wing_beam.value.reshape((17,3)), oml=left_wing_beam_mesh)
        right_wing_map = self.sisr(right_wing_beam.value.reshape((17,3)), oml=right_wing_beam_mesh)
        output_map = np.block([
            [left_wing_map, np.zeros(left_wing_map.shape)],
            [np.zeros(left_wing_map.shape), right_wing_map]
        ])
        return output_map
    
    def construct_rotation_map(self, nodal_outputs_mesh):
        num_outputs = np.prod(nodal_outputs_mesh.shape[:-1])
        moment_map = np.zeros((num_outputs,34))
        return moment_map
    
    def construct_mass_matrix(self):
        pass

    def evaluate(self, nodal_outputs_mesh:am.MappedArray, nodal_forces:m3l.Function=None, nodal_moments:m3l.Function=None):
        '''
        Evaluates the model.

        Parameters
        ----------
        nodal_outputs_mesh : am.MappedArray
            The mesh or pointcloud representing the locations at which the nodal displacements and rotations will be returned.
        nodal_forces : m3l.NodalState
            The nodal forces that will be mapped onto the beam.
        nodal_moments : m3l.NodalState
            The nodal moments that will be mapped onto the beam.
        
        Returns
        -------
        nodal_displacements : m3l.NodalState
            The displacements evaluated at the locations specified by nodal_outputs_mesh
        nodal_rotations : m3l.NodalState
            The rotations evluated at the locations specified by the nodal_outputs_mesh
        '''
        if nodal_forces is not None:
            force_map = self.construct_force_map(nodal_forces=nodal_forces)
        if nodal_moments is not None:
            moment_map = self.construct_moment_map(nodal_moments=nodal_moments)
        displacement_map = self.construct_displacement_map(nodal_outputs_mesh=nodal_outputs_mesh)
        rotation_map = self.construct_rotation_map(nodal_outputs_mesh=nodal_outputs_mesh)

        csdl_model = ModuleCSDL()
        
        # beam_nodal_displacements, beam_nodal_rotations = super().evaluate(model_map=self._assemble_csdl(), 
        #                  inputs=[(nodal_forces,force_map), (nodal_moments,moment_map)],
        #                  outputs=[("beam_nodal_displacement_wing",displacement_map), ("beam_nodal_rotation_wing",rotation_map)]
        #                  )

        input_mappings_csdl = csdl.Model()
        inputs_dictionary = {}
        if nodal_forces is not None:
            num_forces = np.prod(nodal_forces.mesh.shape[:-1])
            nodal_forces_csdl = input_mappings_csdl.declare_variable(name='nodal_forces', shape=(num_forces,nodal_forces.mesh.shape[-1]))
            force_map_csdl = input_mappings_csdl.create_input('force_map', val=force_map)
            model_force_inputs = csdl.matmat(force_map_csdl, nodal_forces_csdl)
            input_mappings_csdl.register_output('left_wing_beam_forces', model_force_inputs)

            inputs_dictionary[nodal_forces.name] = nodal_forces
        if nodal_moments is not None:
            num_moments = np.prod(nodal_moments.mesh.shape[:-1])
            nodal_moments_csdl = input_mappings_csdl.declare_variable(name='nodal_moments', shape=(num_moments,nodal_moments.mesh.shape[-1]))
            moment_map_csdl = input_mappings_csdl.create_input('moment_map', val=moment_map)
            model_moment_inputs = csdl.matmat(moment_map_csdl, nodal_moments_csdl)
            input_mappings_csdl.register_output('left_wing_beam_moments', model_moment_inputs)

            inputs_dictionary[nodal_moments.name] = nodal_moments

        beam_csdl = self._assemble_csdl()

        output_mappings_csdl = csdl.Model()
        
        nodal_displacements_csdl = output_mappings_csdl.declare_variable(name='left_wing_beam_displacements', shape=(34,3))
        displacement_map_csdl = output_mappings_csdl.create_input('displacement_map', val=displacement_map)
        nodal_displacements_csdl = csdl.matmat(displacement_map_csdl, nodal_displacements_csdl)
        output_mappings_csdl.register_output('nodal_displacements', nodal_displacements_csdl)
        
        nodal_rotations_csdl = output_mappings_csdl.declare_variable(name='left_wing_beam_rotations', shape=(34,3))
        rotation_map_csdl = output_mappings_csdl.create_input('rotation_map', val=rotation_map)
        nodal_rotations_csdl = csdl.matmat(rotation_map_csdl, nodal_rotations_csdl)
        output_mappings_csdl.register_output('nodal_rotations', nodal_rotations_csdl)

        csdl_model.add(submodel=input_mappings_csdl, name='beam_inputs_mapping')
        csdl_model.add(submodel=beam_csdl, name='beam_model')

        csdl_model.add(submodel=output_mappings_csdl, name='beam_outputs_mapping')

        nodal_displacements = m3l.FunctionValues(name='beam_nodal_displacements_wing', 
                                                 upstream_variables=inputs_dictionary,
                                                 map=csdl_model,
                                                 mesh=nodal_outputs_mesh)
        nodal_rotations = m3l.FunctionValues(name='beam_nodal_rotations_wing', 
                                                 upstream_variables=inputs_dictionary,
                                                 map=csdl_model,
                                                 mesh=nodal_outputs_mesh)
        
        return nodal_displacements, nodal_rotations

    def _assemble_csdl(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        csdl_model = LinearBeamCSDL(
            module=self,
            beams=beams,  
            bounds=bounds,
            joints=joints)

        return csdl_model



# class LinearBeam(MechanicsModel):
#     def initialize(self, kwargs):
#         self.parameters.declare('component', default=None)
#         self.parameters.declare('mesh', default=None)
#         self.parameters.declare('struct_solver', True)
#         self.parameters.declare('compute_mass_properties', default=True, types=bool)

#         self.parameters.declare('beams', default={})
#         self.parameters.declare('bounds', default={})
#         self.parameters.declare('joints', default={})
#         self.parameters.declare('load_factor',default=1)
#         self.num_nodes = None

#     def construct_map_in(self, nodal_forces):
#         # Temporary dummy implementation
#         num_nodes = np.cumprod(nodal_forces.shape[:-1])[-1]
#         self.map_in = np.eye(num_nodes)
#         map_in_csdl = ModuleCSDL()
#         num_forces = np.cumprod(nodal_forces.shape[:-1])[-1]
#         nodal_forces_flattened_shape = tuple((num_forces, nodal_forces.shape[-1]))
#         nodal_forces_csdl = map_in_csdl.declare_variable('nodal_extrinsic_cruise_wing_pressure', shape=nodal_forces_flattened_shape)
#         map_in = map_in_csdl.create_input('map_in', self.map_in.copy())
#         forces_and_moments_on_beam_mesh = csdl.matmat(map_in, nodal_forces_csdl)
#         left_wing_beam_forces = map_in_csdl.create_output('left_wing_beam_forces', shape=(int(num_nodes/2),3))
#         right_wing_beam_forces = map_in_csdl.create_output('right_wing_beam_forces', shape=(int(num_nodes/2),3))
#         left_wing_beam_forces[:,:] = forces_and_moments_on_beam_mesh[:int(num_nodes/2),:]
#         right_wing_beam_forces[:,:] = forces_and_moments_on_beam_mesh[int(num_nodes/2):,:]

#         self.map_in_csdl = map_in_csdl

#     def construct_map_out(self, nodal_outputs_mesh):
#         # Temporary dummy implementation
#         num_nodes = np.cumprod(nodal_outputs_mesh.shape[:-1])[-1]
#         # self.map_out = np.eye(num_nodes)
#         map_out_csdl = ModuleCSDL()
#         # num_displacements = np.cumprod(nodal_outputs_mesh.shape[:-1])[-1]
#         # # nodal_displacements_flattened_shape = tuple((num_displacements, nodal_outputs_mesh.shape[-1]))
#         # left_wing_beam_displacements = map_out_csdl.declare_variable('left_wing_beam_displacement', shape=(int(num_nodes/2),3))
#         # right_wing_beam_displacements = map_out_csdl.declare_variable('right_wing_beam_displacement', shape=(int(num_nodes/2),3))
#         # solver_displacements_csdl = map_out_csdl.create_output('solver_displacements_csdl', shape=(num_nodes,3))
#         # solver_displacements_csdl[:int(num_nodes/2),:] = left_wing_beam_displacements
#         # solver_displacements_csdl[int(num_nodes/2):,:] = right_wing_beam_displacements
#         # # NOTE: The shape will be wrong in the near future.
#         # map_out = map_out_csdl.create_input('map_out', self.map_out.copy())
#         # nodal_displacements_csdl = csdl.matmat(map_out, solver_displacements_csdl)
#         # map_out_csdl.register_output('solver_output', nodal_displacements_csdl)
#         # self.map_out_csdl = map_out_csdl

#         map_out_csdl = ModuleCSDL()
#         left_wing_beam = self.parameters['mesh'].parameters['meshes']['left_wing_beam']
#         right_wing_beam = self.parameters['mesh'].parameters['meshes']['right_wing_beam']
#         left_wing_beam_mesh = nodal_outputs_mesh.value[:,:11,:].reshape((-1, 3))
#         right_wing_beam_mesh = nodal_outputs_mesh.value[:,11:,:].reshape((-1, 3))
#         left_wing_map = self.sisr(left_wing_beam.value.reshape((17,3)), oml=left_wing_beam_mesh)
#         right_wing_map = self.sisr(right_wing_beam.value.reshape((17,3)), oml=right_wing_beam_mesh)
#         output_map = np.block([
#             [left_wing_map, np.zeros(left_wing_map.shape)],
#             [np.zeros(left_wing_map.shape), right_wing_map]
#         ])
#         self.map_out = output_map
#         left_wing_beam_displacements = map_out_csdl.declare_variable('left_wing_beam_displacement', shape=left_wing_beam.shape[1:])
#         right_wing_beam_displacements = map_out_csdl.declare_variable('right_wing_beam_displacement', shape=right_wing_beam.shape[1:])
#         solver_displacements_csdl = map_out_csdl.create_output('solver_displacements_csdl', shape=(left_wing_beam.shape[1] + right_wing_beam.shape[1], 3))
#         solver_displacements_csdl[:left_wing_beam_displacements.shape[0],:] = left_wing_beam_displacements
#         solver_displacements_csdl[left_wing_beam_displacements.shape[0]:,:] = right_wing_beam_displacements
#         # NOTE: The shape will be wrong in the near future.
#         map_out = map_out_csdl.create_input('map_out', self.map_out.copy())
#         nodal_displacements_csdl = csdl.matmat(map_out, solver_displacements_csdl)
#         map_out_csdl.register_output('solver_output', nodal_displacements_csdl)

#         self.map_out_csdl = map_out_csdl

#     def _assemble_csdl(self):
#         beams = self.parameters['beams']
#         bounds = self.parameters['bounds']
#         joints = self.parameters['joints']
#         load_factor = self.parameters['load_factor']

#         csdl_model = LinearBeamCSDL(
#             module=self,
#             beams=beams,  
#             bounds=bounds,
#             joints=joints,
#             load_factor=load_factor,
#         )

#         # new_csdl_model = ModuleCSDL()
#         # new_csdl_model.add(map_in_csdl, 'map_in')
#         # new_csdl_model.add_module(model_csdl, model_name_snake)
#         # new_csdl_model.add(map_out_csdl, 'map_out')
#         # new_csdl_model = new_csdl_model

#         return csdl_model


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
    
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            cs = beams[beam_name]['cs']

            if cs == 'box':
                xweb = self.register_module_input(beam_name+'t_web_in',shape=(n-1), computed_upstream=False)
                xcap = self.register_module_input(beam_name+'t_cap_in',shape=(n-1), computed_upstream=False)
                self.register_output(beam_name+'_tweb',1*xweb)
                self.register_output(beam_name+'_tcap',1*xcap)
                
            elif cs == 'tube':
                thickness = self.register_module_input(beam_name+'thickness_in',shape=(n-1), computed_upstream=False)
                radius = self.register_module_input(beam_name+'radius_in',shape=(n-1), computed_upstream=False)
                self.register_output(beam_name+'_t', 1*thickness)
                self.register_output(beam_name+'_r', 1*radius)

        # solve the beam group:
        self.add_module(Aframe(beams=beams, bounds=bounds, joints=joints), name='Aframe')