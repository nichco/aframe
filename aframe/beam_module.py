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
        wing_beam_mesh = self.parameters['mesh'].parameters['meshes']['wing_beam_mesh']
        wing_beam_oml_mesh = nodal_forces.mesh.value.reshape((-1, 3))
        wing_force_map = self.fmap(wing_beam_mesh.value.reshape((-1,3)), oml=wing_beam_oml_mesh)
        return wing_force_map
    
    def construct_moment_map(self, nodal_moments):
        num_nodes = np.prod(nodal_moments.mesh.shape[:-1])
        moment_map = np.eye(num_nodes)
        return moment_map

    def construct_displacement_map(self, nodal_outputs_mesh):
        wing_beam_mesh = self.parameters['mesh'].parameters['meshes']['wing_beam_mesh']
        wing_beam_oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
        wing_displacement_map = self.umap(wing_beam_mesh.value.reshape((-1,3)), oml=wing_beam_oml_mesh)

        return wing_displacement_map
    
    def construct_rotation_map(self, nodal_outputs_mesh):
        wing_beam_mesh = self.parameters['mesh'].parameters['meshes']['wing_beam_mesh'].reshape((-1,3))
        wing_beam_oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
        # wing_rotation_map = self.mmap(wing_beam_mesh.value, oml=wing_beam_oml_mesh)

        rotation_map = np.zeros((wing_beam_oml_mesh.shape[0],wing_beam_mesh.shape[0]))

        return rotation_map
    
    def construct_invariant_matrix(self):
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
        mesh = self.parameters['mesh'].parameters['meshes']['wing_beam_mesh']

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
            nodal_forces_csdl = input_mappings_csdl.declare_variable(name=nodal_forces.name, shape=(num_forces,nodal_forces.mesh.shape[-1]))
            force_map_csdl = input_mappings_csdl.create_input('force_map', val=force_map)
            model_force_inputs = csdl.matmat(force_map_csdl, nodal_forces_csdl)
            input_mappings_csdl.register_output('wing_beam_forces', model_force_inputs)

            inputs_dictionary[nodal_forces.name] = nodal_forces
        if nodal_moments is not None:
            num_moments = np.prod(nodal_moments.mesh.shape[:-1])
            nodal_moments_csdl = input_mappings_csdl.declare_variable(name=nodal_moments.name, shape=(num_moments,nodal_moments.mesh.shape[-1]))
            moment_map_csdl = input_mappings_csdl.create_input('moment_map', val=moment_map)
            model_moment_inputs = csdl.matmat(moment_map_csdl, nodal_moments_csdl)
            input_mappings_csdl.register_output('wing_beam_moments', model_moment_inputs)

            inputs_dictionary[nodal_moments.name] = nodal_moments

        beam_csdl = self._assemble_csdl()

        output_mappings_csdl = csdl.Model()
        
        nodal_displacements_csdl = output_mappings_csdl.declare_variable(name='wing_beam_displacement', shape=(mesh.shape[0],3))
        displacement_map_csdl = output_mappings_csdl.create_input('displacement_map', val=displacement_map)
        nodal_displacements_csdl = csdl.matmat(displacement_map_csdl, nodal_displacements_csdl)
        output_mappings_csdl.register_output('beam_nodal_displacements_wing', nodal_displacements_csdl)
        
        nodal_rotations_csdl = output_mappings_csdl.declare_variable(name='wing_beam_rotation', shape=(mesh.shape[0],3))
        rotation_map_csdl = output_mappings_csdl.create_input('rotation_map', val=rotation_map)
        nodal_rotations_csdl = csdl.matmat(rotation_map_csdl, nodal_rotations_csdl)
        output_mappings_csdl.register_output('beam_nodal_rotations_wing', nodal_rotations_csdl)

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