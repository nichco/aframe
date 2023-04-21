import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import numpy as np
from aframe.localk import LocalK
from aframe.globalk import GlobalK
from aframe.globalloads import GlobalLoads
from aframe.model import Model
from aframe.stress import StressTube, StressBox
from aframe.massprop import MassProp
from aframe.sectionproperties import SectionPropertiesBox, SectionPropertiesRect, SectionPropertiesTube






class BeamGroup(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('joints',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('mesh_units',default='m')
        self.parameters.declare('load_factor',default=1)
        self.parameters.declare('sy',default=450E6)
        self.parameters.declare('fos',default=1)
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']
        bounds = self.parameters['bounds']
        mesh_units = self.parameters['mesh_units']
        load_factor = self.parameters['load_factor']
        sy = self.parameters['sy']
        fos = self.parameters['fos']


        # error handling
        if not beams: raise Exception('Error: empty beam dictionary')
        if not bounds: raise Exception('Error: an empty boundary condition dictionary is guaranteed to yield a singular system')


        # automated beam node assignment:
        nodes = {}
        # start by populating the nodes dictionary as if there aren't any joints:
        index = 0
        for beam_name in beams:
            nodes[beam_name] = np.arange(index, index + beams[beam_name]['n'])
            index += beams[beam_name]['n']

        # modify the nodes dictionary for any joints:
        if joints:
            # all joints inherit the node number of the first beam listed in the joint-beam-name list:
            for joint_name in joints:
                beam_list = joints[joint_name]['beam_names']
                first_beam_name = beam_list[0]
                # determine which end/node of the first beam is part of the joint:
                first_beam_end = joints[joint_name]['nodes'][0]
                if first_beam_end == 'a': joint_node = nodes[first_beam_name][0]
                elif first_beam_end == 'b': joint_node = nodes[first_beam_name][-1]

                for i, beam_name in enumerate(beam_list):
                    if i != 0:
                        beam_end = joints[joint_name]['nodes'][i]
                        if beam_end == 'a': nodes[beam_name][0] = joint_node
                        if beam_end == 'b': nodes[beam_name][-1] = joint_node
                        


        # create the element dictionary and output relevant properties:
        elements = {}
        for beam_name in beams:
            n = beams[beam_name]['n']
            rho, typ = beams[beam_name]['rho'], beams[beam_name]['type']

            # register the mesh input:
            mesh_input = self.register_module_input(beam_name,shape=(n,3), promotes=True)
            #self.print_var(mesh_input)

            if mesh_units == 'ft': mesh = mesh_input/3.281
            elif mesh_units == 'm': mesh = mesh_input

            # iterate over the beam elements:
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                elements[element_name] = {'rho': rho,'type': typ,'node_a': nodes[beam_name][i],'node_b': nodes[beam_name][i + 1]}
                self.register_output(element_name+'node_a_position', csdl.reshape(mesh[i, :], (3)))
                self.register_output(element_name+'node_b_position', csdl.reshape(mesh[i + 1, :], (3)))
                self.create_input(element_name+'E', beams[beam_name]['E'])
                self.create_input(element_name+'G', beams[beam_name]['G'])





        # compute the system dimension:
        node_list = [*set([elements[name]['node_a'] for name in elements] + [elements[name]['node_b'] for name in elements])]
        num_unique_nodes = len(node_list)
        dim = num_unique_nodes*6
        # create a dictionary that contains the nodes and the node index in the global system:
        node_index = {node_list[i]: i for i in range(num_unique_nodes)}






        # parse the cross section parameter meshes:
        for beam_name in beams:
            n = beams[beam_name]['n']

            if beams[beam_name]['type'] == 'tube':
                #thickness = self.declare_variable(beam_name+'thickness',shape=(n - 1),val=0.001)
                #radius = self.declare_variable(beam_name+'radius',shape=(n - 1),val=0.25)

                thickness = self.register_module_input(beam_name+'_thickness',shape=(n-1))
                radius = self.register_module_input(beam_name+'_radius',shape=(n-1))


                for i in range(n - 1):
                    element_name = beam_name + '_element_' + str(i)

                    #if mesh_units == 'ft':
                    #    self.register_output(element_name+'thickness',thickness[i]/3.281)
                    #    self.register_output(element_name+'radius',radius[i]/3.281)

                    #elif mesh_units == 'm':
                    self.register_output(element_name+'thickness',1*thickness[i])
                    self.register_output(element_name+'radius',1*radius[i])

            elif beams[beam_name]['type'] == 'box':

                #width_mesh = self.register_module_input(beam_name+'_width',shape=(n),promotes=True)
                #height_mesh = self.register_module_input(beam_name+'_height',shape=(n),promotes=True)

                width_mesh_i = self.register_module_input(beam_name+'_width',shape=(n,3),promotes=True)
                width_mesh = csdl.pnorm(width_mesh_i,axis=1,pnorm_type=2)*0.4
                height_mesh_i = self.register_module_input(beam_name+'_height',shape=(n,3),promotes=True)
                height_mesh = csdl.pnorm(height_mesh_i,axis=1,pnorm_type=2)

                #t_web = self.register_module_input(beam_name+'t_web',shape=(n-1))
                t_cap = self.register_module_input(beam_name+'t_cap',shape=(n-1))
                t_web = 1*t_cap # temporarily make the web and cap the same thickness


                # process the meshes to get average element dimensions:
                width = self.create_output(beam_name+'element_width',shape=(n-1))
                height = self.create_output(beam_name+'element_height',shape=(n-1))
                for i in range(n-1): width[i] = (width_mesh[i] + width_mesh[i+1])/2
                for i in range(n-1): height[i] = (height_mesh[i] + height_mesh[i+1])/2

                for i in range(n - 1):
                    element_name = beam_name + '_element_' + str(i)

                    if mesh_units == 'ft':
                        self.register_output(element_name+'width',width[i]/3.281)
                        self.register_output(element_name+'height',height[i]/3.281)
                        self.register_output(element_name+'t_web',t_web[i]) # thickness units are still meters
                        self.register_output(element_name+'t_cap',t_cap[i])

                    elif mesh_units == 'm':
                        self.register_output(element_name+'width',1*width[i])
                        self.register_output(element_name+'height',1*height[i])
                        self.register_output(element_name+'t_web',1*t_web[i])
                        self.register_output(element_name+'t_cap',1*t_cap[i])





        # compute the section properties for each element:
        for element_name in elements:
            if elements[element_name]['type'] == 'tube':
                self.add(SectionPropertiesTube(element_name=element_name), name=element_name+'SectionPropertiesTube')
            elif elements[element_name]['type'] == 'box': 
                self.add(SectionPropertiesBox(element_name=element_name), name=element_name+'SectionPropertiesBox')
            elif elements[element_name]['type'] == 'rect': 
                self.add(SectionPropertiesRect(element_name=element_name), name=element_name+'SectionPropertiesRect')
            else: raise NotImplementedError('Error: type for' + element_name + 'is not implemented')






        # compute the transformed, local stiffness matrices for each element:
        self.add(LocalK(elements=elements,dim=dim,node_index=node_index),name='LocalK')

        # compute the global stiffness matrix:
        self.add(GlobalK(dim=dim,elements=elements,bounds=bounds,node_index=node_index,nodes=nodes), name='GlobalK')
        K = self.declare_variable('K',shape=(dim,dim))

        # create the global loads vector:
        self.add(GlobalLoads(beams=beams,num_unique_nodes=num_unique_nodes,nodes=nodes,node_index=node_index,bounds=bounds,load_factor=load_factor), name='GlobalLoads')
        Fi = self.declare_variable('Fi',shape=(dim))







        # solve the linear system
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=100,iprint=False,atol=1E-5,)
        solve_res.linear_solver = csdl.ScipyKrylov()
        U = solve_res(K, Fi)







        # recover the local elemental forces/moments:
        for element_name in elements:
            # get the nodes and the node ID's:
            node_a_id, node_b_id = node_index[elements[element_name]['node_a']], node_index[elements[element_name]['node_b']]
            # get the nodal displacements for the current element:
            disp_a, disp_b = U[node_a_id*6:node_a_id*6 + 6], U[node_b_id*6:node_b_id*6 + 6]
            # concatenate the nodal displacements:
            d = self.create_output(element_name+'d',shape=(12),val=0)
            d[0:6], d[6:12] = disp_a, disp_b
            # declare the variables for the local stiffness matrix and the element transformation matrix:
            kp = self.declare_variable(element_name+'kp',shape=(12,12))
            T = self.declare_variable(element_name+'T',shape=(12,12))
            # element local loads output (required for the stress recovery):
            self.register_output(element_name+'local_loads', csdl.matvec(kp,csdl.matvec(T,d)))







        # parse the displacements to get the new nodal coordinates:
        for element_name in elements:
            # get the undeformed nodal coordinates:
            node_a_position = self.declare_variable(element_name+'node_a_position',shape=(3))
            node_b_position = self.declare_variable(element_name+'node_b_position',shape=(3))
            # get the nodes and the node ID's:
            node_a, node_b =  elements[element_name]['node_a'], elements[element_name]['node_b']
            node_a_index, node_b_index = node_index[node_a], node_index[node_b]
            # get the nodal displacements for the current element:
            dn1 = U[node_a_index*6:node_a_index*6 + 3] # node 1 displacements
            dn2 = U[node_b_index*6:node_b_index*6 + 3] # node 2 displacements
            # assign the elemental output:
            self.register_output(element_name+'node_a_def',node_a_position + dn1)
            self.register_output(element_name+'node_b_def',node_b_position + dn2)





        # perform a stress recovery:
        vonmises_stress = self.create_output('vonmises_stress',shape=(len(elements)),val=0) # the global element stress vector
        for i, element_name in enumerate(elements):
            if elements[element_name]['type'] == 'tube': 
                self.add(StressTube(name=element_name), name=element_name+'Stress')
                vonmises_stress[i] = self.declare_variable(element_name+'s_vm')
            
            elif elements[element_name]['type'] == 'box':
                self.add(StressBox(name=element_name), name=element_name+'Stress')
                vonmises_stress[i] = self.declare_variable(element_name+'s_vm')
            
            else: raise NotImplementedError('Error: stress recovery for [beam type] is not implemented')


        # compute the maximum stress in the entire system:
        max_stress = csdl.max(vonmises_stress)
        #self.register_output('max_stress',max_stress)
        self.register_module_output('max_stress', max_stress)
        self.print_var(max_stress/1E6)
        

        #margin = (sy/(max_stress*fos)) - 1
        #self.register_output('margin', margin)
        #self.print_var(margin)


        
        # compute the cg and moi:
        self.add(MassProp(elements=elements), name='MassProp')



        # output dummy forces and moments:
        zero = self.declare_variable('zero_vec',shape=(3),val=0)
        self.register_output('F', 1*zero)
        self.register_output('M', 1*zero)