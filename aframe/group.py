import numpy as np
import csdl
from aframe.sectionproperties import SectionPropertiesTube, SectionPropertiesBox, SectionPropertiesRect
from aframe.localstiffness import LocalStiffness
from aframe.model import Model
from aframe.stress import StressTube, StressBox
from aframe.cg import Cg
from aframe.globalloads import GlobalLoads



class Group(csdl.Model):
    def initialize(self):
        self.parameters.declare('options',default={})
        self.parameters.declare('beams',default={})
        self.parameters.declare('bcond',default={})
    def define(self):
        options = self.parameters['options']
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']

        """
        # parse the beam dictionary to create the elemental options dictionary:
        for beam_name in beams:
            nodes = beams[beam_name]['nodes']
            type = beams[beam_name]['type']
            E, G, rho = beams[beam_name]['E'], beams[beam_name]['G'], beams[beam_name]['rho']
            num_nodes = len(nodes)

            mesh = self.declare_variable(beam_name+'mesh',shape=(num_nodes,6),val=0)
            for i in range(num_nodes - 1):
                element_name = beam_name + '_element_' + str(i)
                options[element_name] = {}
                # constant material properties and type:
                options[element_name]['type'] = type
                options[element_name]['E'], options[element_name]['G'], options[element_name]['rho'] = E, G, rho
                # define the elemental start node and stop node from the node list:
                options[element_name]['nodes'] = [nodes[i] , nodes[i+1]]
                # compute the elemental start and stop node coordinates:
                na = csdl.reshape(mesh[i,:], (6))
                nb = csdl.reshape(mesh[i+1,:], (6))
                # register the outputs:
                self.register_output(element_name+'node_a', na)
                self.register_output(element_name+'node_b', nb)
        """






        # process the options dictionary to compute the total number of unique nodes:
        node_list = [*set([options[name]['nodes'][0] for name in options] + [options[name]['nodes'][1] for name in options])]
        num_unique_nodes = len(node_list)
        num_elements = len(options)
        dim = num_unique_nodes*6

        # create a dictionary that contains the nodes and the node index
        node_id = {node_list[i]: i for i in range(num_unique_nodes)}

        # create a list of unique bc nodes:
        bc_list = [*set([bcond[bc_name]['node'] for bc_name in bcond])]




        # create the global loads vector:
        if beams:
            self.add(GlobalLoads(options=options,
                                beams=beams,
                                bcond=bcond,
                                node_id=node_id,
                                num_unique_nodes=num_unique_nodes
                                ), name='GlobalLoads')
        
        F = self.declare_variable('F',shape=(dim),val=0)




        # compute the section properties for each element:
        for element_name in options:
            if options[element_name]['type'] == 'tube': 
                self.add(SectionPropertiesTube(name=element_name), name=element_name+'SectionPropertiesTube')
            elif options[element_name]['type'] == 'box': 
                self.add(SectionPropertiesBox(name=element_name), name=element_name+'SectionPropertiesBox')
            elif options[element_name]['type'] == 'rect': 
                self.add(SectionPropertiesRect(name=element_name), name=element_name+'SectionPropertiesRect')
            else: raise NotImplementedError('Error: type for' + element_name + 'is not implemented')






        # compute the local stiffness matrix for each element:
        for element_name in options:
            self.add(LocalStiffness(options=options[element_name],name=element_name,dim=dim,node_id=node_id), name=element_name+'LocalStiffness')

        


        # construct the global stiffness matrix:
        helper = self.create_output('helper',shape=(num_elements,dim,dim),val=0)
        for i, element_name in enumerate(options):
            k = self.declare_variable(element_name+'k',shape=(dim,dim))
            helper[i,:,:] = csdl.expand(k, (1,dim,dim), 'ij->aij')

        sum_k = csdl.sum(helper, axes=(0, ))


        # boundary conditions
        bc_id = []
        for node, id in node_id.items():
            
            for bc_name in bcond:
                if bcond[bc_name]['node'] == node:
                    # iterate over 'fdim' to see which dof's are constrained:
                    for i, fdim in enumerate(bcond[bc_name]['fdim']):
                        if fdim == 1:
                            # add the constrained dof ID to the bc_id list:
                            bc_id.append(id*6 + i)
            


        mask = self.create_output('mask',shape=(dim,dim),val=np.eye(dim))
        mask_eye = self.create_output('mask_eye',shape=(dim,dim),val=0)
        zero = self.create_input('zero',shape=(1,1),val=0)
        one = self.create_input('one',shape=(1,1),val=1)
        for i in range(dim):
            if i in bc_id: 
                mask[i,i] = 1*zero
                mask_eye[i,i] = 1*one


        # modify the global stiffness matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = csdl.transpose(csdl.matmat(mask, csdl.transpose(csdl.matmat(mask,sum_k)))) + mask_eye
        self.register_output('K', K)

        



        # solve the linear system
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=1000,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()

        U = solve_res(K, F)

        

        # recover the local elemental forces/moments (fp):
        local_loads = self.create_output('local_loads',shape=(num_elements,12),val=0)
        for i, element_name in enumerate(options):
            # get the nodes and the node ID's:
            node_1, node_2 =  options[element_name]['nodes'][0], options[element_name]['nodes'][1]
            node_1_id = [id for node, id in node_id.items() if node == node_1][0]
            node_2_id = [id for node, id in node_id.items() if node == node_2][0]

            # get the nodal displacements for the current element:
            dn1 = U[node_1_id*6:node_1_id*6 + 6] # node 1 displacements
            dn2 = U[node_2_id*6:node_2_id*6 + 6] # node 2 displacements

            # concatenate the nodal displacements:
            d = self.create_output(element_name+'d',shape=(12),val=0)
            d[0:6], d[6:12] = dn1, dn2

            # declare the variables for the local stiffness matrix and the element transformation matrix:
            kp = self.declare_variable(element_name+'kp',shape=(12,12))
            T = self.declare_variable(element_name+'T',shape=(12,12))

            # solve for the local loads:
            # group output (required for plotting and post processing):
            local_loads[i,:] = csdl.reshape(csdl.matvec(kp,csdl.matvec(T,d)), (1,12))

            # element output (required for the stress recovery):
            self.register_output(element_name+'local_loads', csdl.matvec(kp,csdl.matvec(T,d)))






        # parse the displacements to get the new nodal coordinates:
        coord = self.create_output('coord',shape=(num_elements,2,6)) # (element,node a/node b,(x,y,z,phi,theta,psi))
        for i, element_name in enumerate(options):
            # get the undeformed nodal coordinates:
            node_a = self.declare_variable(element_name+'node_a',shape=(6))
            node_b = self.declare_variable(element_name+'node_b',shape=(6))

            # get the nodes and the node ID's:
            node_1, node_2 =  options[element_name]['nodes'][0], options[element_name]['nodes'][1]
            node_1_id = [id for node, id in node_id.items() if node == node_1][0]
            node_2_id = [id for node, id in node_id.items() if node == node_2][0]

            # get the nodal displacements for the current element:
            dn1 = U[node_1_id*6:node_1_id*6 + 6] # node 1 displacements
            dn2 = U[node_2_id*6:node_2_id*6 + 6] # node 2 displacements

            # assign the elemental output:
            self.register_output(element_name+'node_a_def',node_a + dn1)
            self.register_output(element_name+'node_b_def',node_b + dn2)

            # assign grouped output:
            coord[i,0,:] = csdl.reshape(node_a + dn1, (1,1,6))
            coord[i,1,:] = csdl.reshape(node_b + dn2, (1,1,6))




        # perform a stress recovery:
        vonmises_stress = self.create_output('vonmises_stress',shape=(num_elements)) # the global element stress vector
        for i, element_name in enumerate(options):

            if options[element_name]['type'] == 'tube': 
                self.add(StressTube(options=options[element_name],name=element_name), name=element_name+'Stress')
                vonmises_stress[i] = self.declare_variable(element_name+'s_vm')

            elif options[element_name]['type'] == 'rect':
                raise NotImplementedError('Error: stress recovery for rectangular beams is not implemented')
            
            elif options[element_name]['type'] == 'box':
                self.add(StressBox(options=options[element_name],name=element_name), name=element_name+'Stress')
                vonmises_stress[i] = self.declare_variable(element_name+'s_vm')
            
            else:
                raise NotImplementedError('Error: stress recovery for [beam type] is not implemented')


        
        # compute the (undeformed) structural cg:
        self.add(Cg(options=options), name='Cg')