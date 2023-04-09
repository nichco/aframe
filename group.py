import numpy as np
import csdl
import python_csdl_backend
from sectionpropertiestube import SectionPropertiesTube
from localstiffness import LocalStiffness
from model import Model
from stress import StressTube



class Group(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('bcond')
    def define(self):
        options = self.parameters['options']
        bcond = self.parameters['bcond']


        # process the options dictionary to compute the total number of unique nodes:
        node_list = [options[name]['nodes'][0] for name in options] + [options[name]['nodes'][1] for name in options]
        node_list = [*set(node_list)]
        num_unique_nodes = len(node_list)
        num_elements = len(options)
        dim = num_unique_nodes*6


        # create a dictionary that contains the nodes and the node index
        node_id = {node_list[i]: i for i in range(num_unique_nodes)}


        # create nodal inputs for each element:
        for element_name in options:
            self.create_input(element_name+'node_a',shape=(6),val=options[element_name]['node_a'])
            self.create_input(element_name+'node_b',shape=(6),val=options[element_name]['node_b'])


        # compute the section properties for each element:
        for element_name in options:
            if options[element_name]['type'] == 'tube': 
                self.add(SectionPropertiesTube(name=element_name), name=element_name+'SectionPropertiesTube')
            elif options[element_name]['type'] == 'box': 
                raise NotImplementedError('Error: type box for ' + element_name + ' is not implemented')
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
            # check if the current node has a boundary condition:
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



        K = csdl.transpose(csdl.matmat(mask, csdl.transpose(csdl.matmat(mask,sum_k)))) + mask_eye
        self.register_output('K', K)

        

        # declare the global loads vector
        F = self.declare_variable('F',shape=(dim))


        # solve the linear system
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=200,
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
            d[0:6] = dn1
            d[6:12] = dn2

            # declare the variables for the local stiffness matrix and the element transformation matrix:
            kp = self.declare_variable(element_name+'kp',shape=(12,12))
            T = self.declare_variable(element_name+'T',shape=(12,12))

            # solve for the local loads:
            # group output:
            local_loads[i,:] = csdl.reshape(csdl.matvec(kp,csdl.matvec(T,d)), (1,12))

            # element output:
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

            # get the displacements:
            # get the nodal displacements for the current element:
            dn1 = U[node_1_id*6:node_1_id*6 + 6] # node 1 displacements
            dn2 = U[node_2_id*6:node_2_id*6 + 6] # node 2 displacements

            # assign grouped output:
            coord[i,0,:] = csdl.reshape(node_a + dn1, (1,1,6))
            coord[i,1,:] = csdl.reshape(node_b + dn2, (1,1,6))




        # perform a stress recovery
        vonmises_stress = self.create_output('vonmises_stress',shape=(num_elements))
        for i, element_name in enumerate(options):
            self.add(StressTube(options=options[element_name],name=element_name), name=element_name+'Stress')
            vonmises_stress[i] = self.declare_variable(element_name+'s_vm')