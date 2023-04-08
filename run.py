import numpy as np
import csdl
import python_csdl_backend
from sectionpropertiestube import SectionPropertiesTube
from transform import Transform
from localstiffness import LocalStiffness
from model import Model



class Run(csdl.Model):
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

        # create a dictionary that contains the nodes and the node index
        node_id = {node_list[i]: i for i in range(num_unique_nodes)}


        # create nodal inputs for each element:
        for element_name in options:
            self.create_input(element_name+'node_a',shape=(3),val=options[element_name]['node_a'])
            self.create_input(element_name+'node_b',shape=(3),val=options[element_name]['node_b'])


        # compute the section properties for each element:
        for element_name in options:
            if options[element_name]['type'] == 'tube': 
                self.add(SectionPropertiesTube(name=element_name), name=element_name+'SectionPropertiesTube')
            elif options[element_name]['type'] == 'box': 
                raise NotImplementedError('Error: type box for ' + element_name + ' is not implemented')
            else: raise NotImplementedError('Error: type for' + element_name + 'is not implemented')


        # compute the local stiffness matrix for each element:
        dim = num_unique_nodes*6
        for element_name in options:
            self.add(LocalStiffness(options=options[element_name],name=element_name,dim=dim,node_id=node_id), name=element_name+'LocalStiffness')


        # transform the local stiffness matrices to global coordinates:
        for element_name in options:
            self.add(Transform(options=options[element_name],name=element_name), name=element_name+'Transform')


        # construct the global stiffness matrix:
        helper = self.create_output('helper',shape=(len(options),dim,dim),val=0)
        for i, element_name in enumerate(options):
            kp = self.declare_variable(element_name+'kp',shape=(dim,dim))
            helper[i,:,:] = csdl.expand(kp, (1,dim,dim), 'ij->aij')

        K = csdl.sum(helper, axes=(0, ))
        self.register_output('K', K)


        # boundary conditions
        #one = self.create_input('one',val=1)
        #mask = self.create_output('mask',shape=(dim,dim),val=0)
        #for i in range(dim):
        #    mask[i,i] = csdl.reshape(one, (1,1))

        #for i in range(dim):
            
        #    if i in bcond['nodes']:


        

        # create the global loads vector
        F = self.declare_variable('F',shape=(dim),val=0)


        # solve the linear system
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()

        solve_res(K, F)


        # recover the elemental forces


        # perform a stress recovery
        







if __name__ == '__main__':

    options = {}

    name = 'element_1'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 1E20
    options[name]['nodes'] = [0,1] # node indices for [node_a, node_b]
    options[name]['node_a'] = [0,0,0] # node_a coordinates
    options[name]['node_b'] = [1,1,0] # node_b coordinates
    options[name]['type'] = 'tube' # element type


    bcond = {}

    name = 'root'
    bcond[name] = {}
    bcond[name]['node'] = 0
    bcond[name]['fdim'] = [1,1,1,1,1,1] # x, y, z, phi, theta, psi: a 1 indicates the corresponding dof is fixed





    sim = python_csdl_backend.Simulator(Run(options=options,bcond=bcond))
    sim.run()


    #K = sim['K']
    #print(K[0,:])
