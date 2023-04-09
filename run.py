import numpy as np
import csdl
import python_csdl_backend
from sectionpropertiestube import SectionPropertiesTube
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
        num_elements = len(options)

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

                    for i, fdim in enumerate(bcond[name]['fdim']):
                        if fdim == 1:
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

        

        # create the global loads vector
        F = self.declare_variable('F',shape=(dim),val=np.array([0,0,0,0,0,0,100000,0,0,0,0,0]))


        # solve the linear system
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()

        U = solve_res(K, F)



        # recover the local elemental forces/moments (fp):
        fp = self.create_output('fp',shape=(num_elements,12),val=0)
        for i, element_name in enumerate(options):
            d = U[i*12:i*12 + 12]
            kp = self.declare_variable(element_name+'kp',shape=(12,12))
            T = self.declare_variable(element_name+'T',shape=(12,12))
            fp[i,:] = csdl.reshape(csdl.matvec(kp,csdl.matvec(T,d)), (1,12))


        # perform a stress recovery
        







if __name__ == '__main__':

    options = {}

    name = 'element_1'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['nodes'] = [0,1] # node indices for [node_a, node_b]
    options[name]['node_a'] = [0,0,0] # node_a coordinates
    options[name]['node_b'] = [3,0,0] # node_b coordinates
    options[name]['type'] = 'tube' # element type


    bcond = {}

    name = 'root'
    bcond[name] = {}
    bcond[name]['node'] = 0
    bcond[name]['fdim'] = [1,1,1,1,1,1] # x, y, z, phi, theta, psi: a 1 indicates the corresponding dof is fixed





    sim = python_csdl_backend.Simulator(Run(options=options,bcond=bcond))
    sim.run()


    Ks = np.array(sim['K'])
    #print(Ks)

    U = sim['U']
    print(U)
