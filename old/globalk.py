import numpy as np
import csdl



class GlobalK(csdl.Model):
    def initialize(self):
        self.parameters.declare('dim')
        self.parameters.declare('elements')
        self.parameters.declare('bounds')
        self.parameters.declare('node_index')
        self.parameters.declare('nodes')

    def define(self):
        dim = self.parameters['dim']
        elements = self.parameters['elements']
        bounds = self.parameters['bounds']
        node_index = self.parameters['node_index']
        nodes = self.parameters['nodes']


        # construct the global stiffness matrix:
        helper = self.create_output('helper',shape=(len(elements),dim,dim),val=0)
        for i, element_name in enumerate(elements):
            k = self.declare_variable(element_name+'k',shape=(dim,dim))
            helper[i,:,:] = csdl.expand(k, (1,dim,dim), 'ij->aij')

        sum_k = csdl.sum(helper, axes=(0, ))


        b_index_list = []
        for b_name in bounds:
            beam_name = bounds[b_name]['beam']
            fpos = bounds[b_name]['fpos']
            fdim = bounds[b_name]['fdim']
            if fpos == 'a': b_node = nodes[beam_name][0]
            elif fpos == 'b': b_node = nodes[beam_name][-1]

            b_node_index = node_index[b_node]

            # add the constrained dof index to the b_index_list:
            for i, fdim in enumerate(fdim):
                if fdim == 1: b_index_list.append(b_node_index*6 + i)

        


        mask = self.create_output('mask',shape=(dim,dim),val=np.eye(dim))
        mask_eye = self.create_output('mask_eye',shape=(dim,dim),val=0)
        zero = self.create_input('zero',shape=(1,1),val=0)
        one = self.create_input('one',shape=(1,1),val=1)
        [(mask.__setitem__((i,i),1*zero), mask_eye.__setitem__((i,i),1*one)) for i in range(dim) if i in b_index_list]



        # modify the global stiffness matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye
        self.register_output('K', K)