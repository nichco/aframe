import numpy as np
import csdl
import python_csdl_backend

from transform import Transform
from localstiffness import LocalStiffness


class AssembleGlobalStiffness(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']

        # compute the local element stiffness matrices
        self.add(LocalStiffness(options=options), name='LocalStiff')

        # transform the local stiffness matrices to global coordinates
        self.add(Transform(options=options), name='Transform')

        k = self.declare_variable('k',shape=(n-1,12,12))

        #kf = self.create_output('kf', shape=((n-1)*12, (n-1)*12))
        #for i in range(n-1):
        #    k_i = csdl.reshape(k[i,:,:], new_shape=(12,12))
        #    kf[i*12:i*12 + 12, i*12:i*12 + 12] = k_i







if __name__ == '__main__':

    options = {}
    options['E'] = 69E9
    options['G'] = 1E20
    options['n'] = 2

    sim = python_csdl_backend.Simulator(AssembleGlobalStiffness(options=options))

    nodes = np.zeros((3,options['n']))
    nodes[0,1] = 2
    sim['nodes'] = nodes



    sim.run()


    #k = sim['k']

    #np.set_printoptions(linewidth=200)
    #print(np.round(k, 2))