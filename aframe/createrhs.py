import numpy as np
import csdl
import python_csdl_backend




class CreateRHS(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']

        # loads: Fx, Fy, Fz, Mx, My, Mz
        loads = self.declare_variable('loads',shape=(6,n),val=0)

        rhs = csdl.reshape(csdl.transpose(loads), new_shape=(6*n))

        self.register_output('rhs', rhs)




if __name__ == '__main__':

    options = {}
    options['n']= n = 3

    sim = python_csdl_backend.Simulator(CreateRHS(options=options))

    loads = np.zeros((6,n))
    loads[1,:] = np.linspace(1,n+1,n)
    print(loads)

    sim['loads'] = loads


    sim.run()

    print(sim['rhs'])