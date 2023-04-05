import numpy as np
import csdl
import python_csdl_backend



class LocalStiffness(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']


        E = options['E']
        G = options['G']

        A = self.declare_variable('A',shape=(n-1))
        Iy = self.declare_variable('Iy',shape=(n-1))
        Iz = self.declare_variable('Iz',shape=(n-1))
        J = self.declare_variable('J',shape=(n-1))
        #L = self.declare_variable('L',shape=(n-1))

        nodes = self.declare_variable('nodes',shape=(3,n))

        # compute lengths
        L = self.create_output('L',shape=(n-1))
        for i in range(n-1):
            L[i] = csdl.pnorm(nodes[:,i+1] - nodes[:,i], pnorm_type=2)



        kp = self.create_output('kp',shape=(n-1,12,12),val=0)

        # the upper left block
        kp[:, 0, 0] = csdl.expand(A*E/L, (n-1,1,1), 'i->ijk')
        kp[:, 1, 1] = csdl.expand(12*E*Iz/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 1, 5] = csdl.expand(6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 2, 2] = csdl.expand(12*E*Iy/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 2, 4] = csdl.expand(-6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 3, 3] = csdl.expand(G*J/L, (n-1,1,1), 'i->ijk')
        kp[:, 4, 2] = csdl.expand(-6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 4, 4] = csdl.expand(4*E*Iy/L, (n-1,1,1), 'i->ijk')
        kp[:, 5, 1] = csdl.expand(6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 5, 5] = csdl.expand(4*E*Iz/L, (n-1,1,1), 'i->ijk')

        # the upper right block
        kp[:, 0, 6] = csdl.expand(-A*E/L, (n-1,1,1), 'i->ijk')
        kp[:, 1, 7] = csdl.expand(-12*E*Iz/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 1, 11] = csdl.expand(6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 2, 8] = csdl.expand(-12*E*Iy/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 2, 10] = csdl.expand(-6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 3, 9] = csdl.expand(-G*J/L, (n-1,1,1), 'i->ijk')
        kp[:, 4, 8] = csdl.expand(6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 4, 10] = csdl.expand(2*E*Iy/L, (n-1,1,1), 'i->ijk')
        kp[:, 5, 7] = csdl.expand(-6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 5, 11] = csdl.expand(2*E*Iz/L, (n-1,1,1), 'i->ijk')

        # the bottom right block
        kp[:, 6, 6] = csdl.expand(A*E/L, (n-1,1,1), 'i->ijk')
        kp[:, 7, 7] = csdl.expand(12*E*Iz/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 7, 11] = csdl.expand(-6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 8, 8] = csdl.expand(12*E*Iy/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 8, 10] = csdl.expand(6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 9, 9] = csdl.expand(G*J/L, (n-1,1,1), 'i->ijk')
        kp[:, 10, 8] = csdl.expand(6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 10, 10] = csdl.expand(4*E*Iy/L, (n-1,1,1), 'i->ijk')
        kp[:, 11, 7] = csdl.expand(-6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 11, 11] = csdl.expand(4*E*Iz/L, (n-1,1,1), 'i->ijk')

        # the bottom left block
        kp[:, 6, 0] = csdl.expand(-A*E/L, (n-1,1,1), 'i->ijk')
        kp[:, 7, 1] = csdl.expand(-12*E*Iz/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 7, 5] = csdl.expand(-6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 8, 2] = csdl.expand(-12*E*Iy/(L**3), (n-1,1,1), 'i->ijk')
        kp[:, 8, 4] = csdl.expand(6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 9, 3] = csdl.expand(-G*J/L, (n-1,1,1), 'i->ijk')
        kp[:, 10, 2] = csdl.expand(-6*E*Iy/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 10, 4] = csdl.expand(2*E*Iy/L, (n-1,1,1), 'i->ijk')
        kp[:, 11, 1] = csdl.expand(6*E*Iz/(L**2), (n-1,1,1), 'i->ijk')
        kp[:, 11, 5] = csdl.expand(2*E*Iz/L, (n-1,1,1), 'i->ijk')
            









if __name__ == '__main__':

    options = {}
    options['E'] = 69E9
    options['G'] = 1E20
    options['n'] = 2

    sim = python_csdl_backend.Simulator(LocalStiffness(options=options))
    sim.run()


    kp = sim['kp']

    np.set_printoptions(linewidth=200)
    print(np.round(kp, 2))