import numpy as np
import csdl
import python_csdl_backend



coeffs_2 = np.array(
    [
        [1.0, -1.0],
        [-1.0, 1.0],
    ]
)

coeffs_y = np.array(
    [
        [12.0, -6.0, -12.0, -6.0],
        [-6.0, 4.0, 6.0, 2.0],
        [-12.0, 6.0, 12.0, 6.0],
        [-6.0, 2.0, 6.0, 4.0],
    ]
)

coeffs_z = np.array(
    [
        [12.0, 6.0, -12.0, 6.0],
        [6.0, 4.0, -6.0, 2.0],
        [-12.0, -6.0, 12.0, -6.0],
        [6.0, 2.0, -6.0, 4.0],
    ]
)



class LocalStiff(csdl.Model):
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
        L = self.declare_variable('L',shape=(n-1))


        local_stiff = self.create_output('local_stiff',shape=(n-1,12,12),val=0)
        for i in range(2):
            for j in range(2):
                local_stiff[:, 0 + i, 0 + j] = csdl.expand(E * A / L * coeffs_2[i, j], (n-1,1,1), 'i->ijk')
                local_stiff[:, 2 + i, 2 + j] = csdl.expand(G * J / L * coeffs_2[i, j], (n-1,1,1), 'i->ijk')

        for i in range(4):
            for j in range(4):
                local_stiff[:, 4 + i, 4 + j] = csdl.expand(E * Iy / L**3 * coeffs_y[i, j], (n-1,1,1), 'i->ijk')
                local_stiff[:, 8 + i, 8 + j] = csdl.expand(E * Iz / L**3 * coeffs_z[i, j], (n-1,1,1), 'i->ijk')

        for i in [1, 3]:
            for j in range(4):
                local_stiff[:, 4 + i, 4 + j] *= L
                local_stiff[:, 8 + i, 8 + j] *= L
        
        for i in range(4):
            for j in [1, 3]:
                local_stiff[:, 4 + i, 4 + j] *= L
                local_stiff[:, 8 + i, 8 + j] *= L







if __name__ == '__main__':

    options = {}
    options['E'] = 69E9
    options['G'] = 1E20
    options['n'] = 10

    sim = python_csdl_backend.Simulator(LocalStiff(options=options))
    sim.run()