import numpy as np
import csdl
import python_csdl_backend



class LocalStiffness(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('name')
        self.parameters.declare('dim')
        self.parameters.declare('node_list')
    def define(self):
        options = self.parameters['options']
        name = self.parameters['name']
        dim = self.parameters['dim']
        node_list = self.parameters['node_list']

        # nodal indices:
        node_1 =  options['nodes'][0]
        node_2 =  options['nodes'][1]

        # the constant element properties:
        E = options['E']
        G = options['G']

        # the variable element properties:
        A = self.declare_variable(name+'A')
        Iy = self.declare_variable(name+'Iy')
        Iz = self.declare_variable(name+'Iz')
        J = self.declare_variable(name+'J')

        # nodal coordinates:
        node_a = self.declare_variable(name+'node_a',shape=(3))
        node_b = self.declare_variable(name+'node_b',shape=(3))

        # get the element length:
        L = csdl.pnorm(node_b - node_a, pnorm_type=2)

        # compute the nodal stiffness blocks (the four blocks comprising kp):
        k11 = self.create_output(name+'k11',shape=(6,6),val=0) # the upper left block
        k11[0,0] = A*E/L
        k11[1,1] = 12*E*Iz/L**3
        k11[1,5] = k11[5,1] = 6*E*Iz/L**2
        k11[2,2] = 12*E*Iy/L**3
        k11[2,4] = k11[4,2] = -6*E*Iy/L**2
        k11[3,3] = G*J/L
        k11[4,4] = 4*E*Iy/L
        k11[5,5] = 4*E*Iz/L

        k12 = self.create_output(name+'k12',shape=(6,6),val=0) # the upper right block
        k12[0,0] = -A*E/L
        k12[1,1] = -12*E*Iz/L**3
        k12[1,5] = 6*E*Iz/L**2
        k12[2,2] = -12*E*Iy/L**3
        k12[2,4] = -6*E*Iy/L**2
        k12[3,3] = -G*J/L
        k12[4,2] = 6*E*Iy/L**2
        k12[4,4] = 2*E*Iy/L
        k12[5,1] = -6*E*Iz/L**2
        k12[5,5] = 2*E*Iz/L

        k21 = self.create_output(name+'k21',shape=(6,6),val=0) # the lower left block
        k21[0,0] = -A*E/L
        k21[1,1] = -12*E*Iz/L**3
        k21[1,5] = -6*E*Iz/L**2
        k21[2,2] = -12*E*Iy/L**3
        k21[2,4] = 6*E*Iy/L**2
        k21[3,3] = -G*J/L
        k21[4,2] = -6*E*Iy/L**2
        k21[4,4] = 2*E*Iy/L
        k21[5,1] = 6*E*Iz/L**2
        k21[5,5] = 2*E*Iz/L

        k22 = self.create_output(name+'k22',shape=(6,6),val=0) # the lower right block
        k22[0,0] = A*E/L
        k22[1,1] = 12*E*Iz/L**3
        k22[1,5] = k22[5,1] = -6*E*Iz/L**2
        k22[2,2] = 12*E*Iy/L**3
        k22[2,4] = k22[4,2] = 6*E*Iy/L**2
        k22[3,3] = G*J/L
        k22[4,4] = 4*E*Iy/L
        k22[5,5] = 4*E*Iz/L





        # create the local element stiffness matrix using the direct-stiffness method:
        kp = self.create_output(name+'kp',shape=(dim,dim),val=0)


        """
        kp = self.create_output(name+'kp',shape=(12,12),val=0)

        # the upper left block
        kp[0, 0] = csdl.expand(A*E/L, (1,1), 'i->ij')
        kp[1, 1] = csdl.expand(12*E*Iz/(L**3), (1,1), 'i->ij')
        kp[1, 5] = csdl.expand(6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[2, 2] = csdl.expand(12*E*Iy/(L**3), (1,1), 'i->ij')
        kp[2, 4] = csdl.expand(-6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[3, 3] = csdl.expand(G*J/L, (1,1), 'i->ij')
        kp[4, 2] = csdl.expand(-6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[4, 4] = csdl.expand(4*E*Iy/L, (1,1), 'i->ij')
        kp[5, 1] = csdl.expand(6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[5, 5] = csdl.expand(4*E*Iz/L, (1,1), 'i->ij')

        # the upper right block
        kp[0, 6] = csdl.expand(-A*E/L, (1,1), 'i->ij')
        kp[1, 7] = csdl.expand(-12*E*Iz/(L**3), (1,1), 'i->ij')
        kp[1, 11] = csdl.expand(6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[2, 8] = csdl.expand(-12*E*Iy/(L**3), (1,1), 'i->ij')
        kp[2, 10] = csdl.expand(-6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[3, 9] = csdl.expand(-G*J/L, (1,1), 'i->ij')
        kp[4, 8] = csdl.expand(6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[4, 10] = csdl.expand(2*E*Iy/L, (1,1), 'i->ij')
        kp[5, 7] = csdl.expand(-6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[5, 11] = csdl.expand(2*E*Iz/L, (1,1), 'i->ij')

        # the bottom right block
        kp[6, 6] = csdl.expand(A*E/L, (1,1), 'i->ij')
        kp[7, 7] = csdl.expand(12*E*Iz/(L**3), (1,1), 'i->ij')
        kp[7, 11] = csdl.expand(-6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[8, 8] = csdl.expand(12*E*Iy/(L**3), (1,1), 'i->ij')
        kp[8, 10] = csdl.expand(6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[9, 9] = csdl.expand(G*J/L, (1,1), 'i->ij')
        kp[10, 8] = csdl.expand(6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[10, 10] = csdl.expand(4*E*Iy/L, (1,1), 'i->ij')
        kp[11, 7] = csdl.expand(-6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[11, 11] = csdl.expand(4*E*Iz/L, (1,1), 'i->ij')

        # the bottom left block
        kp[6, 0] = csdl.expand(-A*E/L, (1,1), 'i->ij')
        kp[7, 1] = csdl.expand(-12*E*Iz/(L**3), (1,1), 'i->ij')
        kp[7, 5] = csdl.expand(-6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[8, 2] = csdl.expand(-12*E*Iy/(L**3), (1,1), 'i->ij')
        kp[8, 4] = csdl.expand(6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[9, 3] = csdl.expand(-G*J/L, (1,1), 'i->ij')
        kp[10, 2] = csdl.expand(-6*E*Iy/(L**2), (1,1), 'i->ij')
        kp[10, 4] = csdl.expand(2*E*Iy/L, (1,1), 'i->ij')
        kp[11, 1] = csdl.expand(6*E*Iz/(L**2), (1,1), 'i->ij')
        kp[11, 5] = csdl.expand(2*E*Iz/L, (1,1), 'i->ij')
        """









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