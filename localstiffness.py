import numpy as np
import csdl
import python_csdl_backend



class LocalStiffness(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('name')
        self.parameters.declare('dim')
        self.parameters.declare('node_id')
    def define(self):
        options = self.parameters['options']
        name = self.parameters['name']
        dim = self.parameters['dim']
        node_id = self.parameters['node_id']

        # nodal indices:
        node_1 =  options['nodes'][0]
        node_2 =  options['nodes'][1]

        # node ID's:
        node_1_id = [id for node, id in node_id.items() if node == node_1][0]
        node_2_id = [id for node, id in node_id.items() if node == node_2][0]
        print('ID1: ',node_1_id)
        print('ID2: ',node_2_id)

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
        k11[0,0] = csdl.expand(A*E/L, (1,1), 'i->ij')
        k11[1,1] = csdl.expand(12*E*Iz/L**3, (1,1), 'i->ij')
        k11[1,5] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        k11[5,1] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        k11[2,2] = csdl.expand(12*E*Iy/L**3, (1,1), 'i->ij')
        k11[2,4] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        k11[4,2] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        k11[3,3] = csdl.expand(G*J/L, (1,1), 'i->ij')
        k11[4,4] = csdl.expand(4*E*Iy/L, (1,1), 'i->ij')
        k11[5,5] = csdl.expand(4*E*Iz/L, (1,1), 'i->ij')

        k12 = self.create_output(name+'k12',shape=(6,6),val=0) # the upper right block
        k12[0,0] = csdl.expand(-A*E/L, (1,1), 'i->ij')
        k12[1,1] = csdl.expand(-12*E*Iz/L**3, (1,1), 'i->ij')
        k12[1,5] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        k12[2,2] = csdl.expand(-12*E*Iy/L**3, (1,1), 'i->ij')
        k12[2,4] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        k12[3,3] = csdl.expand(-G*J/L, (1,1), 'i->ij')
        k12[4,2] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        k12[4,4] = csdl.expand(2*E*Iy/L, (1,1), 'i->ij')
        k12[5,1] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        k12[5,5] = csdl.expand(2*E*Iz/L, (1,1), 'i->ij')

        k21 = self.create_output(name+'k21',shape=(6,6),val=0) # the lower left block
        k21[0,0] = csdl.expand(-A*E/L, (1,1), 'i->ij')
        k21[1,1] = csdl.expand(-12*E*Iz/L**3, (1,1), 'i->ij')
        k21[1,5] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        k21[2,2] = csdl.expand(-12*E*Iy/L**3, (1,1), 'i->ij')
        k21[2,4] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        k21[3,3] = csdl.expand(-G*J/L, (1,1), 'i->ij')
        k21[4,2] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        k21[4,4] = csdl.expand(2*E*Iy/L, (1,1), 'i->ij')
        k21[5,1] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        k21[5,5] = csdl.expand(2*E*Iz/L, (1,1), 'i->ij')

        k22 = self.create_output(name+'k22',shape=(6,6),val=0) # the lower right block
        k22[0,0] = csdl.expand(A*E/L, (1,1), 'i->ij')
        k22[1,1] = csdl.expand(12*E*Iz/L**3, (1,1), 'i->ij')
        k22[1,5] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        k22[5,1] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        k22[2,2] = csdl.expand(12*E*Iy/L**3, (1,1), 'i->ij')
        k22[2,4] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        k22[4,2] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        k22[3,3] = csdl.expand(G*J/L, (1,1), 'i->ij')
        k22[4,4] = csdl.expand(4*E*Iy/L, (1,1), 'i->ij')
        k22[5,5] = csdl.expand(4*E*Iz/L, (1,1), 'i->ij')





        # create the local element stiffness matrix using the direct-stiffness method:
        kp = self.create_output(name+'kp',shape=(dim,dim),val=0)

        # assign the four block matrices to their respective positions in kp
        row_i = node_1_id*6
        row_f = node_1_id*6 + 6
        col_i = node_1_id*6
        col_f = node_1_id*6 + 6
        kp[row_i:row_f, col_i:col_f] = k11


        row_i = node_1_id*6
        row_f = node_1_id*6 + 6
        col_i = node_2_id*6
        col_f = node_2_id*6 + 6
        kp[row_i:row_f, col_i:col_f] = k12


        row_i = node_2_id*6
        row_f = node_2_id*6 + 6
        col_i = node_1_id*6
        col_f = node_1_id*6 + 6
        kp[row_i:row_f, col_i:col_f] = k21


        row_i = node_2_id*6
        row_f = node_2_id*6 + 6
        col_i = node_2_id*6
        col_f = node_2_id*6 + 6
        kp[row_i:row_f, col_i:col_f] = k22









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