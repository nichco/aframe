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


        # the constant element properties:
        E = options['E']
        G = options['G']

        # the variable element properties:
        A = self.declare_variable(name+'A')
        Iy = self.declare_variable(name+'Iy')
        Iz = self.declare_variable(name+'Iz')
        J = self.declare_variable(name+'J')

        # nodal coordinates:
        node_a = self.declare_variable(name+'node_a',shape=(6))
        node_b = self.declare_variable(name+'node_b',shape=(6))

        # get the element length:
        L = csdl.pnorm(node_b[0:3] - node_a[0:3], pnorm_type=2)
        #self.print_var(L)


        kp = self.create_output(name+'kp',shape=(12,12),val=0)
        # the upper left block
        kp[0,0] = csdl.expand(A*E/L, (1,1), 'i->ij')
        kp[1,1] = csdl.expand(12*E*Iz/L**3, (1,1), 'i->ij')
        kp[1,5] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        kp[5,1] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        kp[2,2] = csdl.expand(12*E*Iy/L**3, (1,1), 'i->ij')
        kp[2,4] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        kp[4,2] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        kp[3,3] = csdl.expand(G*J/L, (1,1), 'i->ij')
        kp[4,4] = csdl.expand(4*E*Iy/L, (1,1), 'i->ij')
        kp[5,5] = csdl.expand(4*E*Iz/L, (1,1), 'i->ij')
        # the upper right block
        kp[0,6] = csdl.expand(-A*E/L, (1,1), 'i->ij')
        kp[1,7] = csdl.expand(-12*E*Iz/L**3, (1,1), 'i->ij')
        kp[1,11] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        kp[2,8] = csdl.expand(-12*E*Iy/L**3, (1,1), 'i->ij')
        kp[2,10] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        kp[3,9] = csdl.expand(-G*J/L, (1,1), 'i->ij')
        kp[4,8] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        kp[4,10] = csdl.expand(2*E*Iy/L, (1,1), 'i->ij')
        kp[5,7] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        kp[5,11] = csdl.expand(2*E*Iz/L, (1,1), 'i->ij')
        # the lower left block
        kp[6,0] = csdl.expand(-A*E/L, (1,1), 'i->ij')
        kp[7,1] = csdl.expand(-12*E*Iz/L**3, (1,1), 'i->ij')
        kp[7,5] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        kp[8,2] = csdl.expand(-12*E*Iy/L**3, (1,1), 'i->ij')
        kp[8,4] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        kp[9,3] = csdl.expand(-G*J/L, (1,1), 'i->ij')
        kp[10,2] = csdl.expand(-6*E*Iy/L**2, (1,1), 'i->ij')
        kp[10,4] = csdl.expand(2*E*Iy/L, (1,1), 'i->ij')
        kp[11,1] = csdl.expand(6*E*Iz/L**2, (1,1), 'i->ij')
        kp[11,5] = csdl.expand(2*E*Iz/L, (1,1), 'i->ij')
        # the lower right block
        kp[6,6] = csdl.expand(A*E/L, (1,1), 'i->ij')
        kp[7,7] = csdl.expand(12*E*Iz/L**3, (1,1), 'i->ij')
        kp[7,11] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        kp[11,7] = csdl.expand(-6*E*Iz/L**2, (1,1), 'i->ij')
        kp[8,8] = csdl.expand(12*E*Iy/L**3, (1,1), 'i->ij')
        kp[8,10] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        kp[10,8] = csdl.expand(6*E*Iy/L**2, (1,1), 'i->ij')
        kp[9,9] = csdl.expand(G*J/L, (1,1), 'i->ij')
        kp[10,10] = csdl.expand(4*E*Iy/L, (1,1), 'i->ij')
        kp[11,11] = csdl.expand(4*E*Iz/L, (1,1), 'i->ij')




        # transformation
        x1 = node_a[0]
        y1 = node_a[1]
        z1 = node_a[2]

        x2 = node_b[0]
        y2 = node_b[1]
        z2 = node_b[2]

        cxxp = ll = (x2 - x1)/L
        cyxp = mm = (y2 - y1)/L
        czxp = nn = (z2 - z1)/L

        D = (ll**2 + mm**2)**0.5

        lam = self.create_output(name+'lam',shape=(3,3),val=0)
        # row 1
        lam[0,0] = csdl.expand(ll, (1,1),'i->ij')
        lam[0,1] = csdl.expand(mm, (1,1),'i->ij')
        lam[0,2] = csdl.expand(nn, (1,1),'i->ij')
        # row 2
        lam[1,0] = csdl.expand(-mm/D, (1,1),'i->ij')
        lam[1,1] = csdl.expand(ll/D, (1,1),'i->ij')
        # row 3
        lam[2,0] = csdl.expand(-ll*nn/D, (1,1),'i->ij')
        lam[2,1] = csdl.expand(-mm*nn/D, (1,1),'i->ij')
        lam[2,2] = csdl.expand(D, (1,1),'i->ij')

        # construct the block-diagonal transformation matrix
        T = self.create_output(name+'T',shape=(12,12),val=0)
        T[0:3,0:3] = 1*lam
        T[3:6,3:6] = 1*lam
        T[6:9,6:9] = 1*lam
        T[9:12,9:12] = 1*lam



        tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))

        #self.print_var(tkt)

        # expand the transformed stiffness matrix to the global dimensions:
        k = self.create_output(name+'k',shape=(dim,dim),val=0)

        k11 = tkt[0:6,0:6] # upper left
        k12 = tkt[0:6,6:12] # upper right
        k21 = tkt[6:12,0:6] # lower left
        k22 = tkt[6:12,6:12] # lower right


        # assign the four block matrices to their respective positions in k:
        row_i = node_1_id*6
        row_f = node_1_id*6 + 6
        col_i = node_1_id*6
        col_f = node_1_id*6 + 6
        k[row_i:row_f, col_i:col_f] = k11


        row_i = node_1_id*6
        row_f = node_1_id*6 + 6
        col_i = node_2_id*6
        col_f = node_2_id*6 + 6
        k[row_i:row_f, col_i:col_f] = k12


        row_i = node_2_id*6
        row_f = node_2_id*6 + 6
        col_i = node_1_id*6
        col_f = node_1_id*6 + 6
        k[row_i:row_f, col_i:col_f] = k21


        row_i = node_2_id*6
        row_f = node_2_id*6 + 6
        col_i = node_2_id*6
        col_f = node_2_id*6 + 6
        k[row_i:row_f, col_i:col_f] = k22









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