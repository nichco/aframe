import numpy as np
import csdl



class LocalK(csdl.Model):
    def initialize(self):
        self.parameters.declare('elements')
        self.parameters.declare('dim')
        self.parameters.declare('node_index')

    def define(self):
        elements = self.parameters['elements']
        dim = self.parameters['dim']
        node_index = self.parameters['node_index']



        # compute the local stiffness matrix for every element:
        for element_name in elements:
            # get the material properties for the element:
            E = self.declare_variable(element_name+'E')
            G = self.declare_variable(element_name+'G')
            A = self.declare_variable(element_name+'A')
            Iy = self.declare_variable(element_name+'Iy')
            Iz = self.declare_variable(element_name+'Iz')
            J = self.declare_variable(element_name+'J')
            #L = self.declare_variable(element_name+'L')
            node_a_position = self.declare_variable(element_name+'node_a_position',shape=(3))
            node_b_position = self.declare_variable(element_name+'node_b_position',shape=(3))
            L = csdl.pnorm(node_b_position - node_a_position, pnorm_type=2)
            self.register_output(element_name+'L', L)

            kp = self.create_output(element_name+'kp',shape=(12,12),val=0)
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


            # transform the local stiffness to global coordinates:

            cp = (node_b_position - node_a_position)/csdl.expand(L, (3))
            ll = cp[0]
            mm = cp[1]
            nn = cp[2]
            D = (ll**2 + mm**2)**0.5

            block = self.create_output(element_name+'block',shape=(3,3),val=0)
            block[0,0] = csdl.reshape(ll, (1,1))
            block[0,1] = csdl.reshape(mm, (1,1))
            block[0,2] = csdl.reshape(nn, (1,1))
            block[1,0] = csdl.reshape(-mm/D, (1,1))
            block[1,1] = csdl.reshape(ll/D, (1,1))
            block[2,0] = csdl.reshape(-ll*nn/D, (1,1))
            block[2,1] = csdl.reshape(-mm*nn/D, (1,1))
            block[2,2] = csdl.reshape(D, (1,1))

            # construct the block-diagonal transformation matrix
            T = self.create_output(element_name+'T',shape=(12,12),val=0)
            T[0:3,0:3] = 1*block
            T[3:6,3:6] = 1*block
            T[6:9,6:9] = 1*block
            T[9:12,9:12] = 1*block

            tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))
            

            # expand the transformed stiffness matrix to the global dimensions:
            k = self.create_output(element_name+'k',shape=(dim,dim),val=0)

            # parse tkt:
            k11 = tkt[0:6,0:6] # upper left
            k12 = tkt[0:6,6:12] # upper right
            k21 = tkt[6:12,0:6] # lower left
            k22 = tkt[6:12,6:12] # lower right

            # assign the four block matrices to their respective positions in k:
            node_a = elements[element_name]['node_a']
            node_b = elements[element_name]['node_b']
            node_a_index = node_index[node_a]
            node_b_index = node_index[node_b]

            row_i = node_a_index*6
            row_f = node_a_index*6 + 6
            col_i = node_a_index*6
            col_f = node_a_index*6 + 6
            k[row_i:row_f, col_i:col_f] = k11

            row_i = node_a_index*6
            row_f = node_a_index*6 + 6
            col_i = node_b_index*6
            col_f = node_b_index*6 + 6
            k[row_i:row_f, col_i:col_f] = k12

            row_i = node_b_index*6
            row_f = node_b_index*6 + 6
            col_i = node_a_index*6
            col_f = node_a_index*6 + 6
            k[row_i:row_f, col_i:col_f] = k21

            row_i = node_b_index*6
            row_f = node_b_index*6 + 6
            col_i = node_b_index*6
            col_f = node_b_index*6 + 6
            k[row_i:row_f, col_i:col_f] = k22
        
