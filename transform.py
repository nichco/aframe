import numpy as np
import csdl
import python_csdl_backend




class Transform(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
        self.parameters.declare('options')
    def define(self):
        name = self.parameters['name']
        options = self.parameters['options']

        node_a = self.declare_variable(name+'node_a',shape=(3))
        node_b = self.declare_variable(name+'node_b',shape=(3))

        x1 = node_a[0]
        y1 = node_a[1]
        z1 = node_a[2]

        x2 = node_b[0]
        y2 = node_b[1]
        z2 = node_b[2]

        L = csdl.pnorm(node_b - node_a, pnorm_type=2)
        

        # the local stiffness matrix
        kp = self.declare_variable(name+'kp',shape=(12,12))


        cxxp = ll = (x2 - x1)/L
        cyxp = mm = (y2 - y1)/L
        czxp = nn = (z2 - z1)/L

        D = (ll**2 + mm**2)**0.5

        lam = self.create_output('lam',shape=(3,3),val=0)
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
        T = self.create_output('T',shape=(12,12),val=0)
        T[0:3,0:3] = 1*lam
        T[3:6,3:6] = 1*lam
        T[6:9,6:9] = 1*lam
        T[9:12,9:12] = 1*lam


        # compute the global element stiffness matrix:
        k = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))
        self.register_output(name+'k',k)