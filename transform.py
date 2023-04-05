import numpy as np
import csdl
import python_csdl_backend




class Transform(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']



        nodes = self.declare_variable('nodes', shape=(3,n))
        L = self.declare_variable('L',shape=(n-1))

        # the local stiffness matrix
        kp = self.declare_variable('kp',shape=(n-1,12,12))



        x = csdl.reshape(nodes[0,:], new_shape=(n))
        y = csdl.reshape(nodes[1,:], new_shape=(n))
        z = csdl.reshape(nodes[2,:], new_shape=(n))


        lam = self.create_output('lam',shape=(n-1,3,3),val=0)
        for i in range(n-1):
            cxxp = ll = (x[i+1] - x[i])/L[i]
            cyxp = mm = (y[i+1] - y[i])/L[i]
            czxp = nn = (z[i+1] - z[i])/L[i]

            D = (ll**2 + mm**2)**0.5

            # row 1
            lam[i,0,0] = csdl.expand(ll, (1,1,1),'i->ijk')
            lam[i,0,1] = csdl.expand(mm, (1,1,1),'i->ijk')
            lam[i,0,2] = csdl.expand(nn, (1,1,1),'i->ijk')
            # row 2
            lam[i,1,0] = csdl.expand(-mm/D, (1,1,1),'i->ijk')
            lam[i,1,1] = csdl.expand(ll/D, (1,1,1),'i->ijk')
            # row 3
            lam[i,2,0] = csdl.expand(-ll*nn/D, (1,1,1),'i->ijk')
            lam[i,2,1] = csdl.expand(-mm*nn/D, (1,1,1),'i->ijk')
            lam[i,2,2] = csdl.expand(D, (1,1,1),'i->ijk')




        # the transformation matrix T
        T = self.create_output('T',shape=(n-1,12,12),val=0)
        for i in range(n-1):
            T[i,0:3,0:3] = 1*lam
            T[i,3:6,3:6] = 1*lam
            T[i,6:9,6:9] = 1*lam
            T[i,9:12,9:12] = 1*lam


        # output the global stiffness matrix
        k = self.create_output('k',shape=(n-1,12,12),val=0)
        for i in range(n-1):
            T_i = csdl.reshape(T[i,:,:], new_shape=(12,12))
            kp_i = csdl.reshape(kp[i,:,:], new_shape=(12,12))

            k[i,:,:] = csdl.expand(csdl.matmat(csdl.transpose(T_i), csdl.matmat(kp_i, T_i)), (1,12,12), 'ij->aij')
