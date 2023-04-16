import numpy as np
import csdl
import python_csdl_backend



class MassProp(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']

        # calculate the mass and the position of each element:
        rm_vec = self.create_output('rm_vec',shape=(len(options),3),val=0)
        rm_vec_def = self.create_output('rm_vec_def',shape=(len(options),3),val=0)
        m_vec = self.create_output('m_vec',shape=(len(options)),val=0)
        for i, element_name in enumerate(options):
            element = options[element_name]
            rho = element['rho']

            A = self.declare_variable(element_name+'A')
            L = self.declare_variable(element_name+'L')

            # compute the element volume:
            V = A*L

            # compute the element mass:
            m = V*rho
            self.register_output(element_name+'m', m)


            # get the (undeformed) position vector of the cg for each element:
            r_a = self.declare_variable(element_name+'node_a',shape=(6))[0:3]
            r_b = self.declare_variable(element_name+'node_b',shape=(6))[0:3]

            # get the (undeformed) position vector of the cg for each element:
            r_a_def = self.declare_variable(element_name+'node_a_def',shape=(6))[0:3]
            r_b_def = self.declare_variable(element_name+'node_b_def',shape=(6))[0:3]

            r_cg = (r_a + r_b)/2
            r_cg_def = (r_a_def + r_b_def)/2
            self.register_output(element_name+'r_cg', r_cg)

            # assign r_cg to the r*mass vector:
            rm_vec[i,:] = csdl.reshape(r_cg*csdl.expand(m, (3)), new_shape=(1,3))
            rm_vec_def[i,:] = csdl.reshape(r_cg_def*csdl.expand(m, (3)), new_shape=(1,3))
            # assign the mass to the mass vector:
            m_vec[i] = m



        # compute the center of gravity for the entire structure:
        total_mass = csdl.sum(m_vec)
        self.register_output('total_mass', total_mass)
        sum_rm = csdl.sum(rm_vec,axes=(0,))
        sum_rm_def = csdl.sum(rm_vec_def,axes=(0,))

        cg = sum_rm/csdl.expand(total_mass, (3))
        self.register_output('cg',cg)

        cg_def = sum_rm_def/csdl.expand(total_mass, (3))
        self.register_output('cg_def',cg_def)



        
        # compute moments of inertia:
        eixx = self.create_output('eixx',shape=(len(options)),val=0)
        eiyy = self.create_output('eiyy',shape=(len(options)),val=0)
        eizz = self.create_output('eizz',shape=(len(options)),val=0)
        eixz = self.create_output('eixz',shape=(len(options)),val=0)
        for i, element_name in enumerate(options):

            # get the mass:
            m = m_vec[i]

            # get the position vector:
            r = self.declare_variable(element_name+'r_cg',shape=(3))
            x = r[0]
            y = r[1]
            z = r[2]

            rxx = y**2 + z**2
            ryy = x**2 + z**2
            rzz = x**2 + y**2
            rxz = x*z


            eixx[i] = m*rxx
            eiyy[i] = m*ryy
            eizz[i] = m*rzz
            eixz[i] = m*rxz
            
        
        # sum the m*r vector to get the moi:
        Ixx = csdl.sum(eixx)
        Iyy = csdl.sum(eiyy)
        Izz = csdl.sum(eizz)
        Ixz = csdl.sum(eixz)
        self.register_output('Ixx',Ixx)
        self.register_output('Iyy',Iyy)
        self.register_output('Izz',Izz)
        self.register_output('Ixz',Ixz)

        # self.print_var(Iyy)



        
            

        