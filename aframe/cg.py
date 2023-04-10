import numpy as np
import csdl
import python_csdl_backend



class Cg(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']

        # calculate the mass and the position of each element:
        rm_vec = self.create_output('rm_vec',shape=(len(options),3),val=0)
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


            # calculate the position vector of the cg for each element:
            r_a = self.declare_variable(element_name+'node_a',shape=(6))[0:3]
            r_b = self.declare_variable(element_name+'node_b',shape=(6))[0:3]

            r_cg = (r_a + r_b)/2
            self.register_output(element_name+'r_cg', r_cg)

            # assign r_cg to the r*mass vector:
            rm_vec[i,:] = csdl.reshape(r_cg*csdl.expand(m, (3)), new_shape=(1,3))
            # assign the mass to the mass vector:
            m_vec[i] = m



        # compute the center of gravity for the entire structure:
        total_mass = csdl.sum(m_vec)
        sum_rm = csdl.sum(rm_vec,axes=(0,))

        cg = sum_rm/csdl.expand(total_mass, (3))
        self.register_output('cg',cg)

        #self.print_var(cg)

        
            

        