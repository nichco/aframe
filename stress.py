import numpy as np
import csdl
import python_csdl_backend




class StressTube(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('name')
    def define(self):
        options = self.parameters['options']
        name = self.parameters['name']
        E = options['E']
        G = options['G']

        A = self.declare_variable(name+'A')
        J = self.declare_variable(name+'J')
        Iy = self.declare_variable(name+'Iy')
        r = self.declare_variable(name+'radius')

        # get the local loads:
        local_loads = self.declare_variable(name+'local_loads',shape=(12))



        # compute the normal stress:
        normal_force = local_loads[0]
        s_normal = normal_force/A

        
        # compute the torsional stress:
        t = local_loads[3]
        tau = t*r/J

        # compute the bending stress:
        bend_moment_1 = local_loads[4]
        bend_moment_2 = local_loads[5]

        net_moment = (bend_moment_1**2 + bend_moment_2**2)**0.5

        s_bend = net_moment*r/Iy # note: Iy = Iz for a tube

        # sum the bending and normal stresses:
        s_axial = s_normal + s_bend


        # compute the von-mises stress
        s_vm = (s_axial**2 + 3*tau**2)**0.5

        self.register_output(name+'s_vm', s_vm)
        