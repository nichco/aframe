import numpy as np
import csdl




class StressTube(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
    def define(self):
        name = self.parameters['name']

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

        # compute the maximum von-mises stress
        s_vm = (s_axial**2 + 3*tau**2)**0.5

        self.register_output(name+'s_vm', s_vm)







"""
the stress for box beams is evaluated at four points:
    1 ------------------------------------- 2
      -                y                  -
      -                |                  -
      -                --> x              -
      -                                   -
      -                                   -
    4 ------------------------------------- 3
"""
class StressBox(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
    def define(self):
        name = self.parameters['name']

        A = self.declare_variable(name+'A')
        J = self.declare_variable(name+'J')
        Iy = self.declare_variable(name+'Iy')
        Iz = self.declare_variable(name+'Iz')

        w = self.declare_variable(name+'width')
        h = self.declare_variable(name+'height')

        # get the local loads:
        local_loads = self.declare_variable(name+'local_loads',shape=(12))

        # create the point coordinate matrix
        x_coord = self.create_output(name+'x_coord',shape=(4),val=0)
        y_coord = self.create_output(name+'y_coord',shape=(4),val=0)
        # point 1
        x_coord[0] = -w/2
        y_coord[0] = h/2
        # point 2
        x_coord[1] = w/2
        y_coord[1] = h/2
        # point 3
        x_coord[2] = w/2
        y_coord[2] = -h/2
        # point 4
        x_coord[3] = -w/2
        y_coord[3] = -h/2

        # compute the normal stress (same at all four points):
        normal_force = local_loads[0]
        s_normal = normal_force/A


        # compute the bending stress at each point:
        bend_moment_1 = local_loads[4]
        bend_moment_2 = local_loads[5]
        bend_stress = self.create_output(name+'bend_stress',shape=(4),val=0)
        for point in range(4):
            x = x_coord[point]
            y = y_coord[point]
            bend_stress[point] = (bend_moment_1*y/Iy) + (bend_moment_2*x/Iz)

        # take the maximum bending stress:
        s_bend = csdl.max(bend_stress)


        # compute the torsional stress:
        r = ((w/2)**2 + (h/2)**2)**0.5
        t = local_loads[3]
        tau = t*r/J


        # sum the max bending stress and normal stress:
        s_axial = s_normal + s_bend
        

        # compute the maximum von-mises stress
        s_vm = (s_axial**2 + 3*tau**2)**0.5

        self.register_output(name+'s_vm', s_vm)
        