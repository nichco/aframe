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
    0 ------------------------------------- 1
      -                y                  -
      -                |                  -
      4                --> x              -
      -                                   -
      -                                   -
    3 ------------------------------------- 2
"""
class StressBox(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
    def define(self):
        name = self.parameters['name']

        A = self.declare_variable(name+'A')
        J = self.declare_variable(name+'J')
        Iy = self.declare_variable(name+'Iy') # height axis
        Iz = self.declare_variable(name+'Iz') # width axis

        w = self.declare_variable(name+'width')
        h = self.declare_variable(name+'height')

        # get the local loads:
        local_loads = self.declare_variable(name+'local_loads',shape=(12))

        # create the point coordinate matrix
        x_coord = self.create_output(name+'x_coord',shape=(5),val=0)
        y_coord = self.create_output(name+'y_coord',shape=(5),val=0)
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
        # point 5
        x_coord[4] = -w/2




        # compute the stress at each point:
        normal_force = local_loads[0]
        shear_force_w = local_loads[1]
        shear_force_h = local_loads[2]
        torque = local_loads[3]
        bend_moment_1 = local_loads[4] # height bending moment
        bend_moment_2 = local_loads[5] # width bending moment

        axial_stress = self.create_output(name+'bend_stress',shape=(5),val=0)
        torsional_stress = self.create_output(name+'torsional_stress',shape=(5),val=0)
        transverse_shear_stress = self.create_output(name+'shear_stress',shape=(5),val=0)
        s_vonmises = self.create_output(name+'s_vonmises',shape=(5),val=0)

        t_web = self.declare_variable(name+'t_web')
        Q = self.declare_variable(name+'Q')
        for point in range(5):
            x = x_coord[point]
            y = y_coord[point]
            r = ((x/2)**2 + (y/2)**2)**0.5

            axial_stress[point] = (normal_force/A) + (bend_moment_1*y/Iy) + (bend_moment_2*x/Iz)

            torsional_stress[point] = torque*r/J

            if point == 4: # the max shear at the neutral axis:
                transverse_shear_stress[point] = shear_force_h*Q/(Iy*t_web)

            tau = torsional_stress[point] + transverse_shear_stress[point]

            s_vonmises[point] = (axial_stress[point]**2 + 3*tau**2)**0.5


        #self.print_var(shear_force_h)
        #self.print_var(bend_moment_2)
        # take the maximum stress:
        s_max = csdl.max(s_vonmises)
        self.register_output(name+'s_vm', s_max)
        