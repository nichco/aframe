import numpy as np
import csdl
import python_csdl_backend




class SectionPropertiesTube(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')

    def define(self):
        name = self.parameters['name']

        radius = self.declare_variable(name+'radius',val=0.125)
        thickness = self.declare_variable(name+'thickness',val=0.001)

        r1 = radius - thickness # IR
        r2 = radius # OR

        # Compute the area, area moments of inertia, and polar moment of inertia
        A = np.pi * (r2**2 - r1**2)
        Iy = np.pi * (r2**4 - r1**4) / 4.0
        Iz = np.pi * (r2**4 - r1**4) / 4.0
        J = np.pi * (r2**4 - r1**4) / 2.0

        self.register_output(name+'A', A)
        self.register_output(name+'Iy', Iy)
        self.register_output(name+'Iz', Iz)
        self.register_output(name+'J', J)




class SectionPropertiesRect(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')

    def define(self):
        name = self.parameters['name']

        width = self.declare_variable(name+'width',val=0.1)
        height = self.declare_variable(name+'height',val=0.1)



        # Compute the area, area moments of inertia, and polar moment of inertia
        A = width*height
        Iz = (width**3)*height/12
        Iy = width*(height**3)/12
        J = width*height*(height**2 + width**2)/12

        self.register_output(name+'A', A)
        self.register_output(name+'Iy', Iy)
        self.register_output(name+'Iz', Iz)
        self.register_output(name+'J', J)





class SectionPropertiesBox(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')

    def define(self):
        name = self.parameters['name']

        width = self.declare_variable(name+'width',val=0.1)
        height = self.declare_variable(name+'height',val=0.25)
        t_web = self.declare_variable(name+'t_web',val=0.001)
        t_cap = self.declare_variable(name+'t_cap',val=0.001)

        width_i = width - 2*t_web
        height_i = height - 2*t_cap

        # Compute the area, area moments of inertia, and polar moment of inertia
        A = width*height - width_i*height_i
        Iz = ((width**3)*height - (width_i**3)*height_i)/12
        Iy = (width*(height**3) - width_i*(height_i**3))/12
        J = (width*height*(height**2 + width**2)/12) - (width_i*height_i*(height_i**2 + width_i**2)/12)


        self.register_output(name+'A', A)
        self.register_output(name+'Iy', Iy)
        self.register_output(name+'Iz', Iz)
        self.register_output(name+'J', J)