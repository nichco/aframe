import numpy as np
import csdl
import python_csdl_backend




class SectionPropertiesTube(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')

    def define(self):
        name = self.parameters['name']

        radius = self.declare_variable(name+'radius',val=0.5)
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