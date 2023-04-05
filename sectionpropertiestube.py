import numpy as np
import csdl
import python_csdl_backend




class SectionPropertiesTube(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')

    def define(self):
        options = self.parameters['options']
        n = options['n']

        radius = self.declare_variable('radius',shape=(n-1))
        thickness = self.declare_variable('thickness',shape=(n-1))

        # Add thickness to the interior of the radius.
        # The outer radius is the inputs['radius'] amount.
        r1 = radius - thickness
        r2 = radius

        # Compute the area, area moments of inertia, and polar moment of inertia
        A = np.pi * (r2**2 - r1**2)
        Iy = np.pi * (r2**4 - r1**4) / 4.0
        Iz = np.pi * (r2**4 - r1**4) / 4.0
        J = np.pi * (r2**4 - r1**4) / 2.0

        self.register_output('A', A)
        self.register_output('Iy', Iy)
        self.register_output('Iz', Iz)
        self.register_output('J', J)