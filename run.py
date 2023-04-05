import numpy as np
import csdl
import python_csdl_backend
from sectionpropertiestube import SectionPropertiesTube
from transform import Transform
from localstiffness import LocalStiffness
from createrhs import CreateRHS



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']

        self.add(SectionPropertiesTube(options=options), name='SectionPropertiesTube')
        self.add(LocalStiffness(options=options), name='LocalStiff')
        self.add(Transform(options=options), name='Transform')
        self.add(CreateRHS(options=options), name='CreateRHS')

        k = self.declare_variable('k',shape=(12*n,12*n))
        f = self.declare_variable('f',shape=(6*n))


        u = self.declare_variable('u',shape=(6,n),val=0)
        residual = csdl.matmat(k, u) - f








if __name__ == '__main__':

    options = {}
    options['E'] = 69E9
    options['G'] = 1E20
    options['n'] = 2





    sim = python_csdl_backend.Simulator(Run(options=options))
    sim['radius'] = np.ones((options['n']-1))*2
    sim['thickness'] = np.ones((options['n']-1))*0.1


    nodes = np.zeros((3,options['n']))
    nodes[0,1] = 2
    sim['nodes'] = nodes


    sim.run()

