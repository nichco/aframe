import numpy as np
import csdl
import python_csdl_backend
from sectionpropertiestube import SectionPropertiesTube
from transform import Transform
from localstiffness import LocalStiffness



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']

        # process the options dictionary to compute the total number of unique nodes:
        node_list = [options[name]['node_a'] for name in options] + [options[name]['node_b'] for name in options]
        num_unique_nodes = len(set(node_list))

        # compute the section properties for each element:
        for element_name in options:
            if options[element_name]['type'] == 'tube': 
                self.add(SectionPropertiesTube(name=element_name), name=element_name+'SectionPropertiesTube')
            elif options[element_name]['type'] == 'box': 
                raise NotImplementedError('Error: type box for ' + element_name + ' is not implemented')


        # compute the local stiffness matrix for each element:
        for element_name in options:
            self.add(LocalStiffness(options=options[element_name],name=element_name), name=element_name+'LocalStiffness')


        # transform the local stiffness matrices to global coordinates:
        #for element_name in options:
        #    self.add(Transform())










if __name__ == '__main__':

    options = {}

    name = 'element_1'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 1E20
    options[name]['node_a'] = 0
    options[name]['node_b'] = 1
    options[name]['type'] = 'tube'





    sim = python_csdl_backend.Simulator(Run(options=options))
    sim.run()

