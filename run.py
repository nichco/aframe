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
        num_unique_nodes = len(set(tuple(sub_list) for sub_list in node_list))


        # create nodal inputs for each element:
        for element_name in options:
            self.create_input(element_name+'node_a',shape=(3),val=options[element_name]['node_a'])
            self.create_input(element_name+'node_b',shape=(3),val=options[element_name]['node_b'])


        # compute the section properties for each element:
        for element_name in options:
            if options[element_name]['type'] == 'tube': 
                self.add(SectionPropertiesTube(name=element_name), name=element_name+'SectionPropertiesTube')
            elif options[element_name]['type'] == 'box': 
                raise NotImplementedError('Error: type box for ' + element_name + ' is not implemented')
            else: raise NotImplementedError('Error: type for' + element_name + 'is not implemented')


        # compute the local stiffness matrix for each element:
        for element_name in options:
            self.add(LocalStiffness(options=options[element_name],name=element_name), name=element_name+'LocalStiffness')


        # transform the local stiffness matrices to global coordinates:
        for element_name in options:
            self.add(Transform(options=options[element_name],name=element_name), name=element_name+'Transform')


        # construct the global stiffness matrix:







if __name__ == '__main__':

    options = {}

    name = 'element_1'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 1E20
    options[name]['node_a'] = [0,0,0]
    options[name]['node_b'] = [1,1,0]
    options[name]['type'] = 'tube'





    sim = python_csdl_backend.Simulator(Run(options=options))
    sim.run()

