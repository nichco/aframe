import numpy as np
import csdl
import python_csdl_backend




class GlobalLoads(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('beams')
        self.parameters.declare('bcond')
        self.parameters.declare('node_id')
        self.parameters.declare('num_unique_nodes')
    def define(self):
        options = self.parameters['options']
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']
        node_id = self.parameters['node_id']
        num_unique_nodes = self.parameters['num_unique_nodes']
        num_beams = len(beams)

        # create a list of bc nodes:
        bc_list = [bcond[bc_name]['node'] for bc_name in bcond]



        # create the nodal loads vector:
        nodal_loads = self.create_output('nodal_loads',shape=(num_beams,num_unique_nodes,6),val=0)

        for i, beam_name in enumerate(beams):
            nodes = beams[beam_name]['nodes']
            num_nodes = len(nodes)
            loads = self.declare_variable(beam_name+'loads',shape=(num_nodes,6),val=0)

            for j, node in enumerate(nodes):
                id = node_id[node]

                # if the node is not a boundary condition, add the corresponding load:
                if node not in bc_list:
                    nodal_loads[i,id,:] = csdl.reshape(loads[j,:], (1,1,6))






        # sum the nodal loads over the beams (so that loads can be doubly defined where beams join):
        total_loads = csdl.sum(nodal_loads, axes=(0,))

        # flatten the total loads matrix to a vector:
        F = csdl.reshape(total_loads, new_shape=(6*num_unique_nodes))
        self.register_output('F', F)

        #self.print_var(F)




if __name__ == '__main__':

    pass