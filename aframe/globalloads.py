import numpy as np
import csdl
import python_csdl_backend




class GlobalLoads(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('nodes')
        self.parameters.declare('bcond')
        self.parameters.declare('node_id')
        self.parameters.declare('num_unique_nodes')
        self.parameters.declare('bc_node_list')
    def define(self):
        beams = self.parameters['beams']
        nodes = self.parameters['nodes']
        bcond = self.parameters['bcond']
        node_id = self.parameters['node_id']
        num_unique_nodes = self.parameters['num_unique_nodes']
        bc_list = self.parameters['bc_node_list']
        num_beams = len(beams)


        # create the nodal loads vector:
        # if any beams exist in the beams dictionary:
        nodal_loads = self.create_output('nodal_loads',shape=(num_beams,num_unique_nodes,6),val=0)
        # iterate over every beam:
        for i, beam_name in enumerate(beams):
            #beam_nodes = beams[beam_name]['nodes']
            beam_nodes = nodes[beam_name]['nodes']
            num_beam_nodes = len(beam_nodes)

            # declare the beam loads (default is zero):
            forces = self.declare_variable(beam_name+'_forces',shape=(num_beam_nodes, 3),val=0)
            moments = self.declare_variable(beam_name+'_moments',shape=(num_beam_nodes, 3),val=0)
            # self.print_var(loads)
            loads =  self.create_output(f'{beam_name}_loads', shape=(num_beam_nodes, 6), val=0)
            loads[:,0:3] = forces
            loads[:, 3:6] = moments
            # iterate over the beam nodes:
            for j, node in enumerate(beam_nodes):
                beam_node_id = node_id[node]

                # if the node is not a boundary condition, add the corresponding load:
                if node not in bc_list:
                    nodal_loads[i,beam_node_id,:] = csdl.reshape(loads[j,:], (1,1,6))





        # sum the nodal loads over the beams (so that loads can be doubly defined where beams join):
        total_loads = csdl.sum(nodal_loads, axes=(0,))

        # flatten the total loads matrix to a vector:
        F = csdl.reshape(total_loads, new_shape=(6*num_unique_nodes))
        self.register_output('F', F)

        # self.print_var(F)

