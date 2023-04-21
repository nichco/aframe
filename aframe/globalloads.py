import numpy as np
import csdl



class GlobalLoads(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('num_unique_nodes')
        self.parameters.declare('nodes')
        self.parameters.declare('node_index')
        self.parameters.declare('bounds')
        self.parameters.declare('load_factor')


    def define(self):
        beams = self.parameters['beams']
        num_unique_nodes = self.parameters['num_unique_nodes']
        nodes = self.parameters['nodes']
        node_index = self.parameters['node_index']
        bounds = self.parameters['bounds']
        load_factor = self.parameters['load_factor']


        b_index_list = []
        b_node_list = []
        for b_name in bounds:
            beam_name = bounds[b_name]['beam']
            fpos = bounds[b_name]['fpos']
            fdim = bounds[b_name]['fdim']
            if fpos == 'a': b_node = nodes[beam_name][0]
            elif fpos == 'b': b_node = nodes[beam_name][-1]

            b_node_index = node_index[b_node]
            b_node_list.append(b_node)

            # add the constrained dof index to the b_index_list:
            for i, fdim in enumerate(fdim):
                if fdim == 1: b_index_list.append(b_node_index*6 + i)




        nodal_loads = self.create_output('nodal_loads',shape=(len(beams),num_unique_nodes,6),val=0)
        for i, beam_name in enumerate(beams):
            n = beams[beam_name]['n']
            beam_nodes = nodes[beam_name]
            
            forces = self.declare_variable(beam_name+'_forces',shape=(n,3),val=0)
            #moments = self.declare_variable(beam_name+'_moments',shape=(n,3),val=0)

            # concatenate the forces and moments:
            loads = self.create_output(f'{beam_name}_loads',shape=(n,6),val=0)
            loads[:,0:3] = forces*load_factor
            #loads[:, 3:6] = moments*load_factor

          
            for j, bnode in enumerate(beam_nodes):
                index = node_index[bnode]

                #if index not in b_index_list: 
                if bnode not in b_node_list: 
                    nodal_loads[i,index,:] = csdl.reshape(loads[j,:], (1,1,6))

                for k in range(6):
                    pass
            
                    

        #self.print_var(nodal_loads)

        # sum the nodal loads over the beams (so that loads can be doubly defined where beams join):
        total_loads = csdl.sum(nodal_loads, axes=(0,))

        #self.print_var(total_loads)

        # flatten the total loads matrix to a vector:
        Fi = csdl.reshape(total_loads, new_shape=(6*num_unique_nodes))
        self.register_output('Fi', Fi)

        #self.print_var(Fi)