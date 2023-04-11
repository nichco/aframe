import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.group import Group



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('beams')
        self.parameters.declare('bcond')
    def define(self):
        options = self.parameters['options']
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']


        # dummy mesh generation code:
        for beam_name in beams:
            beam_nodes = beams[beam_name]['nodes']
            num_beam_nodes = len(beam_nodes)
            # get the beam start/stop coordinates
            a = self.create_input(beam_name+'a',shape=(6),val=beams[beam_name]['a'])
            b = self.create_input(beam_name+'b',shape=(6),val=beams[beam_name]['b'])
            ds = (b - a)/(num_beam_nodes - 1)
            mesh = self.create_output(beam_name+'mesh',shape=(num_beam_nodes,6),val=0)
            for i in range(num_beam_nodes):
                node_i = a + ds*i
                mesh[i,:] = csdl.reshape(node_i, (1,6))
        


        # NOTE: beam_nodes, mesh, and loads must be linearly correlated
        for beam_name in beams:
            beam_nodes = beams[beam_name]['nodes']
            num_beam_nodes = len(beam_nodes)
            num_elements = num_beam_nodes - 1
            E, G, rho, type = beams[beam_name]['E'], beams[beam_name]['G'], beams[beam_name]['rho'], beams[beam_name]['type']

            #dummy_mesh = np.zeros((num_beam_nodes,6))
            #dummy_mesh[:,0] = np.linspace(0,10,num_beam_nodes)
            dummy_mesh = self.declare_variable(beam_name+'mesh',shape=(num_beam_nodes,6))

            # create an options dictionary entry for each element:
            for i in range(num_elements):
                element_name = beam_name + '_element_' + str(i)
                options[element_name] = {}
                options[element_name]['E'] = E
                options[element_name]['G'] = G
                options[element_name]['rho'] = rho
                options[element_name]['type'] = type
                options[element_name]['nodes'] = [beam_nodes[i], beam_nodes[i+1]]

                na = csdl.reshape(dummy_mesh[i,:], (6))
                nb = csdl.reshape(dummy_mesh[i+1,:], (6))

                #self.create_input(element_name+'node_a',shape=(6),val=na)
                #self.create_input(element_name+'node_b',shape=(6),val=nb)
                self.register_output(element_name+'node_a',na)
                self.register_output(element_name+'node_b',nb)

        """
        # pre-process the options dictionary to get dim:
        node_list = [options[name]['nodes'][0] for name in options] + [options[name]['nodes'][1] for name in options]
        node_list = [*set(node_list)]
        num_unique_nodes = len(node_list)
        dim = num_unique_nodes*6
        node_id = {node_list[i]: i for i in range(num_unique_nodes)}

        # create the undeformed nodal inputs for each element:
        for element_name in options:
            self.create_input(element_name+'node_a',shape=(6),val=options[element_name]['node_a'])
            self.create_input(element_name+'node_b',shape=(6),val=options[element_name]['node_b'])

        # create the global loads vector
        loads = np.zeros((dim))
        f_id = node_id[3] # apply a force at node 3
        loads[f_id*6 + 1] = 400000

        self.create_input('F',shape=(dim),val=loads)
        """


        dummy_loads = np.zeros((3,6))
        dummy_loads[-1,1] = 1000000
        self.create_input('b4loads',shape=(3,6),val=dummy_loads)

        # solve the beam group:
        self.add(Group(options=options,beams=beams,bcond=bcond), name='Group')






if __name__ == '__main__':

    options, bcond, beams = {}, {}, {}

    name = 'b1'
    beams[name] = {}
    beams[name]['nodes'] = [0,1,2]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [0,0,0,0,0,0]
    beams[name]['b'] = [1,0,0,0,0,0]

    name = 'b2'
    beams[name] = {}
    beams[name]['nodes'] = [0,3,4]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [0,0,0,0,0,0]
    beams[name]['b'] = [0.5,1,0,0,0,0]

    name = 'b3'
    beams[name] = {}
    beams[name]['nodes'] = [4,5,2]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [0.5,1,0,0,0,0]
    beams[name]['b'] = [1,0,0,0,0,0]

    name = 'b4'
    beams[name] = {}
    beams[name]['nodes'] = [4,6,8]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [0.5,1,0,0,0,0]
    beams[name]['b'] = [1.5,1,0,0,0,0]

    name = 'b5'
    beams[name] = {}
    beams[name]['nodes'] = [2,7,8]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [1,0,0,0,0,0]
    beams[name]['b'] = [1.5,1,0,0,0,0]

    name = 'b6'
    beams[name] = {}
    beams[name]['nodes'] = [2,9,11]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [1,0,0,0,0,0]
    beams[name]['b'] = [2,0,0,0,0,0]

    name = 'b7'
    beams[name] = {}
    beams[name]['nodes'] = [8,10,11]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [1.5,1,0,0,0,0]
    beams[name]['b'] = [2,0,0,0,0,0]

    name = 'b8'
    beams[name] = {}
    beams[name]['nodes'] = [8,12,14]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [1.5,1,0,0,0,0]
    beams[name]['b'] = [2.5,1,0,0,0,0]

    name = 'b9'
    beams[name] = {}
    beams[name]['nodes'] = [11,13,14]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [2,0,0,0,0,0]
    beams[name]['b'] = [2.5,1,0,0,0,0]

    name = 'b10'
    beams[name] = {}
    beams[name]['nodes'] = [11,15,17]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [2,0,0,0,0,0]
    beams[name]['b'] = [3,0,0,0,0,0]

    name = 'b11'
    beams[name] = {}
    beams[name]['nodes'] = [14,16,17]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [2.5,1,0,0,0,0]
    beams[name]['b'] = [3,0,0,0,0,0]


    bcond = {}

    name = 'fixed_left'
    bcond[name] = {}
    bcond[name]['node'] = 0
    bcond[name]['fdim'] = [1,1,1,1,1,1] # x, y, z, phi, theta, psi: a 1 indicates the corresponding dof is fixed

    name = 'fixed_right'
    bcond[name] = {}
    bcond[name]['node'] = 17
    bcond[name]['fdim'] = [1,1,1,1,1,1]





    sim = python_csdl_backend.Simulator(Run(options=options,beams=beams,bcond=bcond))
    sim.run()

    # get the deformed nodal coordinates:
    coord = sim['coord']

    plt.rcParams["figure.figsize"] = (8,3)

    
    # plot the undeformed bridge:
    for element_name in options:
        a = sim[element_name+'node_a']
        b = sim[element_name+'node_b']

        xu = np.array([a[0],b[0]])
        yu = np.array([a[1],b[1]])
        zu = np.array([a[2],b[2]])

        plt.plot(xu,yu,color='silver',linewidth=4)
        plt.scatter(xu,yu,s=50,color='silver',edgecolors='black',linewidth=0.5,zorder=5)
    
    # plot the deformed bridge:
    for i, element_name in enumerate(options):
        a = sim[element_name+'node_a']
        b = sim[element_name+'node_b']
        coord_a = coord[i,0,:]
        coord_b = coord[i,1,:]

        da = a - coord_a
        db = b - coord_b

        mag = 3

        na = a - da*mag
        nb = b - db*mag


        nx = np.array([na[0], nb[0]])
        ny = np.array([na[1], nb[1]])
        nz = np.array([na[2], nb[2]])

        

        x = np.array([coord_a[0], coord_b[0]])
        y = np.array([coord_a[1], coord_b[1]])
        z = np.array([coord_a[2], coord_b[2]])

        plt.plot(nx,ny,color='k',zorder=7)
        plt.scatter(nx,ny,s=50,zorder=10,color='yellow',edgecolors='black',linewidth=1)

    # plot the applied force arrow:
    plt.arrow(1.5,1,0,0.2,width=0.04,color='red')

    # plot the cg:
    cg = sim['cg']
    cg_def = sim['cg_def']
    plt.scatter(cg[0],cg[1],color='blue',s=50,edgecolors='black')
    plt.scatter(cg_def[0],cg_def[1],color='red',s=40,edgecolors='black')


    plt.xlim(-0.1,3.1)
    plt.show()