import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.group import Group



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bcond',default={})
        self.parameters.declare('connections',default={})
    def define(self):
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']
        connections = self.parameters['connections']


        # dummy mesh generation code:
        for beam_name in beams:
            num_beam_nodes = beams[beam_name]['n']
            # get the beam start/stop coordinates
            a = self.create_input(beam_name+'a',shape=(3),val=beams[beam_name]['a'])
            b = self.create_input(beam_name+'b',shape=(3),val=beams[beam_name]['b'])
            ds = (b - a)/(num_beam_nodes - 1)

            mesh = self.create_output(beam_name+'mesh',shape=(num_beam_nodes,3),val=0)
            for i in range(num_beam_nodes):
                node_i = a + ds*i
                mesh[i,:] = csdl.reshape(node_i, (1,3))
        


        # add a load:
        dummy_loads = np.zeros((5,3))
        dummy_loads[-1,1] = 1000000
        self.create_input('b4_forces',shape=(5,3),val=dummy_loads)

        # solve the beam group:
        self.add(Group(beams=beams,bcond=bcond,connections=connections), name='Group')






if __name__ == '__main__':

    #options, bcond, beams = {}, {}, {}
    beams, bcond, connections = {}, {}, {}

    name = 'b1'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5

    beams[name]['a'] = [0,0,0]
    beams[name]['b'] = [1,0,0]

    name = 'b2'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [0,0,0]
    beams[name]['b'] = [0.5,1,0]

    name = 'b3'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [0.5,1,0]
    beams[name]['b'] = [1,0,0]

    name = 'b4'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [0.5,1,0]
    beams[name]['b'] = [1.5,1,0]

    name = 'b5'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [1,0,0]
    beams[name]['b'] = [1.5,1,0]

    name = 'b6'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [1,0,0]
    beams[name]['b'] = [2,0,0]

    name = 'b7'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [1.5,1,0]
    beams[name]['b'] = [2,0,0]

    name = 'b8'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [1.5,1,0]
    beams[name]['b'] = [2.5,1,0]

    name = 'b9'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [2,0,0]
    beams[name]['b'] = [2.5,1,0]

    name = 'b10'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [2,0,0]
    beams[name]['b'] = [3,0,0]

    name = 'b11'
    beams[name] = {}
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 5
    beams[name]['a'] = [2.5,1,0]
    beams[name]['b'] = [3,0,0]



    # boundary conditions:
    name = 'fixed_left'
    bcond[name] = {}
    bcond[name]['beam'] = 'b1'
    bcond[name]['fpos'] = 'a'
    bcond[name]['fdim'] = [1,1,1,1,1,1] # x, y, z, phi, theta, psi: a 1 indicates the corresponding dof is fixed

    name = 'fixed_right'
    bcond[name] = {}
    bcond[name]['beam'] = 'b10'
    bcond[name]['fpos'] = 'b'
    bcond[name]['fdim'] = [1,1,1,1,1,1]


    # connections:
    name = 'c1'
    connections[name] = {}
    connections[name]['beam_names'] = ['b1','b2']
    connections[name]['nodes'] = ['a','a'] # connects the end of b1 to the start of b2

    name = 'c2'
    connections[name] = {}
    connections[name]['beam_names'] = ['b2','b3','b4']
    connections[name]['nodes'] = ['b','a','a']

    name = 'c3'
    connections[name] = {}
    connections[name]['beam_names'] = ['b1','b3','b5','b6']
    connections[name]['nodes'] = ['b','b','a','a']

    name = 'c4'
    connections[name] = {}
    connections[name]['beam_names'] = ['b4','b5','b7','b8']
    connections[name]['nodes'] = ['b','b','a','a']

    name = 'c5'
    connections[name] = {}
    connections[name]['beam_names'] = ['b6','b7','b9','b10']
    connections[name]['nodes'] = ['b','b','a','a']

    name = 'c6'
    connections[name] = {}
    connections[name]['beam_names'] = ['b8','b9','b11']
    connections[name]['nodes'] = ['b','b','a']

    name = 'c7'
    connections[name] = {}
    connections[name]['beam_names'] = ['b10','b11']
    connections[name]['nodes'] = ['b','b']





    sim = python_csdl_backend.Simulator(Run(beams=beams,bcond=bcond,connections=connections))
    sim.run()


    for beam_name in beams:
        num_beam_nodes = beams[beam_name]['n']
        num_elements = num_beam_nodes - 1

        for i in range(num_elements):
            element_name = beam_name + '_element_' + str(i)
            na = sim[element_name+'node_a_def']
            nb = sim[element_name+'node_b_def']

            x = np.array([na[0], nb[0]])
            y = np.array([na[1], nb[1]])
            z = np.array([na[2], nb[2]])

            plt.plot(x,y,color='k',zorder=7)
            plt.scatter(x,y,s=50,zorder=10,color='yellow',edgecolors='black',linewidth=1)


    # plot the cg:
    cg = sim['cg']
    cg_def = sim['cg_def']
    plt.scatter(cg[0],cg[1],color='blue',s=50,edgecolors='black')
    plt.scatter(cg_def[0],cg_def[1],color='red',s=40,edgecolors='black')


    plt.show()
    """
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

        #x = np.array([coord_a[0], coord_b[0]])
        #y = np.array([coord_a[1], coord_b[1]])
        #z = np.array([coord_a[2], coord_b[2]])

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
    """