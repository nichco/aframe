import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.group import Group



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('bcond')
    def define(self):
        options = self.parameters['options']
        bcond = self.parameters['bcond']

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

        # solve the beam group:
        self.add(Group(options=options,bcond=bcond), name='Group')






if __name__ == '__main__':

    options = {}

    name = 'element_1'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [0,1] # node indices for [node_a, node_b]
    options[name]['node_a'] = [0,0,0,0,0,0] # node_a coordinates
    options[name]['node_b'] = [0.5,1,0,0,0,0] # node_b coordinates
    options[name]['type'] = 'tube' # element type

    name = 'element_2'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [0,2]
    options[name]['node_a'] = [0,0,0,0,0,0]
    options[name]['node_b'] = [1,0,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_3'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [1,2]
    options[name]['node_a'] = [0.5,1,0,0,0,0]
    options[name]['node_b'] = [1,0,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_4'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [1,3]
    options[name]['node_a'] = [0.5,1,0,0,0,0]
    options[name]['node_b'] = [1.5,1,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_5'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [2,3]
    options[name]['node_a'] = [1,0,0,0,0,0]
    options[name]['node_b'] = [1.5,1,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_6'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [2,4]
    options[name]['node_a'] = [1,0,0,0,0,0]
    options[name]['node_b'] = [2,0,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_7'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [3,4]
    options[name]['node_a'] = [1.5,1,0,0,0,0]
    options[name]['node_b'] = [2,0,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_8'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [3,5]
    options[name]['node_a'] = [1.5,1,0,0,0,0]
    options[name]['node_b'] = [2.5,1,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_9'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [4,5]
    options[name]['node_a'] = [2,0,0,0,0,0]
    options[name]['node_b'] = [2.5,1,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_10'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [4,6]
    options[name]['node_a'] = [2,0,0,0,0,0]
    options[name]['node_b'] = [3,0,0,0,0,0]
    options[name]['type'] = 'tube'

    name = 'element_11'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [5,6]
    options[name]['node_a'] = [2.5,1,0,0,0,0]
    options[name]['node_b'] = [3,0,0,0,0,0]
    options[name]['type'] = 'tube'


    bcond = {}

    name = 'fixed_left'
    bcond[name] = {}
    bcond[name]['node'] = 0
    bcond[name]['fdim'] = [1,1,1,1,1,1] # x, y, z, phi, theta, psi: a 1 indicates the corresponding dof is fixed

    name = 'fixed_right'
    bcond[name] = {}
    bcond[name]['node'] = 6
    bcond[name]['fdim'] = [1,1,1,1,1,1]





    sim = python_csdl_backend.Simulator(Run(options=options,bcond=bcond))
    sim.run()

    # get the deformed nodal coordinates:
    coord = sim['coord']

    plt.rcParams["figure.figsize"] = (8,3)

    # plot the undeformed bridge:
    for element_name in options:
        a = options[element_name]['node_a']
        b = options[element_name]['node_b']

        xu = np.array([a[0],b[0]])
        yu = np.array([a[1],b[1]])
        zu = np.array([a[2],b[2]])

        plt.plot(xu,yu,color='silver',linewidth=4)
        plt.scatter(xu,yu,s=50,color='silver',edgecolors='black',linewidth=0.5,zorder=5)

    # plot the deformed bridge:
    for i, element_name in enumerate(options):
        coord_a = coord[i,0,:]
        coord_b = coord[i,1,:]

        x = np.array([coord_a[0], coord_b[0]])
        y = np.array([coord_a[1], coord_b[1]])
        z = np.array([coord_a[2], coord_b[2]])

        plt.plot(x,y,color='k',zorder=7)
        plt.scatter(x,y,s=50,zorder=10,color='yellow',edgecolors='black',linewidth=1)

    # plot the applied force arrow:
    plt.arrow(1.5,1,0,0.2,width=0.04,color='red')

    # plot the cg:
    cg = sim['cg']
    cg_def = sim['cg_def']
    plt.scatter(cg[0],cg[1],color='blue',s=50,edgecolors='black')
    plt.scatter(cg_def[0],cg_def[1],color='red',s=40,edgecolors='black')


    plt.xlim(-0.1,3.1)
    plt.show()