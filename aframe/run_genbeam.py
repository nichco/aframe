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

        # generate the beam mesh:
        beam_name = 'beam1'
        start = self.declare_variable(beam_name+'start',shape=(6),val=[0,0,0,0,0,0])
        stop = self.declare_variable(beam_name+'stop',shape=(6),val=[10,0,0,0,0,0])
        nodes = [0,1,2,3,4,5,6,7,8,9]
        num_nodes = len(nodes)
        
        for i in range(num_nodes - 1):
            element_name = 'element_' + str(i)
            options[element_name] = {}
            # constant material properties:
            options[element_name]['type'] = 'tube'
            options[element_name]['E'], options[element_name]['G'], options[element_name]['rho'] = 69E9, 26E9, 2700
            # define the elemental start node and stop node from the node list:
            options[element_name]['nodes'] = [nodes[i] , nodes[i+1]]
            # compute the elemental start and stop node coordinates:
            ds = (stop - start)/num_nodes
            node_a = start + ds*i
            node_b = start + ds*(i + 1)
            # register the outputs:
            self.register_output(element_name+'node_a', node_a)
            self.register_output(element_name+'node_b', node_b)


        # generate the loads vector:


        
        # pre-process the options dictionary to get dim:
        node_list = [options[name]['nodes'][0] for name in options] + [options[name]['nodes'][1] for name in options]
        node_list = [*set(node_list)]
        num_unique_nodes = len(node_list)
        dim = num_unique_nodes*6

        # create the global loads vector
        loads = np.zeros((dim))
        loads[dim-4] = -200
        F = self.create_input('F',shape=(dim),val=loads)

        # solve the beam group:
        self.add(Group(options=options,bcond=bcond), name='Group')
        






if __name__ == '__main__':

    options = {}

    bcond = {}

    name = 'root'
    bcond[name] = {}
    bcond[name]['node'] = 0
    bcond[name]['fdim'] = [1,1,1,1,1,1] # x, y, z, phi, theta, psi: a 1 indicates the corresponding dof is fixed





    sim = python_csdl_backend.Simulator(Run(options=options,bcond=bcond))
    sim.run()

    
    U = sim['U']
    vonmises_stress = sim['vonmises_stress']
    print(vonmises_stress)


    coord = sim['coord']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, element_name in enumerate(options):
        coord_a = coord[i,0,:]
        coord_b = coord[i,1,:]

        x = np.array([coord_a[0], coord_b[0]])
        y = np.array([coord_a[1], coord_b[1]])
        z = np.array([coord_a[2], coord_b[2]])

        ax.plot(x,y,z,color='k')
        ax.scatter(coord_a[0], coord_a[1], coord_a[2],color='yellow',edgecolors='black',linewidth=1)
        ax.scatter(coord_b[0], coord_b[1], coord_b[2],color='yellow',edgecolors='black',linewidth=1)


    ax.set_xlim(0,10)
    ax.set_ylim(-1,1)
    ax.set_zlim(-0.4,0.1)

    # plot the cg:
    cg = sim['cg']
    cg_def = sim['cg_def']
    ax.scatter(cg[0],cg[1],cg[2],color='blue',s=50,edgecolors='black')
    ax.scatter(cg_def[0],cg_def[1],cg_def[2],color='red',s=50,edgecolors='black')
    #ax.text(cg[0], cg[1], cg[2]-0.05, s='CG', fontsize=10)


    plt.show()
    

    """
    # validation:
    F = -1000
    L = 3
    E = 69E9
    I = sim['element_1Iy']
    dmax = F*(L**3)/(3*E*I)
    print('dmax: ',dmax)
    """