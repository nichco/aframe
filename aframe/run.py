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

        # create the undeformed nodal inputs for each element:
        for element_name in options:
            self.create_input(element_name+'node_a',shape=(6),val=options[element_name]['node_a'])
            self.create_input(element_name+'node_b',shape=(6),val=options[element_name]['node_b'])

        # create the global loads vector
        loads = np.zeros((dim))
        loads[dim-4] = 100
        F = self.create_input('F',shape=(dim),val=loads)

        # solve the beam group:
        self.add(Group(options=options,bcond=bcond), name='Group')






if __name__ == '__main__':

    options = {}

    name = 'element_1'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700 # (kg/m^3)
    options[name]['nodes'] = [0,1] # node indices for [node_a, node_b]
    options[name]['node_a'] = [0,0,0,0,0,0] # node_a coordinates
    options[name]['node_b'] = [1,0,0,0,0,0] # node_b coordinates
    options[name]['type'] = 'box' # element type

    name = 'element_2'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [1,2]
    options[name]['node_a'] = [1,0,0,0,0,0]
    options[name]['node_b'] = [2,0,0,0,0,0]
    options[name]['type'] = 'box'

    name = 'element_3'
    options[name] = {}
    options[name]['E'] = 69E9
    options[name]['G'] = 26E9
    options[name]['rho'] = 2700
    options[name]['nodes'] = [2,3]
    options[name]['node_a'] = [2,0,0,0,0,0]
    options[name]['node_b'] = [3,0,0,0,0,0]
    options[name]['type'] = 'box'


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


    ax.set_xlim(0,3)
    ax.set_ylim(-1,1)
    ax.set_zlim(-0.4,0.1)
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