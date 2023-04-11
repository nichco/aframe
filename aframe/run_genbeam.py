import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.group import Group



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options',default={})
        self.parameters.declare('bcond',default={})
    def define(self):
        options = self.parameters['options']
        bcond = self.parameters['bcond']

        # need to write code to autogenerate the options dict:
        """
        for beam_name in beams:
            nodes = beams[beam_name]['nodes']
            num_nodes = len(nodes)

            dummy_load = np.zeros((num_nodes,6))
            dummy_load[:,2] = 15 # z-force at every node

            #self.create_input(beam_name+'mesh',shape=(num_nodes,6),val=0)
            self.create_input(beam_name+'loads',shape=(num_nodes,6),val=dummy_load)
        """
        E = 69E9
        G = 26E9
        rho = 2700
        type = 'tube'

        num_nodes = 10
        dummy_mesh = np.zeros((num_nodes,6))
        dummy_mesh[:,0] = np.linspace(0,10,num_nodes)

        for i in range(num_nodes - 1):
            element_name = 'element_' + str(i)
            options[element_name] = {}
            options[element_name]['E'] = E
            options[element_name]['G'] = G
            options[element_name]['rho'] = rho
            options[element_name]['type'] = type
            options[element_name]['nodes'] = [i, i+1]

            na = dummy_mesh[i,:]
            nb = dummy_mesh[i+1,:]

            self.create_input(element_name+'node_a',shape=(6),val=na)
            self.create_input(element_name+'node_b',shape=(6),val=nb)

        
        # solve the beam group:
        self.add(Group(options=options,bcond=bcond), name='Group')
        
        






if __name__ == '__main__':

    options, bcond= {}, {}

    bcond['root1'] = {}
    bcond['root1']['node'] = 0
    bcond['root1']['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed




    sim = python_csdl_backend.Simulator(Run(options=options,bcond=bcond))
    sim.run()

    
    U = sim['U']
    vonmises_stress = sim['vonmises_stress']
    #print(vonmises_stress)


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