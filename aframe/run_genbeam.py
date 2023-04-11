import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.group import Group



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options',default={})
        self.parameters.declare('bcond',default={})
        self.parameters.declare('beams',default={})
    def define(self):
        options = self.parameters['options']
        bcond = self.parameters['bcond']
        beams = self.parameters['beams']


        for beam_name in beams:
            nodes = beams[beam_name]['nodes']
            num_nodes = len(nodes)

            dummy_load = np.zeros((num_nodes,6))
            dummy_load[:,2] = 10 # z-force at every node

            #self.create_input(beam_name+'mesh',shape=(num_nodes,6),val=0)
            self.create_input(beam_name+'loads',shape=(num_nodes,6),val=dummy_load)


        
        # solve the beam group:
        self.add(Group(options=options,beams=beams,bcond=bcond), name='Group')
        
        






if __name__ == '__main__':

    options, bcond, beams = {}, {}, {}

    bcond['root1'] = {}
    bcond['root1']['node'] = 0
    bcond['root1']['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed

    #bcond['root2'] = {}
    #bcond['root2']['node'] = 10
    #bcond['root2']['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed


    name = 'beam_1'
    beams[name] = {}
    beams[name]['nodes'] = [0,1,2,3,4,5,6,7,8,9]
    beams[name]['type'] = 'tube'
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700

    b1mesh = np.zeros((len(beams[name]['nodes']),6))
    b1mesh[:,0] = np.linspace(0,10,len(beams[name]['nodes']))

    name = 'beam_2'
    beams[name] = {}
    beams[name]['nodes'] = [10,11,12,13,14,15,16,17,18,9]
    beams[name]['type'] = 'tube'
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700




    sim = python_csdl_backend.Simulator(Run(options=options,bcond=bcond,beams=beams))
    sim['beam_1mesh'] = b1mesh
    sim['beam_2mesh'] = b1mesh

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