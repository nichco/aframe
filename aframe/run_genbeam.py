import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.group import Group



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options',default={})
        self.parameters.declare('beams',default={})
        self.parameters.declare('bcond',default={})
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
        


        dummy_loads = np.zeros((10,6))
        dummy_loads[-1,2] = 100
        self.create_input('b1loads',shape=(10,6),val=dummy_loads)




        
        # solve the beam group:
        self.add(Group(options=options,beams=beams,bcond=bcond), name='Group')
        
        






if __name__ == '__main__':

    options, bcond, beams = {}, {}, {}

    name = 'b1'
    beams[name] = {}
    beams[name]['nodes'] = [0,1,2,3,4,5,6,7,8,9]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    
    beams[name]['a'] = [0,0,0,0,0,0]
    beams[name]['b'] = [10,0,0,0,0,0]

    
    name = 'b2'
    beams[name] = {}
    beams[name]['nodes'] = [9,10,11,12]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'

    beams[name]['a'] = [10,0,0,0,0,0]
    beams[name]['b'] = [10,1,0,0,0,0]
    

    bcond['root1'] = {}
    bcond['root1']['node'] = 0
    bcond['root1']['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed




    sim = python_csdl_backend.Simulator(Run(options=options,beams=beams,bcond=bcond))
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