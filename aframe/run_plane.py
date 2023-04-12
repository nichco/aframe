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
                mesh[i,:] = csdl.reshape(a + ds*i, (1,6))



        dummy_loads = np.zeros((21,6))
        dummy_loads[:,2] = 10
        self.create_input('wingloads',shape=(21,6),val=dummy_loads)


        # solve the beam group:
        self.add(Group(options=options,beams=beams,bcond=bcond), name='Group')







if __name__ == '__main__':

    options, bcond, beams = {}, {}, {}

    name = 'wing'
    beams[name] = {}
    beams[name]['nodes'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [-38,14,0,0,0,0]
    beams[name]['b'] = [38,14,0,0,0,0]

    name = 'lfuse'
    beams[name] = {}
    beams[name]['nodes'] = [21,22,23,7,24,25,26,27]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [-7,0,0,0,0,0]
    beams[name]['b'] = [-7,28,0,0,0,0]

    name = 'rfuse'
    beams[name] = {}
    beams[name]['nodes'] = [28,29,30,13,31,32,33,34]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['a'] = [7,0,0,0,0,0]
    beams[name]['b'] = [7,28,0,0,0,0]


    bcond['root1'] = {}
    bcond['root1']['node'] = 10
    bcond['root1']['fdim'] = [1,1,1,1,1,1]







    sim = python_csdl_backend.Simulator(Run(options=options,beams=beams,bcond=bcond))
    sim.run()
    U = sim['U']



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


    ax.set_xlim(-40,40)
    ax.set_ylim(-40,40)
    ax.set_zlim(-5,5)

    # plot the cg:
    cg = sim['cg']
    cg_def = sim['cg_def']
    ax.scatter(cg[0],cg[1],cg[2],color='blue',s=50,edgecolors='black')
    ax.scatter(cg_def[0],cg_def[1],cg_def[2],color='red',s=50,edgecolors='black')


    plt.show()