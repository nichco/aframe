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
            #beam_nodes = beams[beam_name]['nodes']
            #num_beam_nodes = len(beam_nodes)
            num_beam_nodes = beams[beam_name]['n']
            # get the beam start/stop coordinates
            a = self.create_input(beam_name+'a',shape=(3),val=beams[beam_name]['a'])
            b = self.create_input(beam_name+'b',shape=(3),val=beams[beam_name]['b'])

            ds = (b - a)/(num_beam_nodes - 1)

            mesh = self.create_output(beam_name+'mesh',shape=(num_beam_nodes,3),val=0)
            for i in range(num_beam_nodes):
                node_i = a + ds*i
                mesh[i,:] = csdl.reshape(node_i, (1,3))
        


        dummy_loads = np.zeros((10,3))
        dummy_loads[-1,2] = 100
        self.create_input('b1_forces',shape=(10,3),val=dummy_loads)




        
        # solve the beam group:
        self.add(Group(beams=beams,bcond=bcond,connections=connections), name='Group')
        
        






if __name__ == '__main__':

    bcond, beams = {}, {}

    name = 'b1'
    beams[name] = {}
    #beams[name]['nodes'] = [0,1,2,3,4,5,6,7,8,9]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 10
    
    beams[name]['a'] = [0,0,0]
    beams[name]['b'] = [10,0,0]

    
    name = 'b2'
    beams[name] = {}
    #beams[name]['nodes'] = [9,10,11,12]
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700
    beams[name]['type'] = 'tube'
    beams[name]['n'] = 4

    beams[name]['a'] = [10,0,0]
    beams[name]['b'] = [10,1,0]
    

    name = 'root'
    bcond[name] = {}
    bcond[name]['beam'] = 'b1'
    bcond[name]['fpos'] = 'a'
    bcond[name]['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed


    connections = {}
    
    name = 'c1'
    connections[name] = {}
    connections[name]['beam_names'] = ['b1','b2']
    connections[name]['nodes'] = ['b','a'] # connects the end of b1 to the start of b2




    sim = python_csdl_backend.Simulator(Run(beams=beams,bcond=bcond,connections=connections))
    sim.run()

    
    U = sim['U']
    vonmises_stress = sim['vonmises_stress']
    #print(vonmises_stress)


    coord = sim['coord']
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    for beam_name in beams:
        #beam_nodes = beams[beam_name]['nodes']
        #num_beam_nodes = len(beam_nodes)
        num_beam_nodes = beams[beam_name]['n']
        num_elements = num_beam_nodes - 1

        for i in range(num_elements):
            element_name = beam_name + '_element_' + str(i)
            na = sim[element_name+'node_a_def']
            nb = sim[element_name+'node_b_def']

            x = np.array([na[0], nb[0]])
            y = np.array([na[1], nb[1]])
            z = np.array([na[2], nb[2]])

            ax.plot(x,y,z,color='k')
            ax.scatter(na[0], na[1], na[2],color='yellow',edgecolors='black',linewidth=1)
            ax.scatter(nb[0], nb[1], nb[2],color='yellow',edgecolors='black',linewidth=1)

    

    
    # plot the cg:
    cg = sim['cg']
    cg_def = sim['cg_def']
    ax.scatter(cg[0],cg[1],cg[2],color='blue',s=50,edgecolors='black')
    ax.scatter(cg_def[0],cg_def[1],cg_def[2],color='red',s=50,edgecolors='black')


    ax.set_xlim(0,10)
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