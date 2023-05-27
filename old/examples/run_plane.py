import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.beamgroup import BeamGroup




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

            mesh = self.create_output(beam_name,shape=(num_beam_nodes,3),val=0)
            for i in range(num_beam_nodes):
                mesh[i,:] = csdl.reshape(a + ds*i, (1,3))



        w_loads = np.zeros((10,3))
        w_loads[:,2] = 8
        self.create_input('lwing_forces',shape=(10,3),val=w_loads)
        self.create_input('rwing_forces',shape=(10,3),val=w_loads)

        f_loads = np.zeros((4,3))
        f_loads[:,2] = -10
        self.create_input('flfuse_forces',shape=(4,3),val=f_loads)
        self.create_input('frfuse_forces',shape=(4,3),val=f_loads)

        t_loads = np.zeros((5,3))
        t_loads[:,2] = -15
        self.create_input('htail_forces',shape=(5,3),val=t_loads)

        c_loads = np.zeros((6,3))
        c_loads[:,2] = 15
        self.create_input('cwing_forces',shape=(6,3),val=c_loads)


        # solve the beam group:
        self.add(BeamGroup(beams=beams,bcond=bcond,connections=connections), name='BeamGroup')







if __name__ == '__main__':

    bcond, beams, connections = {}, {}, {}

    beams['lwing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 10,'a': [-38,14,0],'b': [-7,14,0]}
    beams['cwing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 6,'a': [-7,14,0],'b': [7,14,0]}
    beams['rwing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 10,'a': [7,14,0],'b': [38,14,0]}
    beams['flfuse'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 4,'a': [-7,0,0],'b': [-7,14,0]}
    beams['blfuse'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 4,'a': [-7,14,0],'b': [-7,28,0]}
    beams['frfuse'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 4,'a': [7,0,0],'b': [7,14,0]}
    beams['brfuse'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 4,'a': [7,14,0],'b': [7,28,0]}
    beams['lvtail'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 3,'a': [-7,28,0],'b': [-6,28,0.5]}
    beams['rvtail'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 3,'a': [7,28,0],'b': [6,28,0.5]}
    beams['htail'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [-6,28,0.5],'b': [6,28,0.5]}



    name = 'root_1'
    bcond[name] = {}
    bcond[name]['beam'] = 'lwing'
    bcond[name]['fpos'] = 'b'
    bcond[name]['fdim'] = [1,1,1,1,1,1]

    name = 'root_2'
    bcond[name] = {}
    bcond[name]['beam'] = 'rwing'
    bcond[name]['fpos'] = 'a'
    bcond[name]['fdim'] = [1,1,1,1,1,1]

    name = 'c1'
    connections[name] = {}
    connections[name]['beam_names'] = ['lwing','cwing','flfuse','blfuse']
    connections[name]['nodes'] = ['b','a','b','a']

    name = 'c2'
    connections[name] = {}
    connections[name]['beam_names'] = ['cwing','rwing','frfuse','brfuse']
    connections[name]['nodes'] = ['b','a','b','a']

    name = 'c3'
    connections[name] = {}
    connections[name]['beam_names'] = ['blfuse','lvtail']
    connections[name]['nodes'] = ['b','a']

    name = 'c4'
    connections[name] = {}
    connections[name]['beam_names'] = ['brfuse','rvtail']
    connections[name]['nodes'] = ['b','a']

    name = 'c5'
    connections[name] = {}
    connections[name]['beam_names'] = ['lvtail','htail']
    connections[name]['nodes'] = ['b','a']

    name = 'c6'
    connections[name] = {}
    connections[name]['beam_names'] = ['rvtail','htail']
    connections[name]['nodes'] = ['b','b']







    sim = python_csdl_backend.Simulator(Run(beams=beams,bcond=bcond,connections=connections))
    sim.run()


    U = sim['U']
    vonmises_stress = sim['vonmises_stress']
    #print(vonmises_stress)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.set_box_aspect(aspect = (2.625,1,1))


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

            ax.plot(x,y,z,color='k',linewidth=0.5)
            ax.scatter(na[0], na[1], na[2],s=7,color='yellow',edgecolors='black',linewidth=0.3)
            ax.scatter(nb[0], nb[1], nb[2],s=7,color='yellow',edgecolors='black',linewidth=0.3)


    # plot the cg:
    #cg = sim['cg']
    #cg_def = sim['cg_def']
    #ax.scatter(cg[0],cg[1],cg[2],color='blue',s=30,edgecolors='black',linewidth=0.5)
    #ax.scatter(cg_def[0],cg_def[1],cg_def[2],color='red',s=30,edgecolors='black')


    ax.set_xlim(-38,38)
    ax.set_ylim(-38,38)
    ax.set_zlim(-1,2)

    ax.view_init(28, 285)

    plt.savefig('plane.png',dpi=1200,bbox_inches='tight',transparent=True)
    plt.show()

