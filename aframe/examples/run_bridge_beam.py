import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.beamgroup import BeamGroup
plt.rcParams.update(plt.rcParamsDefault)


class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('joints',default={})
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']


        # dummy mesh generation code:
        for beam_name in beams:
            num_beam_nodes = beams[beam_name]['n']
            # get the beam start/stop coordinates
            a = self.create_input(beam_name+'a',shape=(3),val=beams[beam_name]['a'])
            b = self.create_input(beam_name+'b',shape=(3),val=beams[beam_name]['b'])
            ds = (b - a)/(num_beam_nodes - 1)

            mesh = self.create_output(beam_name, shape=(num_beam_nodes,3), val=0)
            for i in range(num_beam_nodes):
                node_i = a + ds*i
                mesh[i,:] = csdl.reshape(node_i, (1,3))
        


        # add a load:
        dummy_loads = np.zeros((5,3))
        dummy_loads[-1,1] = 1000000
        self.create_input('b4_forces',shape=(5,3),val=dummy_loads)

        # solve the beam group:
        self.add(BeamGroup(beams=beams,bounds=bounds,joints=joints), name='BeamGroup')






if __name__ == '__main__':

    beams, bounds, joints = {}, {}, {}


    beams['b1'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [0,0,0],'b': [1,0,0]}

    
    beams['b2'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [0,0,0],'b': [0.5,1,0]}
    beams['b3'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [0.5,1,0],'b': [1,0,0]}
    beams['b4'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [0.5,1,0],'b': [1.5,1,0]}
    beams['b5'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [1,0,0],'b': [1.5,1,0]}
    beams['b6'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [1,0,0],'b': [2,0,0]}
    beams['b7'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [1.5,1,0],'b': [2,0,0]}
    beams['b8'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [1.5,1,0],'b': [2.5,1,0]}
    beams['b9'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [2,0,0],'b': [2.5,1,0]}
    beams['b10'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [2,0,0],'b': [3,0,0]}
    beams['b11'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'tube','n': 5,'a': [2.5,1,0],'b': [3,0,0]}



    # boundary conditions:
    name = 'fixed_left'
    bounds[name] = {}
    bounds[name]['beam'] = 'b1'
    bounds[name]['fpos'] = 'a'
    bounds[name]['fdim'] = [1,1,1,1,1,1] # x, y, z, phi, theta, psi: a 1 indicates the corresponding dof is fixed

    name = 'fixed_right'
    bounds[name] = {}
    bounds[name]['beam'] = 'b10'
    bounds[name]['fpos'] = 'b'
    bounds[name]['fdim'] = [1,1,1,1,1,1]


    # joints:
    name = 'c1'
    joints[name] = {}
    joints[name]['beam_names'] = ['b1','b2']
    joints[name]['nodes'] = ['a','a'] # connects the end of b1 to the start of b2

    name = 'c2'
    joints[name] = {}
    joints[name]['beam_names'] = ['b2','b3','b4']
    joints[name]['nodes'] = ['b','a','a']

    name = 'c3'
    joints[name] = {}
    joints[name]['beam_names'] = ['b1','b3','b5','b6']
    joints[name]['nodes'] = ['b','b','a','a']

    name = 'c4'
    joints[name] = {}
    joints[name]['beam_names'] = ['b4','b5','b7','b8']
    joints[name]['nodes'] = ['b','b','a','a']

    name = 'c5'
    joints[name] = {}
    joints[name]['beam_names'] = ['b6','b7','b9','b10']
    joints[name]['nodes'] = ['b','b','a','a']

    name = 'c6'
    joints[name] = {}
    joints[name]['beam_names'] = ['b8','b9','b11']
    joints[name]['nodes'] = ['b','b','a']

    name = 'c7'
    joints[name] = {}
    joints[name]['beam_names'] = ['b10','b11']
    joints[name]['nodes'] = ['b','b']





    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()




    # plot the undeformed bridge:
    for beam_name in beams:
        num_beam_nodes = beams[beam_name]['n']
        num_elements = num_beam_nodes - 1

        for i in range(num_elements):
            element_name = beam_name + '_element_' + str(i)
            a = sim[element_name+'node_a_position']
            b = sim[element_name+'node_b_position']

            xu = np.array([a[0],b[0]])
            yu = np.array([a[1],b[1]])
            zu = np.array([a[2],b[2]])

            plt.plot(xu,yu,color='silver',linewidth=4)
            plt.scatter(xu,yu,s=50,color='silver',edgecolors='black',linewidth=0.5,zorder=5)


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


    # plot the applied force arrow:
    plt.arrow(1.5,1,0,0.2,width=0.04,color='red')


    plt.show()