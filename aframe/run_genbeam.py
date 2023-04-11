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
            type = beams[beam_name]['type']
            E = beams[beam_name]['E']
            G = beams[beam_name]['G']
            rho = beams[beam_name]['rho']

            start = self.declare_variable(beam_name+'start',shape=(6),val=[0,0,0,0,0,0])
            stop = self.declare_variable(beam_name+'stop',shape=(6),val=[10,0,0,0,0,0])
            
            for i in range(len(nodes) - 1):
                element_name = beam_name + '_element_' + str(i)
                options[element_name] = {}
                # constant material properties:
                options[element_name]['type'] = type
                options[element_name]['E'], options[element_name]['G'], options[element_name]['rho'] = E, G, rho
                # define the elemental start node and stop node from the node list:
                options[element_name]['nodes'] = [nodes[i] , nodes[i+1]]
                # compute the elemental start and stop node coordinates:
                ds = (stop - start)/len(nodes)
                node_a = start + ds*i
                node_b = start + ds*(i + 1)
                # register the outputs:
                self.register_output(element_name+'node_a', node_a)
                self.register_output(element_name+'node_b', node_b)

        """
        # generate the beam mesh(es):
        beam_name = 'beam_1'
        start = self.declare_variable(beam_name+'start',shape=(6),val=[0,0,0,0,0,0])
        stop = self.declare_variable(beam_name+'stop',shape=(6),val=[10,0,0,0,0,0])
        nodes = beams[beam_name]['nodes']

        type = beams[beam_name]['type']
        E = beams[beam_name]['E']
        G = beams[beam_name]['G']
        rho = beams[beam_name]['rho']
        
        for i in range(len(nodes) - 1):
            element_name = 'element_' + str(i)
            options[element_name] = {}
            # constant material properties:
            options[element_name]['type'] = type
            options[element_name]['E'], options[element_name]['G'], options[element_name]['rho'] = E, G, rho
            # define the elemental start node and stop node from the node list:
            options[element_name]['nodes'] = [nodes[i] , nodes[i+1]]
            # compute the elemental start and stop node coordinates:
            ds = (stop - start)/len(nodes)
            node_a = start + ds*i
            node_b = start + ds*(i + 1)
            # register the outputs:
            self.register_output(element_name+'node_a', node_a)
            self.register_output(element_name+'node_b', node_b)
        """
        
        # generate the loads vector(s) (for each beam if there are multiple beams):
        #beam_name = 'beam1'
        #loads = self.declare_variable(beam_name+'loads',shape=(6,len(nodes)))


        
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

    options, bcond, beams = {}, {}, {}

    bcond['root1'] = {}
    bcond['root1']['node'] = 0
    bcond['root1']['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed

    bcond['root2'] = {}
    bcond['root2']['node'] = 10
    bcond['root2']['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed


    name = 'beam_1'
    beams[name] = {}
    beams[name]['nodes'] = [0,1,2,3,4,5,6,7,8,9]
    beams[name]['type'] = 'tube'
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700

    name = 'beam_2'
    beams[name] = {}
    beams[name]['nodes'] = [10,11,12,13,14,15,16,17,18,19]
    beams[name]['type'] = 'tube'
    beams[name]['E'] = 69E9
    beams[name]['G'] = 26E9
    beams[name]['rho'] = 2700




    sim = python_csdl_backend.Simulator(Run(options=options,bcond=bcond,beams=beams))
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