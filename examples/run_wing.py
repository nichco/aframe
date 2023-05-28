import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.beamgroup import BeamGroup
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

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
        


        dummy_loads = np.zeros((8,3))
        dummy_loads[:,2] = -1*np.flip(np.array([0,-102.3,-144.9,-161.5,-176.22,-183.54,-187.04,-189.1]))
        self.create_input('b1_forces',shape=(8,3),val=dummy_loads*3)

        #plt.plot(dummy_loads[:,2])
        #plt.show()
        #exit()



        #self.create_input('b1thickness',shape=(9),val=0.002)
        #self.create_input('b1radius',shape=(9),val=0.25)
        self.create_input('b1_height',shape=(8),val=np.linspace(0.153,0.1,8))
        self.create_input('b1_width',shape=(8),val=np.linspace(0.2133,0.463,8))
        #self.create_input('b1t_web',shape=(14),val=0.002)
        self.create_input('b1_t_cap',shape=(7),val=np.array([0.0005, 0.00063, 0.00079902, 0.0009, 0.00187722, 0.0030468, 0.00421048]))
        
        # solve the beam group:
        self.add(BeamGroup(beams=beams,bounds=bounds,joints=joints), name='BeamGroup')


        self.add_constraint('vonmises_stress',upper=450E6/2,scaler=1E-8)
        #self.add_constraint('margin',equals=0,scaler=1)

        #self.add_design_variable('b1thickness',lower=0.0001,scaler=10)
        #self.add_design_variable('b1radius',lower=0.1,upper=0.5,scaler=1)
        #self.add_design_variable('b1_height',lower=0.1,upper=0.3,scaler=1)
        #self.add_design_variable('b1_width',lower=0.1,upper=0.5,scaler=1)
        #self.add_design_variable('b1t_web',lower=0.0005,upper=0.01,scaler=1E4)
        self.add_design_variable('b1_t_cap',lower=0.0001,upper=0.01,scaler=1E3)
        self.add_objective('mass',scaler=1E-2)
        
        






if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 12})

    joints, bounds, beams = {}, {}, {}

    
    name = 'b1'
    beams[name] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'box','n': 8,'a': [0,0,0],'b': [0,10,0]}
    

    name = 'root1'
    bounds[name] = {}
    bounds[name]['beam'] = 'b1'
    bounds[name]['fpos'] = 'a'
    bounds[name]['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed




    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()

    exit()

    prob = CSDLProblem(problem_name='run_opt', simulator=sim)
    optimizer = SLSQP(prob, maxiter=1000, ftol=1E-8)
    optimizer.solve()
    optimizer.print_results()

    
    U = sim['U']
    vonmises_stress = sim['vonmises_stress']
    #b1thickness = sim['b1thickness']
    #b1radius = sim['b1radius']
    b1height = sim['b1_height']
    b1width = sim['b1_width']
    #b1t_web = sim['b1t_web']
    b1t_cap = sim['b1t_cap']
    print('stress: ', vonmises_stress)
    print('height: ', b1height)
    print('width: ', b1width)
    #print('t_web: ', b1t_web)
    print('t_cap: ', b1t_cap)
    print(sim['mass'])

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


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

            ax.plot(x,y,z,color='k',label='_nolegend_',linewidth=max(b1t_cap[i]*0.5E4,1))
            ax.scatter(na[0], na[1], na[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=max(b1t_cap[i]*1E5,40))
            ax.scatter(nb[0], nb[1], nb[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=max(b1t_cap[i]*1E5,40))



    plt.show()