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
        


        dummy_loads = np.zeros((10,3))
        dummy_loads[:,2] = 100
        self.create_input('b1_forces',shape=(10,3),val=dummy_loads)



        #self.create_input('b1thickness',shape=(9),val=0.002)
        #self.create_input('b1radius',shape=(9),val=0.25)
        self.create_input('b1height',shape=(10),val=0.5)
        self.create_input('b1width',shape=(10),val=0.25)
        self.create_input('b1t_web',shape=(9),val=0.005)
        self.create_input('b1t_cap',shape=(9),val=0.005)
        
        # solve the beam group:
        self.add(BeamGroup(beams=beams,bounds=bounds,joints=joints), name='BeamGroup')
 

        self.add_constraint('vonmises_stress',upper=450E6/6,scaler=1E-8)

        #self.add_design_variable('b1thickness',lower=0.0001,scaler=10)
        #self.add_design_variable('b1radius',lower=0.1,upper=0.5,scaler=1)
        self.add_design_variable('b1height',lower=0.1,upper=1,scaler=1)
        self.add_design_variable('b1width',lower=0.1,upper=1,scaler=1)
        self.add_design_variable('b1t_web',lower=0.001,upper=0.01,scaler=1E4)
        self.add_design_variable('b1t_cap',lower=0.001,upper=0.01,scaler=1E4)
        self.add_objective('total_mass',scaler=1E-1)
        
        






if __name__ == '__main__':

    joints, bounds, beams = {}, {}, {}

    
    name = 'b1'
    beams[name] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'box','n': 10,'a': [0,0,0],'b': [10,0,0]}
    

    name = 'root'
    bounds[name] = {}
    bounds[name]['beam'] = 'b1'
    bounds[name]['fpos'] = 'a'
    bounds[name]['fdim'] = [1,1,1,1,1,1] # [x, y, z, phi, theta, psi]: a 1 indicates the corresponding dof is fixed




    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    #sim.run()

    prob = CSDLProblem(problem_name='run_opt', simulator=sim)
    optimizer = SLSQP(prob, maxiter=1000, ftol=1E-8)
    optimizer.solve()
    optimizer.print_results()

    
    U = sim['U']
    vonmises_stress = sim['vonmises_stress']
    #b1thickness = sim['b1thickness']
    #b1radius = sim['b1radius']
    b1height = sim['b1height']
    b1width = sim['b1width']
    b1t_web = sim['b1t_web']
    b1t_cap = sim['b1t_cap']
    print('stress: ', vonmises_stress)
    print('height: ', b1height)
    print('width: ', b1width)
    print('t_web: ', b1t_web)
    print('t_cap: ', b1t_cap)
    print(sim['total_mass'])

    
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
    ax.set_zlim(-0.5,0.5)
    plt.show()
    

    """
    # validation:
    F = -200
    L = 10
    E = 69E9
    I = sim['b1_element_1Iy']
    dmax = F*(L**3)/(3*E*I)
    print('dmax: ',dmax)
    """