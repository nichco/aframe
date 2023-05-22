import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.beamgroup import BeamGroup
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import scipy.io as sio

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


axis_nodes_dict = sio.loadmat('axis_nodes.mat')
axis_nodes = axis_nodes_dict['axis_nodes']/39.3700787
# add the initial node:
axis_nodes = np.concatenate([np.array([[9.86,0,1.042]]),axis_nodes]) # check values
airfoil_ribs_points_dict = sio.loadmat('ribs_oml_points.mat')
airfoil_ribs_points = airfoil_ribs_points_dict['airfoil_ribs_points']/39.3700787


w, h = np.zeros((len(axis_nodes))), np.zeros((len(axis_nodes)))
w[0] = 1.52
h[0] = 0.17
for i in range(len(axis_nodes) - 1):
    top_left = airfoil_ribs_points[0,:,i]
    top_right = airfoil_ribs_points[9,:,i]
    bot_left = airfoil_ribs_points[19,:,i]
    bot_right = airfoil_ribs_points[10,:,i]
    w_top = np.linalg.norm(top_right - top_left)
    w_bot = np.linalg.norm(bot_right - bot_left)
    h_front = np.linalg.norm(top_left - bot_left)
    h_back = np.linalg.norm(top_right - bot_right)

    w[i] = (w_top + w_bot)/2
    h[i] = (h_front + h_back)/2



loads_dict = sio.loadmat('loads_2p5g_n1g_aero_static.mat')
static_forces = loads_dict['forces']*4.44822162
static_moments = loads_dict['moments']*0.11298482933333

forces, moments = np.zeros((len(axis_nodes),3)), np.zeros((len(axis_nodes),3))
for i in range(len(axis_nodes) - 2):
    forces[i+2,:] = static_forces[0,i,:]
    moments[i+2,:] = static_moments[0,i,:]




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('joints',default={})
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']


        axis_nodes_var = self.create_input('axis_nodes_var', axis_nodes)
        for beam_name in beams: 
            self.register_output(beam_name, 1*axis_nodes_var)

        self.create_input('b1_height',shape=(len(axis_nodes)), val=h)
        self.create_input('b1_width',shape=(len(axis_nodes)), val=w)
        self.create_input('b1_t_cap',shape=(len(axis_nodes) - 1), val=0.001)
        self.create_input('b1_t_web',shape=(len(axis_nodes) - 1), val=0.001)
        self.create_input('b1_forces',shape=(len(axis_nodes),3),val=forces)
        self.create_input('b1_moments',shape=(len(axis_nodes),3),val=moments)

        
        # solve the beam group:
        self.add(BeamGroup(beams=beams,bounds=bounds,joints=joints), name='BeamGroup')


        self.add_constraint('vonmises_stress',upper=450E6/1,scaler=1E-8)
        self.add_design_variable('b1_t_cap',lower=0.001,upper=0.03,scaler=1E3)
        self.add_design_variable('b1_t_web',lower=0.001,upper=0.03,scaler=1E3)
        self.add_objective('mass',scaler=1E-2)
        
        






if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 12})

    joints, bounds, beams = {}, {}, {}
    beams['b1'] = {'E': 69E9,'G': 26E9,'rho': 2700,'type': 'box','n': len(axis_nodes)}
    bounds['root'] = {'beam': 'b1','fpos': 'a','fdim':[1,1,1,1,1,1]}


    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()

    print(sim['b1_displacement'])
    exit()

    prob = CSDLProblem(problem_name='run_opt', simulator=sim)
    optimizer = SLSQP(prob, maxiter=1000, ftol=1E-8)
    optimizer.solve()
    optimizer.print_results()

    
    U = sim['U']
    vonmises_stress = sim['vonmises_stress']
    b1t_cap = sim['b1_t_cap']
    b1t_web = sim['b1_t_web']
    print('stress: ', vonmises_stress)
    print('t_web: ', b1t_web)
    print('t_cap: ', b1t_cap)
    print(sim['mass'])


    # plotting:
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

            ax.plot(x,y,z,color='k',label='_nolegend_',linewidth=2)
            ax.scatter(na[0], na[1], na[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=30)
            ax.scatter(nb[0], nb[1], nb[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=30)


    ax.set_xlim(9.5,11)
    ax.set_ylim(0,14)
    ax.set_zlim(0,10)
    plt.show()