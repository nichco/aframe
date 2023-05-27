import numpy as np
import csdl
import python_csdl_backend
from localk import LocalK



class Aframe(csdl.Model):

    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})


    def tube(self, element_name, t, r):
        r1, r2 = r - t, r
        A = np.pi * (r2**2 - r1**2)
        Iy = np.pi * (r2**4 - r1**4) / 4.0
        Iz = np.pi * (r2**4 - r1**4) / 4.0
        J = np.pi * (r2**4 - r1**4) / 2.0

        self.register_output(element_name + '_A', A)
        self.register_output(element_name + '_Iy', Iy)
        self.register_output(element_name + '_Iz', Iz)
        self.register_output(element_name + '_J', J)


    def box(self, element_name, w, h, tweb, tcap):
        w_i = w - 2*tweb
        h_i = h - 2*tcap
        A = (w*h) - (w_i*h_i)
        Iz = ((w**3)*h - (w_i**3)*h_i)/12
        Iy = (w*(h**3) - w_i*(h_i**3))/12
        J = (w*h*(h**2 + w**2)/12) - (w_i*h_i*(h_i**2 + w_i**2)/12)
        Q = 2*(h/2)*tweb*(h/4) + (w - 2*tweb)*tcap*((h/2) - (tcap/2))

        self.register_output(element_name + 'A', A)
        self.register_output(element_name + 'Iy', Iy)
        self.register_output(element_name + 'Iz', Iz)
        self.register_output(element_name + 'J', J)
        self.register_output(element_name + 'Q', Q)


    def local_stiffness(self, element_name, E, G, node_dict, node_index, dim, i):
        A = self.declare_variable(element_name+'A')
        Iy = self.declare_variable(element_name+'Iy')
        Iz = self.declare_variable(element_name+'Iz')
        J = self.declare_variable(element_name+'J')

        node_a = self.declare_variable(element_name + 'node_a', shape=(3))
        node_b = self.declare_variable(element_name + 'node_b', shape=(3))

        L = self.register_output(element_name + '_L', csdl.pnorm(node_b - node_a, pnorm_type=2))

        kp = self.create_output(element_name+'kp',shape=(12,12),val=0)
        # the upper left block
        kp[0,0] = csdl.reshape(A*E/L, (1,1))
        kp[1,1] = csdl.reshape(12*E*Iz/L**3, (1,1))
        kp[1,5] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[5,1] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[2,2] = csdl.reshape(12*E*Iy/L**3, (1,1))
        kp[2,4] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[4,2] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[3,3] = csdl.reshape(G*J/L, (1,1))
        kp[4,4] = csdl.reshape(4*E*Iy/L, (1,1))
        kp[5,5] = csdl.reshape(4*E*Iz/L, (1,1))
        # the upper right block
        kp[0,6] = csdl.reshape(-A*E/L, (1,1))
        kp[1,7] = csdl.reshape(-12*E*Iz/L**3, (1,1))
        kp[1,11] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[2,8] = csdl.reshape(-12*E*Iy/L**3, (1,1))
        kp[2,10] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[3,9] = csdl.reshape(-G*J/L, (1,1))
        kp[4,8] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[4,10] = csdl.reshape(2*E*Iy/L, (1,1))
        kp[5,7] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[5,11] = csdl.reshape(2*E*Iz/L, (1,1))
        # the lower left block
        kp[6,0] = csdl.reshape(-A*E/L, (1,1))
        kp[7,1] = csdl.reshape(-12*E*Iz/L**3, (1,1))
        kp[7,5] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[8,2] = csdl.reshape(-12*E*Iy/L**3, (1,1))
        kp[8,4] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[9,3] = csdl.reshape(-G*J/L, (1,1))
        kp[10,2] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[10,4] = csdl.reshape(2*E*Iy/L, (1,1))
        kp[11,1] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[11,5] = csdl.reshape(2*E*Iz/L, (1,1))
        # the lower right block
        kp[6,6] = csdl.reshape(A*E/L, (1,1))
        kp[7,7] = csdl.reshape(12*E*Iz/L**3, (1,1))
        kp[7,11] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[11,7] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[8,8] = csdl.reshape(12*E*Iy/L**3, (1,1))
        kp[8,10] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[10,8] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[9,9] = csdl.reshape(G*J/L, (1,1))
        kp[10,10] = csdl.reshape(4*E*Iy/L, (1,1))
        kp[11,11] = csdl.reshape(4*E*Iz/L, (1,1))

        # transform the local stiffness to global coordinates:
        cp = (node_b - node_a)/csdl.expand(L, (3))
        ll, mm, nn = cp[0], cp[1], cp[2]
        D = (ll**2 + mm**2)**0.5

        block = self.create_output(element_name + 'block',shape=(3,3),val=0)
        block[0,0] = csdl.reshape(ll, (1,1))
        block[0,1] = csdl.reshape(mm, (1,1))
        block[0,2] = csdl.reshape(nn, (1,1))
        block[1,0] = csdl.reshape(-mm/D, (1,1))
        block[1,1] = csdl.reshape(ll/D, (1,1))
        block[2,0] = csdl.reshape(-ll*nn/D, (1,1))
        block[2,1] = csdl.reshape(-mm*nn/D, (1,1))
        block[2,2] = csdl.reshape(D, (1,1))

        T = self.create_output(element_name + 'T',shape=(12,12),val=0)
        T[0:3,0:3] = 1*block
        T[3:6,3:6] = 1*block
        T[6:9,6:9] = 1*block
        T[9:12,9:12] = 1*block

        tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))

        # expand the transformed stiffness matrix to the global dimensions:
        k = self.create_output(element_name + 'k',shape=(dim,dim),val=0)

        # parse tkt:
        k11 = tkt[0:6,0:6] # upper left
        k12 = tkt[0:6,6:12] # upper right
        k21 = tkt[6:12,0:6] # lower left
        k22 = tkt[6:12,6:12] # lower right

        # assign the four block matrices to their respective positions in k:
        node_a_index = node_index[node_dict[i]]
        node_b_index = node_index[node_dict[i + 1]]

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k11

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k12

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k21

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k22


    def add_beam(self, name, nodes, cs, e, g, rho, node_dict, node_index, dim):
        n = len(nodes)

        default_val = np.zeros((n, 3))
        default_val[:,1] = np.linspace(0,n,n)
        mesh = self.declare_variable(name + '_mesh', shape=(n,3), val=default_val)
        
        # iterate over each element:
        for i in range(n - 1):
            element_name = name + '_element_' + str(i)
            node_a = csdl.reshape(mesh[i, :], (3))
            node_b = csdl.reshape(mesh[i + 1, :], (3))
            self.register_output(element_name + 'node_a', node_a)
            self.register_output(element_name + 'node_b', node_b)


        if cs == 'tube':
            t = self.declare_variable(name + '_t', shape=(n-1))
            r = self.declare_variable(name + '_r', shape=(n-1))

            for i in range(n - 1):
                element_name = name + '_element_' + str(i)
                self.tube(element_name=element_name, t=t[i], r=r[i])


        elif cs == 'box':
            w = self.declare_variable(name + '_w', shape=(n-1))
            h = self.declare_variable(name + '_h', shape=(n-1))
            tweb = self.declare_variable(name + '_tweb', shape=(n-1))
            tcap = self.declare_variable(name + '_tcap', shape=(n-1))

            for i in range(n - 1):
                element_name = name + '_element_' + str(i)
                self.box(element_name=element_name, w=w[i], h=h[i], tweb=tweb[i], tcap=tcap[i])

        # calculate the stiffness matrix for each element:
        for i in range(n - 1):
            element_name = name + '_element_' + str(i)

            self.local_stiffness(element_name=element_name, 
                                 E=e, 
                                 G=g, 
                                 node_dict=node_dict, 
                                 node_index=node_index, 
                                 dim=dim,
                                 i=i)





    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']

        # automated beam node assignment:
        node_dict = {}
        # start by populating the nodes dictionary as if there aren't any joints:
        index = 0
        for beam_name in beams:
            node_dict[beam_name] = np.arange(index, index + len(beams[beam_name]['nodes']))
            index += len(beams[beam_name]['nodes'])

        # assign nodal indices in the global system:
        for joint_name in joints:
            joint_beam_list = joints[joint_name]['beams']
            joint_node_list = joints[joint_name]['nodes']
            joint_node_a = node_dict[joint_beam_list[0]][joint_node_list[0]]
            
            for i, beam_name in enumerate(joint_beam_list):
                if i != 0: node_dict[beam_name][joint_node_list[i]] = joint_node_a



        node_set = set(node_dict[beam_name][i] for beam_name in beams for i in range(len(beams[beam_name]['nodes'])))
        num_unique_nodes = len(node_set)
        dim = num_unique_nodes*6
        # create a dictionary that contains the nodes and the node index in the global system:
        node_index = {list(node_set)[i]: i for i in range(num_unique_nodes)}
        # print(node_set)
        # print(node_dict)
        # print(node_index)



        # create a list of element names:
        elements = []
        num_elements = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            num_elements += n - 1
            for i in range(n - 1): elements.append(beam_name + '_element_' + str(i))



        
        for beam_name in beams:
            self.add_beam(name=beam_name, 
                          nodes=beams[beam_name]['nodes'], 
                          cs=beams[beam_name]['cs'], 
                          e=beams[beam_name]['E'],
                          g=beams[beam_name]['G'],
                          rho=beams[beam_name]['rho'],
                          node_dict=node_dict[beam_name],
                          node_index=node_index,
                          dim=dim)


        # compute the global stiffness matrix:
        helper = self.create_output('helper', shape=(num_elements,dim,dim), val=0)
        for i, element_name in enumerate(elements):
            k = self.declare_variable(element_name + 'k', shape=(dim,dim))
            helper[i,:,:] = csdl.reshape(k, (1,dim,dim))

        sum_k = csdl.sum(helper, axes=(0, ))

        b_index_list = []
        for b_name in bounds:
            beam_name = bounds[b_name]['beam']
            fpos = bounds[b_name]['node']
            fdim = bounds[b_name]['fdim']
            b_node_index = node_index[node_dict[beam_name][fpos]]

            # add the constrained dof index to the b_index_list:
            for i, fdim in enumerate(fdim):
                if fdim == 1: b_index_list.append(b_node_index*6 + i)



        mask = self.create_output('mask',shape=(dim,dim),val=np.eye(dim))
        mask_eye = self.create_output('mask_eye',shape=(dim,dim),val=0)
        zero = self.create_input('zero',shape=(1,1),val=0)
        one = self.create_input('one',shape=(1,1),val=1)
        [(mask.__setitem__((i,i),1*zero), mask_eye.__setitem__((i,i),1*one)) for i in range(dim) if i in b_index_list]

        # modify the global stiffness matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye
        self.register_output('K', K)











beams, bounds, joints = {}, {}, {}
beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(10))}
beams['boom'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(10))}
joints['joint'] = {'beams': ['wing', 'boom'],'nodes': [4, 4]}
bounds['root'] = {'beam': 'wing','node': 0,'fdim': [1,1,1,1,1,1]}

sim = python_csdl_backend.Simulator(Aframe(beams=beams, joints=joints))
sim.run()

print(sim['wing_element_1_A'])