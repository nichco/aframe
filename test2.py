import numpy as np
import csdl
import python_csdl_backend
from localk import LocalK



class Aframe(csdl.Model):

    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})


    def tube(self, name, t, r):
        r1, r2 = r - t, r
        A = np.pi * (r2**2 - r1**2)
        Iy = np.pi * (r2**4 - r1**4) / 4.0
        Iz = np.pi * (r2**4 - r1**4) / 4.0
        J = np.pi * (r2**4 - r1**4) / 2.0

        self.register_output(name + '_A', A)
        self.register_output(name + '_Iy', Iy)
        self.register_output(name + '_Iz', Iz)
        self.register_output(name + '_J', J)



    def box(self, name, w, h, tweb, tcap):
        w_i = w - 2*tweb
        h_i = h - 2*tcap
        A = (w*h) - (w_i*h_i)
        Iz = ((w**3)*h - (w_i**3)*h_i)/12
        Iy = (w*(h**3) - w_i*(h_i**3))/12
        J = (w*h*(h**2 + w**2)/12) - (w_i*h_i*(h_i**2 + w_i**2)/12)
        Q = 2*(h/2)*tweb*(h/4) + (w - 2*tweb)*tcap*((h/2) - (tcap/2))

        self.register_output(name+'A', A)
        self.register_output(name+'Iy', Iy)
        self.register_output(name+'Iz', Iz)
        self.register_output(name+'J', J)
        self.register_output(name+'Q', Q)





    def add_beam(self, name, nodes, cs, e, g, rho):
        n = len(nodes)

        mesh = self.declare_variable(name + '_mesh', shape=(n,3))
        #self.create_input(name + '_E', e)
        #self.create_input(name + '_G', g)
        
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
                self.tube(name=element_name, t=t[i], r=r[i])


        elif cs == 'box':
            w = self.declare_variable(name + '_w', shape=(n-1))
            h = self.declare_variable(name + '_h', shape=(n-1))
            tweb = self.declare_variable(name + '_tweb', shape=(n-1))
            tcap = self.declare_variable(name + '_tcap', shape=(n-1))

            for i in range(n - 1):
                element_name = name + '_element_' + str(i)
                self.box(name=element_name, w=w[i], h=h[i], tweb=tweb[i], tcap=tcap[i])

        # calculate the stiffness matrix for each element:
        for i in range(n - 1):
            element_name = name + '_element_' + str(i)

            E, G = e, g







    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']


        for name in beams:
            self.add_beam(name=name, 
                          nodes=beams[name]['nodes'], 
                          cs=beams[name]['cs'], 
                          e=beams[name]['E'],
                          g=beams[name]['G'],
                          rho=beams[name]['rho'])









beams, bounds, joints = {}, {}, {}
beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(10))}
beams['boom'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(10))}
joints['joint'] = {'a': 4,'b': 4}

sim = python_csdl_backend.Simulator(Aframe(beams=beams, joints=joints))
sim.run()

print(sim['wing_element_1_A'])