import csdl


# take meshes and compute the box-beam width and height

class BoxProp(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')

    def define(self):
        beams = self.parameters['beams']

        for beam_name in beams:
            num_nodes = beams[beam_name]['n']
            num_elements = num_nodes - 1

            chord = self.declare_variable(beam_name+'chord',shape=(num_nodes,2))

            le = chord[:,0]
            te = chord[:,0]

            width = csdl.pnorm(le - te, axis=1)

            # process the mesh by assigning it to elements:
            for i in range(num_elements):
                element_name = beam_name + '_element_' + str(i)

                self.register_output(element_name+'width', width[i])
                #self.register_output(element_name+'height', height[i])