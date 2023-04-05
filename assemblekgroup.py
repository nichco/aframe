import numpy as np
import csdl
import python_csdl_backend

from transform import Transform
from length import Length
from localstiff import LocalStiff
from local_stiff_permuted import LocalStiffPermuted
from local_stiff_transformed import LocalStiffTransformed


class AssembleKGroup(csdl.Model):
    """Assemble that there compact local stiffness matrix."""

    def initialize(self):
        self.parameters.declare('options')

    def setup(self):
        options = self.parameters['options']

        comp = Transform(options=options)
        self.add("transform", comp, promotes=["*"])

        comp = Length(options=options)
        self.add("length", comp, promotes=["*"])

        self.add(LocalStiff(options=options), name='LocalStiff', promotes=["*"])

        comp = LocalStiffPermuted(options=options)
        self.add("local_stiff_permuted", comp, promotes=["*"])

        comp = LocalStiffTransformed(options=options)
        self.add("local_stiff_transformed", comp, promotes=["*"])