import numpy as np
import csdl
import python_csdl_backend
import matplotlib.pyplot as plt
from aframe.group import Group




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options',default={})
        self.parameters.declare('beams',default={})
        self.parameters.declare('bcond',default={})
    def define(self):
        options = self.parameters['options']
        beams = self.parameters['beams']
        bcond = self.parameters['bcond']