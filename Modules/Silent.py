#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:47:09 2017

@author: bgris
"""



# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import scipy.ndimage
import numpy as np

from odl.discr import Gradient
from odl.space import ProductSpace
from odl.discr import ResizingOperator
from odl.trafos import FourierTransform
import odl
from odl.deform.linearized import linear_deform
from Modules.DeformationModuleAbstract import DeformationModule



__all__ = ('Silent', )



def padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padded_op.range, halfcomplex=False, shift=shifts, impl='pyfftw')

    return ft_op * padded_op

def fitting_kernel(space, kernel):

    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel



class Silent(DeformationModule):
    def __init__(self,DomainField, inv_N):
        """Initialize a new instance.
        DomainField : space to which GD belongs
        invN : used for small differences for applying vector field 
         (supposed to be stepsize for integration)
        """

        self.dim=DomainField.ndim
        self.inv_N = inv_N
        self.grad_op = Gradient(DomainField)
        GDspace = DomainField
        Contspace = odl.space.rn(0)
        self.dimCont = 0

        super().__init__(GDspace,Contspace, DomainField)

    def get_columnvector_control(self, Cont):
        """
        return controls as a column vector
        """
        return Cont
    
    def get_spaceelement_control(self, Cont_vector):
        """
        return controls as a Contspace.element
        """
        return Cont_vector
    
 
    def ComputeField(self, o,h):
        """Return the computed vector field on DomainField
        """
                            

        return self.domain.tangent_bundle.zero()


    def AdjointComputeField(self, o, vectfield):
        
            
        return self.Contspace.zero()
        
#    @property
    def ComputeFieldEvaluate(self, o, h, points):
        " points is a list of points "
        return np.zeros_like(points)
        



#    @property
    def ComputeFieldEvaluateDer(self, o, h, points, vectors):
        return np.zeros_like(points)
        



#    @property
    def AdjointDerEvaluate(self, o, h, vect_field):
            
        return self.GDspace.zero()
        


#    @property
    def ApplyField(self,GD,vect_field):

        I_tmp=self.GDspace.element(linear_deform(GD, -self.inv_N * vect_field).copy())

        return (1/self.inv_N) * self.GDspace.element(I_tmp - GD)


#    @property
    def ApplyModule(self, GD, Module, GDmod, Contmod):
        return self.ApplyField(GD, Module.ComputeField(GDmod, Contmod))
        
      
    
#    @property
    def AdjointApply(self, GD, Mom):
        grad = self.grad_op(GD)
        return -grad*Mom
        
            
#    @property
    def AdjointDerApply(self, GD, Mom, vect_field):
        div_op = -self.grad_op.adjoint
        fac_smooth = 1
        Mom_smooth = self.domain.element(scipy.ndimage.filters.gaussian_filter(Mom.asarray(),fac_smooth))
        return -div_op(Mom_smooth * vect_field)

    def AdjointDerApplyMod(self, GD, Mom, Mod, GDMod, ContMod):

        return self.AdjointDerApply(GD, Mom, Mod.ComputeField(GDMod, ContMod))

        
        
    def Cost(self,GD,Cont):
        
        return 0.0


#    @property
    def DerCost(self, GD, Cont):
        
        return self.GDspace.zero()
        
#    @property
    def InvCo(self, GD):
        return np.zeros((0,0))

#
