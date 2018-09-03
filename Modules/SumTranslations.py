#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:00:54 2018

@author: barbara
"""



# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super


import numpy as np

from odl.discr import Gradient
from odl.space import ProductSpace
from odl.discr import ResizingOperator
from odl.trafos import FourierTransform
import odl
from Modules.DeformationModuleAbstract import DeformationModule
import Modules.Usefulfunctions_DeformationModule as usefun

import copy

__all__ = ('SumTranslations', )



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



class SumTranslations(DeformationModule):
    def __init__(self,DomainField, Ntrans, kernel, partialderdim1kernel, partialder2kernel):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        Ntrans : number of translations
        Kernel : kernel
        partialderdim1kernel :  partial derivative with respect to one componenet of
        the 1st component, 
         to be used as partialder2kernel(x, y, d) where d is in [|0, dim-1|]
        partialder2kernel : partial derivative with respect to 2nd component, 
         to be used as partialder2kernel(x, y, u) with x and y points, and u 
         vectors for differentiation (same number of vectors as the number of 
         points for y)
        """

        self.Ntrans=Ntrans
        self.kernel=kernel
        self.partialderdim1kernel = partialderdim1kernel
        self.partialder2kernel = partialder2kernel
        self.dim=DomainField.ndim
        self.get_unstructured_op = usefun.get_from_structured_to_unstructured(DomainField, kernel)
        self.gradient_op = Gradient(DomainField)
        GDspace=odl.ProductSpace(odl.space.rn(self.dim),self.Ntrans)
        Contspace=odl.ProductSpace(odl.space.rn(self.dim),self.Ntrans)
        self.dimCont = self.Ntrans * self.dim


        super().__init__(GDspace,Contspace, DomainField)

    def get_columnvector_control(self, Cont):
        """
        return controls as a column vector
        """
        return np.array([Cont[u][v] for u in range(self.Ntrans) for v in range(self.dim)])
    
    def get_spaceelement_control(self, Cont_vector):
        """
        return controls as a Contspace.element
        """
        return self.Contspace.element([[Cont_vector[ self.dim * u +v] for v in range(self.dim)] for u in range(self.Ntrans)])
    
    
    def CreateStructuredFromGDCont(self, o, h):
        return usefun.create_structured(np.transpose(np.array(o)), np.transpose(np.array(h)))
        
 
    def ComputeField(self, o,h):
        """Return the computed vector field on DomainField
        """
        if o not in self.GDspace:
            try:
                o = self.GDspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' o is not in `GDspace` instance'
                            '')

        if h not in self.Contspace:
            try:
                o = self.Contspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' h is not in `Contspace` instance'
                            '')

        return self.get_unstructured_op(self.CreateStructuredFromGDCont(o, h))


    def AdjointComputeField(self, o, vectfield):
        
        def app_kernel(z):
            return lambda x: self.kernel(z, x)

        adjoint = []
        for i in range(self.Ntrans):
            appli = self.domain.element(app_kernel(o[i]))
            fun_field = [appli.inner(vu) for vu in vectfield]
            adjoint.append(copy.deepcopy(fun_field))
            
        return self.Contspace.element(adjoint)
       
        

        
        
        
#    @property
    def ComputeFieldEvaluate(self, o, h, points):
        " points is a list of points "
        points = np.array(points)
        mat = usefun.make_covariance_mixte_matrix(np.transpose(np.array(points)), np.transpose(np.array(o)), self.kernel)
        return np.dot(mat, np.array(h))
        



#    @property
    def ComputeFieldEvaluateDer(self, o, h, points, vectors):
        points = np.array(points)
        vectors = np.array(vectors)
        mat = usefun.Make_der_CovMat(np.transpose(np.array(o)), np.transpose(points), np.transpose(vectors), self.partialder2kernel)
        return np.dot(mat, np.array(h))
        



#    @property
    def AdjointDerEvaluate(self, o, h, vect_field):
        # def an operator applying partial derivative of kernel
        def app_der_kernel(z, d):
            return lambda x: self.partialderdim1kernel(z, x, d)
        
        adjoint = []
        
        for i in range(self.Ntrans):
            fun_field = sum([hi * vi for hi, vi in zip(h[i], vect_field)])
            #fun_field = self.domain.element(sum([hi * vi for hi, vi in zip(h[i], vect_field)]))
            adjoint.append([fun_field.inner(self.domain.element(app_der_kernel(o[i], j))) for j in range(self.dim)])
            
        return self.GDspace.element(adjoint)
        


#    @property
    def ApplyField(self,GD,vect_field):

            speed=self.GDspace.element(np.array([vect_field[i].interpolation(np.array(GD).T) for i in range(self.dim)]).T)

            return speed


#    @property
    def ApplyModule(self, GD, Module, GDmod, Contmod):
        return self.GDspace.element(Module.ComputeFieldEvaluate(GDmod, Contmod, GD))
        
    
#    @property
    def AdjointApply(self, GD, Mom):
        return usefun.Dirac(np.array(GD), np.array(Mom), self.domain)
        
            
#    @property
    def AdjointDerApply(self, GD, Mom, vect_field):
        grad = [self.gradient_op(v) for v in vect_field]
        #grad_inter[i][j][k] is j-th gradient of vect_field[i] applied to GD[k] :
        # it is (Dv(GD[k]))_{i, j}
        grad_inter = np.array([[gu.interpolation(np.transpose(np.array(GD))) for gu in g] for g in grad])
        # adjoint[j][k] is sum_i grad_inter[i][j][k] * Mom[i][k] :
        # it is Dv(GD[k])^T Mom
        adjoint = sum([np.array(momi) * gi for momi, gi in zip(np.transpose(np.array(Mom)), grad_inter) ])
        
        return self.GDspace.element(np.transpose(np.array(adjoint)))

    def AdjointDerApplyMod(self, GD, Mom, Mod, GDMod, ContMod):
        #before : grad_inter[i][j][k] is j-th gradient of vect_field[i] applied to GD[k] :
        # it is (Dv(GD[k]))_{i, j}
        u1 = np.ones(self.Ntrans)
        u2 = np.zeros(self.Ntrans)
        #grad_inter[i][j][k] is i-th gradient of vect_field[k] applied to GD[j] :
        # it is (Dv(GD[j]))_{k, i} with vect_field generated by Mod with GDMod, ContMod
        grad_inter = []
        grad_inter.append(Mod.ComputeFieldEvaluateDer(GDMod, ContMod, GD, np.transpose(np.array([u1, u2]))))
        grad_inter.append(Mod.ComputeFieldEvaluateDer(GDMod, ContMod, GD, np.transpose(np.array([u2, u1]))))
        
        grad_inter = np.transpose(np.array(grad_inter), (2, 0, 1))
        
        # adjoint[j][k] is sum_i grad_inter[i][j][k] * Mom[i][k] :
        # it is Dv(GD[k])^T Mom
        adjoint = sum([np.array(momi) * gi for momi, gi in zip(np.transpose(np.array(Mom)), grad_inter) ])
        
        return self.GDspace.element(np.transpose(np.array(adjoint)))

        
        
    def Cost(self,GD,Cont):
        mat = usefun.make_covariance_matrix(np.transpose(np.array(GD)), self.kernel)
        energy = sum([np.dot(np.dot(np.transpose(np.array(Cont))[u], mat), np.transpose(np.array(Cont))[u]) for u in range(self.dim)])
        
        return float(energy)


#    @property
    def DerCost(self, GD, Cont):
        u1 = np.ones(self.Ntrans)
        u2 = np.zeros(self.Ntrans)
        
        # Matrix with derivative wrt 1st component of covariance matrix
        Partial1_covMat = usefun.Make_der_CovMat(np.transpose(np.array(GD)), np.transpose(np.array(GD)), np.array([u1, u2]), self.partialder2kernel)
        
        # Matrix with derivative wrt 2nd component of covariance matrix
        Partial2_covMat = usefun.Make_der_CovMat(np.transpose(np.array(GD)), np.transpose(np.array(GD)), np.array([u2, u1]), self.partialder2kernel)
        
        der_1 = sum([np.dot(Partial1_covMat, Contu) * Contu for Contu in np.transpose(np.array(Cont))])
        der_2 = sum([np.dot(Partial2_covMat, Contu) * Contu for Contu in np.transpose(np.array(Cont))])
        
        return self.GDspace.element(np.transpose(np.array([der_1, der_2])))
        
#    @property
    def InvCo(self, GD):
        mat = usefun.make_covariance_matrix(np.transpose(np.array(GD)), self.kernel)
        return np.kron(np.linalg.inv(mat), np.identity(self.dim))

#
