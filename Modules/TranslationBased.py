#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:00:54 2018

@author: gris
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


__all__ = ('TranslationBased', )



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



class TranslationBased(DeformationModule):
    def __init__(self,DomainField, GDspace, NbGD, kernel, partialderdim1kernel, partialder2kernel, pointfunctionlist, pointfunctiondifflist, vectorfunctionlist, vectorfunctiondifflist):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        dimGD = dimension of GDspace
        Kernel : kernel
        partialderdim1kernel :  partial derivative with respect to one componenet of
        the 1st component, 
         to be used as partialder2kernel(x, y, d) where d is in [|0, dim-1|]
        partialder2kernel : partial derivative with respect to 2nd component, 
         to be used as partialder2kernel(x, y, u) with x and y points, and u 
         vectors for differentiation (same number of vectors as the number of 
         points for y)
         get_columnvector_gd : function that takes a GD and return a vector
             so that when we multiply it by a point/vectorfunctiondiff we
             obtain its differential
         get_spaceelement_gd : function, inverse of get_columnvector_gd 
             (takes a vector and return the corresponding GD)
        pointfunctionlist : list of functions, each one associating a list of
              points to a geometrical descriptor
        pointfunctiondifflist : list of functions, each one associating a list of
              matrix to a geometrical descriptor, the k-th matrix is the differential
              of the k-th point with respect to the geometrical descriptor
        vectorfunctionlist : list of functions, each one associating a list of
              vectors to a geometrical descriptor
        vectorfunctiondifflist : list of functions, each one associating a list of
              matrix to a geometrical descriptor, the k-th matrix is the differential
              of the k-th vector with respect to the geometrical descriptor
              
        """

        self.kernel=kernel
        self.partialderdim1kernel = partialderdim1kernel
        self.partialder2kernel = partialder2kernel
        self.dim=DomainField.ndim
        self.NbGD = NbGD
        self.dimGD = NbGD * self.dim
        self.get_unstructured_op = usefun.get_from_structured_to_unstructured(DomainField, kernel)
        self.gradient_op = Gradient(DomainField)
        self.ptfun = pointfunctionlist
        self.vecfun = vectorfunctionlist
        self.ptfundiff = pointfunctiondifflist
        self.vecfundiff = vectorfunctiondifflist
        self.dimCont = len(pointfunctionlist)
        
        Contspace = odl.space.rn(self.dimCont)


        super().__init__(GDspace,Contspace, DomainField)

    def get_columnvector_control(self, Cont):
        """
        return controls as a column vector
        """
        return np.array(Cont)
    
    def get_spaceelement_control(self, Cont_vector):
        """
        return controls as a Contspace.element
        """
        return self.Contspace.element(Cont_vector)
    
    def get_columnvector_gd(self, GD):
        """
        return controls as a column vector
        """
        return np.array([GD[u][v] for u in range(self.NbGD) for v in range(self.dim)])
    
    def get_spaceelement_gd(self, GD_vector):
        """
        return controls as a Contspace.element
        """
        return self.GDspace.element([[GD_vector[ self.dim * u +v] for v in range(self.dim)] for u in range(self.NbGD)])

    def CreateStructuredListFromGDCont(self, o, h):
        return [usefun.create_structured(np.transpose(self.ptfun[i](o)), h[i] * np.transpose(self.vecfun[i](o))) for i in range(self.dimCont)]

    def CreateStructuredListFromGD(self, o):
        return [usefun.create_structured(np.transpose(self.ptfun[i](o)),  np.transpose(self.vecfun[i](o))) for i in range(self.dimCont)]
        
 
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
        stru_list = self.CreateStructuredListFromGDCont(o, h)
        return sum([self.get_unstructured_op(stru) for stru in stru_list])


    def AdjointComputeField(self, o, vectfield):
        
        stru_list = self.CreateStructuredListFromGD(o)

        adjoint = [vectfield.inner(self.get_unstructured_op(stru)) for stru in stru_list]

        return self.Contspace.element(adjoint)
       
        
        
#    @property
    def ComputeFieldEvaluate(self, o, h, points):
        " points is a list of points "
        points = np.array(points)

        mat_list = [usefun.make_covariance_mixte_matrix(np.transpose(points), np.transpose(funi(o)), self.kernel) for funi in self.ptfun]
        
        return sum([np.dot(mat_list[i], h[i] * np.array(self.vecfun[i](o))) for i in range(self.dimCont)])
        



#    @property
    def ComputeFieldEvaluateDer(self, o, h, points, vectors):
        points = np.array(points)
        vectors = np.array(vectors)
        
        mat_list = [usefun.Make_der_CovMat(np.transpose(funi(o)), np.transpose(points), np.transpose(vectors), self.partialder2kernel) for funi in self.ptfun]
        return sum([np.dot(mat_list[i], h[i] * np.array(self.vecfun[i](o)))  for i in range(self.dimCont)])
        



#    @property
    def AdjointDerEvaluate(self, o, h, vect_field):
        #print('AdjointDrEvaluate')
        # def an operator applying partial derivative of kernel
        def app_der_kernel(z, d):
            return lambda x: self.partialderdim1kernel(z, x, d)
                     
        def app_kernel(z):
            return lambda x: self.kernel(z, x)
        shape_field = vect_field.shape
        shape_im = vect_field[0].shape
        # we define one vector field per dimension of GD
        vect_field_list = [self.DomainField.tangent_bundle.zero() for _ in range(self.dimGD)]
        for i in range(self.dimCont):
            # list of differentials
            Df = self.ptfundiff[i](o)
            Dg = self.vecfundiff[i](o)
            pt_i = self.ptfun[i](o)
            vec_i = self.vecfun[i](o)
            for j in range(len(Df)):
                grad_app = [self.DomainField.element(app_der_kernel(pt_i[j], u)) for u in range(self.dim)]
                app = self.DomainField.element(app_kernel(pt_i[j]))
                for k in range(self.dimGD):
                    appli = sum([grad_app[u] * Df[j][u][k] for u in range(self.dim)])
                    vect_field_list[k] +=  [h[i] * vec_i[j][u] * appli.copy() for u in range(self.dim) ]
                    vect_field_list[k] +=[h[i] * Dg[j][u][k] * app for u in range(self.dim) ]
        adjoint = []    
        for k in range(self.dimGD):
            adjoint.append(vect_field.inner(vect_field_list[k]))

        return self.get_spaceelement_gd(adjoint)
        


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
        grad_inter = np.array([[gu.interpolation(np.transpose(GD)) for gu in g] for g in grad])
        # adjoint[j][k] is sum_i grad_inter[i][j][k] * Mom[i][k] :
        # it is Dv(GD[k])^T Mom
        adjoint = sum([np.array(momi) * gi for momi, gi in zip(np.transpose(Mom), grad_inter) ])
        
        return self.GDspace.element(np.transpose(adjoint))

    def AdjointDerApplyMod(self, GD, Mom, Mod, GDMod, ContMod):
        #before : grad_inter[i][j][k] is j-th gradient of vect_field[i] applied to GD[k] :
        # it is (Dv(GD[k]))_{i, j}
        u1 = np.ones(self.NbGD)
        u2 = np.zeros(self.NbGD)
        #grad_inter[i][j][k] is i-th gradient of vect_field[k] applied to GD[j] :
        # it is (Dv(GD[j]))_{k, i} with vect_field generated by Mod with GDMod, ContMod
        grad_inter = []
        grad_inter.append(Mod.ComputeFieldEvaluateDer(GDMod, ContMod, GD, np.transpose(np.array([u1, u2]))))
        grad_inter.append(Mod.ComputeFieldEvaluateDer(GDMod, ContMod, GD, np.transpose(np.array([u2, u1]))))
        
        grad_inter = np.transpose(grad_inter, (2, 0, 1))
        
        # adjoint[j][k] is sum_i grad_inter[i][j][k] * Mom[i][k] :
        # it is Dv(GD[k])^T Mom
        adjoint = sum([np.array(momi) * gi for momi, gi in zip(np.transpose(Mom), grad_inter) ])
        
        return self.GDspace.element(np.transpose(adjoint))

        
        
    def Cost(self,GD,Cont):
        
        return float(Cont.norm() **2)


#    @property
    def DerCost(self, GD, Cont):
        
        return self.GDspace.zero()
        
#    @property
    def InvCo(self, GD):

        return np.identity(self.dimCont)

##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:36:18 2018

@author: bgris
"""

