
# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object
from builtins import super


import numpy as np

from odl.operator import Operator
import odl
from odl.solvers.functional.functional import Functional
import copy

__all__ = ('Shooting', )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:49:16 2018

@author: bgris
"""

class Shooting(Operator):
    
    
    def __init__(self, nb_time_point_int, Module):
        self.N = nb_time_point_int
        self.Mod = Module
        
        domain = odl.ProductSpace(self.Mod.GDspace, 2)
        ran = odl.ProductSpace(odl.ProductSpace(self.Mod.GDspace, self.N +1),
                                odl.ProductSpace(self.Mod.GDspace, self.N +1),
                                odl.ProductSpace(self.Mod.Contspace, self.N +1),
                                odl.ProductSpace(self.Mod.DomainField.tangent_bundle, self.N +1)
                )
        
        
        super().__init__(domain, ran, linear=False)
        
    
    def ComputeControl(self, GD, Mom):
        mat = self.Mod.InvCo(GD)
        adjointfield = self.Mod.AdjointApply(GD, Mom)
        adjointcont = self.Mod.AdjointComputeField(GD, adjointfield)
        cont_vector = np.dot(mat, self.Mod.get_columnvector_control(adjointcont))
        return self.Mod.get_spaceelement_control(cont_vector)
        
        
    def ComputeMomSpeed(self, GD, Mom, Cont, vectfield):
        # vect field is supposed to be computed via GD and Cont
        Momspeed = 0.5 * self.Mod.DerCost(GD, Mom)
        Momspeed -= self.Mod.AdjointDerApplyMod(GD, Mom, self.Mod, GD, Cont)
        Momspeed -= self.Mod.AdjointDerEvaluate(GD, Cont, self.Mod.AdjointApply(GD, Mom))
        
        return Momspeed.copy()
        
    
    
    def _call(self, X):
        GD_init = self.Mod.GDspace.element(X[0].copy())
        Mom_init = self.Mod.GDspace.element(X[1].copy())
        
        GD_list = []
        Mom_list = []
        Cont_list = []
        field_list = []
        
        GD_list.append(GD_init.copy())
        Mom_list.append(Mom_init.copy())
        Cont_list.append(self.ComputeControl(GD_init, Mom_init))
        field_list.append(self.Mod.ComputeField(GD_init, Cont_list[0]))
        
        for i in range(self.N):
            GD_speed = self.Mod.ApplyField(GD_list[i], field_list[i])
            Mom_speed = self.ComputeMomSpeed(GD_list[i], Mom_list[i], Cont_list[i], field_list[i])
            
            GD_list.append(GD_list[i] + (1.0/self.N) * GD_speed)
            Mom_list.append(Mom_list[i] + (1.0/self.N) * Mom_speed)
            
            Cont_list.append(self.ComputeControl(GD_list[i+1], Mom_list[i+1]))
            field_list.append(self.Mod.ComputeField(GD_list[i+1], Cont_list[i+1]))
            
            
        return self.range.element([GD_list, Mom_list, Cont_list, field_list])
    




class MatchingModule(Functional):
    
    
    def __init__(self, nb_time_point_int, lamb, Module, data, forward_operator, norm, eps):

        """ We suppose that the first module is silent with template as GD"""
        self.N = nb_time_point_int
        self.Mod = Module
        self.Shoot = Shooting(nb_time_point_int, Module)
        self.attach = norm*(forward_operator - data)
        self.lamb = lamb
        # for finite difference
        self.eps = eps
        
        domain = odl.ProductSpace(self.Mod.GDspace, 2)
        
        super().__init__(domain, linear=False)
        

    def _call(self, X):
        GD_list, Mom_list, Cont_list, field_list = self.Shoot._call(X)
        return self.attach(GD_list[-1][0]) + self.lamb * self.Mod.Cost(GD_list[0], Cont_list[0])

        
        
    @property
    def gradient(self):

        ope = self

        class FunctionalGradient(Operator):
            
            def __init__(self):
                """Initialize a new instance."""
                super().__init__(ope.domain, ope.domain,
                                 linear=False)

            def _call(self, X):
                GD_list, Mom_list, Cont_list, field_list = ope.Shoot._call(X)
                
                # we suppose that the first module is silent with template as GD
                adjointGD = ope.Mod.GDspace.zero()
                adjointMom = ope.Mod.GDspace.zero()
                adjointGD[0] = ope.attach.gradient(GD_list[-1][0]).copy()
                eps = ope.eps
                for i in range(ope.N):
                    GD = copy.deepcopy(GD_list[ope.N - i])
                    Mom = copy.deepcopy(Mom_list[ope.N - i])
                    Cont = copy.deepcopy(Cont_list[ope.N - i])
                    field = copy.deepcopy(field_list[ope.N - i]) 
                    epsGD   = eps
                    epsMom   = eps
                    
                    GD_depl = GD + epsMom * adjointMom
                    Mom_depl = Mom - epsGD * adjointGD                     
                    speedGD = ope.Mod.ApplyField(GD, field)
                    speedMom = ope.Shoot.ComputeMomSpeed(GD, Mom, Cont, field)
                    Cont_depl = ope.Shoot.ComputeControl(GD_depl, Mom_depl)
                    field_depl = ope.Mod.ComputeField(GD_depl, Cont_depl)
                    speedGD_depl = ope.Mod.ApplyField(GD_depl, field_depl)
                    speedMom_depl = ope.Shoot.ComputeMomSpeed(GD_depl, Mom_depl, Cont_depl, field_depl)
                    
                    adjointGD +=  (1.0/ope.N) * (speedMom_depl - speedMom) / epsMom
                    adjointMom += - (1.0/ope.N) * (speedGD_depl - speedGD) / epsGD

                Cont = copy.deepcopy(Cont_list[0])
                field = copy.deepcopy(field_list[0])                 
                speedGD = ope.Mod.ApplyField(GD_list[0], field)
                speedMom = ope.Shoot.ComputeMomSpeed(GD_list[0], Mom_list[0], Cont, field)
                return [adjointGD - 2 * ope.lamb * speedMom, adjointMom + 2 * ope.lamb * speedGD]
                    
        return FunctionalGradient()
                
        





























































#
