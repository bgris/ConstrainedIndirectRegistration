#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test with SheppLogan, with shearing and compressing motions
Forward operator: Restrincting operator

"""

name_path = 'gris'
import numpy as np




name_init = '/home/' + name_path
path_results = name_init + '/Results/DeformationModule/PartialObs/'
name_exp_init =  'ComprShear'
name_file_output_init = path_results + 'output' + name_exp_init

import sys
sys.path.insert(0, "/home/" + name_path + "/miniconda3/envs/odl/lib/python3.5/site-packages/")
sys.path.insert(0,"/home/" + name_path + "/git_repo/odl")
sys.path.insert(0, "/home/" + name_path + "/git_repo/ConstrainedIndirectRegistration")

import odl

##%% Create data from lddmm registration
import matplotlib.pyplot as plt

import Modules.DeformationModuleAbstract
from Modules.DeformationModuleAbstract import Compound
#print(sys.path)
import Modules.Shooting_DefModbis as shoot
import Modules.SumTranslations as SumTranslations
import Modules.TranslationBasedbis as TranslationBased
import Modules.Silent as Silent
import copy




from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
import scipy


#plt.ioff()
#%%


def snr_fun(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')
#
#%%


## Load data
  
space_init = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')
template_init = odl.phantom.shepp_logan(space_init, modified=True)

padded_op = ResizingOperator(
        space_init, ran_shp=[int(2* template_init.shape[0]) for _ in range(space_init.ndim)])
template = padded_op(template_init)
space = template.space


path_data = 'data/'
name_target = 'ComprShear'
target = np.loadtxt(path_data + name_target)

ground_truth = space.element(target).copy()

padded_op = ResizingOperator(
        space_init, ran_shp=[int(1.2* template_init.shape[0]) for _ in range(space_init.ndim)])

template = padded_op(template_init)
space = template.space

ground_truth = space.element(ground_truth.interpolation(space.points().T).reshape(space.shape))
template = space.element(np.loadtxt(path_data + 'Template'))



### Define Deformation Module

##%% Compressing Module

fac=0.5
sigma = 8.0
def kernel(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma ** 2))
#

def partialder2kernel(x, y, u):
   return (-2 / (sigma ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma ** 2)) * sum([ (yi - xi) * ui for xi, yi, ui in zip(x, y, u)]) 


def partialderdim1kernel(x, y, d):
   return (-2 / (sigma ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma ** 2)) * (x[d] - y[d]) 


GDspace = odl.ProductSpace(odl.rn(2), 2)
dimGD = 4

def pointfunction(o):
    return np.array(o).copy()

def pointfunctiondiff(o):
    mat0 = np.zeros((2,4))
    mat1 = np.zeros((2,4))
    mat0[0,0]=1
    mat0[1,1]=1
    mat1[0,2]=1
    mat1[1,3]=1
    return [mat0, mat1]

def vectorfunction(o):
    return [o[1] - o[0], o[0] - o[1]]

def vectorfunctiondiff(o):
    mat0 = np.zeros((2,4))
    mat1 = np.zeros((2,4))
    mat0[0,0]=-1
    mat0[1,1]=-1
    mat0[0,2]=1
    mat0[1,3]=1
    mat1[0,0]=1
    mat1[1,1]=1
    mat1[0,2]=-1
    mat1[1,3]=-1
    return [mat0, mat1]

weightCompr = 1
pointfunctionlist = [pointfunction]
pointfunctiondifflist = [pointfunctiondiff]
vectorfunctionlist = [vectorfunction]
vectorfunctiondifflist = [vectorfunctiondiff]
Compr = TranslationBased.TranslationBased(space, GDspace, 2, kernel, partialderdim1kernel, partialder2kernel, pointfunctionlist, pointfunctiondifflist, vectorfunctionlist, vectorfunctiondifflist)

##%% Shearing Module  

fac=0.5
sigmaShear = 8.0
def kernelShear(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaShear ** 2))

def partialder2kernelShear(x, y, u):
   return (-2 / (sigmaShear ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaShear ** 2)) * sum([ (yi - xi) * ui for xi, yi, ui in zip(x, y, u)]) 


def partialderdim1kernelShear(x, y, d):
   return (-2 / (sigmaShear ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaShear ** 2)) * (x[d] - y[d]) 


GDspaceShear = odl.ProductSpace(odl.rn(2), 2)
dimGDShear = 4


def pointfunctionShear(o):
    return np.array(o).copy()

def pointfunctiondiffShear(o):
    mat0 = np.zeros((2,4))
    mat1 = np.zeros((2,4))
    mat0[0,0]=1
    mat0[1,1]=1
    mat1[0,2]=1
    mat1[1,3]=1
    return [mat0, mat1]

def vectorfunctionShear(o):
    u = o[1] - o[0]
    return [np.array([u[1], -u[0]]), np.array([-u[1], u[0]])]

def vectorfunctiondiffShear(o):
    mat0 = np.zeros((2,4))
    mat1 = np.zeros((2,4))
    mat0[0,0]= 0    
    mat0[0,1]= -1    
    mat0[0,2]= 0    
    mat0[0,3]= 1
    mat0[1,0]= 1    
    mat0[1,1]= 0    
    mat0[1,2]= -1    
    mat0[1,3]= 0   

    mat1[0,0]= 0    
    mat1[0,1]= 1    
    mat1[0,2]= 0    
    mat1[0,3]= -1
    mat1[1,0]= -1    
    mat1[1,1]= 0    
    mat1[1,2]= 1    
    mat1[1,3]= 0   

    return [mat0, mat1]


pointfunctionlistShear = [pointfunctionShear]
pointfunctiondifflistShear = [pointfunctiondiffShear]
vectorfunctionlistShear = [vectorfunctionShear]
vectorfunctiondifflistShear = [vectorfunctiondiffShear]
Shear = TranslationBased.TranslationBased(space, GDspaceShear, 2, kernelShear, partialderdim1kernelShear, partialder2kernelShear, pointfunctionlistShear, pointfunctiondifflistShear, vectorfunctionlistShear, vectorfunctiondifflistShear)



GD_Compr = Compr.GDspace.element([[-10, 0], [10,0]])
GD_Shear = Shear.GDspace.element([[0, -10], [0, 10]])


N=20
silent = Silent.Silent(space, 1.0/N)


GD_Silent = silent.GDspace.element(template)
Cont_Silent = silent.Contspace.zero()



Module=Compound([silent, Shear, Compr])
GD_init = Module.GDspace.element([GD_Silent, GD_Shear, GD_Compr])




#%% Define forward operator

limx1 = -5
limx0 = 5
limy1 = -5
limy0 = 5
cache = space.zero()
points = space.points().T
pts0 = points[0].reshape(space.shape)
pts1 = points[1].reshape(space.shape)
cache = space.element(pts0<limx0)*space.element(pts0>limx1)*space.element(pts1<limy0)*space.element(pts1>limy1)
forward_op =  odl.operator.default_ops.MultiplyOperator(cache)

fac_noise = 0.25

### As the forward operator is a restriction one, the noise is applied to the template space
noise =forward_op(fac_noise * odl.phantom.noise.white_noise(forward_op.domain))
# Create projection data by calling the op on the phantom
data_noisless = forward_op(ground_truth) 
#data = data_noisless + noise
snr = snr_fun(data_noisless, noise, 'dB')
#print(snr)

data = forward_op(ground_truth) + noise

name_data = path_results + name_exp_init  + 'data' 
np.savetxt(name_data, data)

##%% Define functional

import copy
lamb = 1e-6
norm = odl.solvers.L2NormSquared(forward_op.range)

functional_match = shoot.MatchingModule(N, lamb, Module, data, forward_op, norm, 0.0001*space.cell_sides[0])


projXGD = odl.operator.pspace_ops.ComponentProjection(functional_match.domain, 0)
projXMOM = odl.operator.pspace_ops.ComponentProjection(functional_match.domain, 1)

proj_op0 = odl.operator.pspace_ops.ComponentProjection(Module.GDspace, 0)
proj_op1 = odl.operator.pspace_ops.ComponentProjection(Module.GDspace, 1)
proj_op2 = odl.operator.pspace_ops.ComponentProjection(Module.GDspace, 2)

proj_opPoints0 = odl.operator.pspace_ops.ComponentProjection(Shear.GDspace, 0)
proj_opPoints1 = odl.operator.pspace_ops.ComponentProjection(Shear.GDspace, 1)

normrn4 = odl.solvers.L2NormSquared(Shear.GDspace)
normrn2 = odl.solvers.L2NormSquared(odl.rn(2))
normrn1 = odl.solvers.L2NormSquared(normrn2.range)
cst = odl.solvers.functional.default_functionals.ConstantFunctional(proj_op1.domain, normrn1.range.element(1))

frac10 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints0*proj_op1-25))
frac11 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints1*proj_op1-25))
frac20 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints0*proj_op2-25))
frac21 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints1*proj_op2-25))


#
norm_GD = frac10 + frac11 + frac20 + frac21  

norm_MOM = normrn4*proj_op1 + normrn4*proj_op2

fac_noise_list = [0.5, 0.05, 1., 0.25, 0.]
fac_noise_list_str = ['0_5', '0_05', '1', '0_25',' 0']

gamma = 1e-5
tau = 1e-5
name_exp_init += '__limx0_' + str(limx0)  + '__limx1_' + str(limx1)  + '__limy0_' + str(limy0)  + '__limy1_' + str(limy1)
name_exp_init += '_gamma__1e_5__tau__1e_5' 

indexnoise = 0

fac_noise = fac_noise_list[indexnoise]
fac_noise_str = fac_noise_list_str[indexnoise] 

### As the forward operator is a restriction one, the noise is applied to the template space
noise =forward_op(fac_noise * odl.phantom.noise.white_noise(forward_op.domain))
# Create projection data by calling the op on the phantom
data_noisless = forward_op(ground_truth) 
#data = data_noisless + noise
snr = snr_fun(data_noisless, noise, 'dB')
print(snr)

data = forward_op(ground_truth) + noise
#data = forward_op.range.element(np.loadtxt(path_data + name_target + '_data'))
name_data = path_results + name_exp_init  + 'data__fac_noise_' + fac_noise_str 
np.savetxt(name_data, data)

functional_match = shoot.MatchingModule(N, lamb, Module, data, forward_op, norm, 0.0001*space.cell_sides[0])


functional = functional_match  + gamma * (norm_GD*projXGD) + tau*(norm_MOM*projXMOM)

GD = copy.deepcopy(GD_init)
X = functional.domain.element([GD, Module.GDspace.zero()])
grad_op = functional.gradient
energy = functional(X)
#print(energy)
energy_tmp = energy

name_exp = name_exp_init + '_fac_noise_' + fac_noise_str
name_file_output = name_file_output_init  + '_fac_noise_'  + fac_noise_str
niter = 300
stepGD =  5e-3
stepMom = 5e-7


#print(energy)
file = open(name_file_output, 'w')
file.write(str(energy) + '\n')
file.close()



##%% Gradient descent
uu= 0
ct=0
for i in range(niter):
    if (uu == 0):
        grad = grad_op(X)
    if (np.mod(i, 5) ==0):
        for k in range(len(X[0])):
            name = path_results + name_exp  + 'GD_Mod_' + str(k)
            np.savetxt(name, X[0][k])
            name = path_results + name_exp + 'MOM_Mod_' + str(k)
            np.savetxt(name, X[1][k])
    X_tmp0 = copy.deepcopy(X)
    X_tmp0[0][1:] = X[0][1:] - stepGD * grad[0][1:]
    X_tmp0[1][1:] = X[1][1:] - stepMom * grad[1][1:]
    try:
        energy_tmp0 = functional(X_tmp0)
        X_tmp1 = copy.deepcopy(X)
        X_tmp1[0][1] = X[0][1] - 0.5*stepGD * grad[0][1]
        X_tmp1[1][1] = X[1][1] - stepMom * grad[1][1]
        energy_tmp1 = functional(X_tmp1)
        X_tmp2 = copy.deepcopy(X)
        X_tmp2[0][1] = X[0][1] - stepGD * grad[0][1]
        X_tmp2[1][1] = X[1][1] - 0.5*stepMom * grad[1][1]
        energy_tmp2 = functional(X_tmp2)
        if (energy_tmp0 <= energy_tmp1 and energy_tmp0 <= energy_tmp2):
            energy_tmp = energy_tmp0
            X_tmp = copy.deepcopy(X_tmp0)
        elif (energy_tmp1 <= energy_tmp0 and energy_tmp1 <= energy_tmp2):
                energy_tmp = energy_tmp1
                X_tmp = copy.deepcopy(X_tmp1)
                stepGD = 0.5 * stepGD
        else:
                energy_tmp = energy_tmp2
                X_tmp = copy.deepcopy(X_tmp2)
                stepMom = 0.5*stepMom
        
        if (energy_tmp <= energy):
            energy = energy_tmp
            X = copy.deepcopy(X_tmp)
            stepGD *= 1.2
            stepMom *= 1.2
            uu = 0
            file = open(name_file_output, 'a')
            file.write('iter =  ' + str(i) + ' energy = ' + str(energy) + '\n')
            file.close()
        else:
            uu = 1
            stepGD *= 0.8
            stepMom *= 0.8
            ct += 1
            if ct == 5:
                if functional_match.eps > 1e-13:
                    functional_match.eps = 0.1 * functional_match.eps
                else:
                    functional_match.eps = 0.0001*space.cell_sides[0]
                grad_op = functional.gradient
                stepGD *= 5
                stepMom *= 5
                ct = 0
                uu = 0
            
            file = open(name_file_output, 'a')
            file.write('iter =  '  + str(i)  + 'stepGD = ' + str(stepGD) + 'stepMom' + str(stepMom)+ ' energy = ' + str(energy) + '\n')
            file.close()
    except: 
            uu = 1
            stepGD *= 0.5
            stepMom *= 0.5
            ct += 1
            if ct == 5:
                if functional.eps > 1e-13:
                    functional.eps = 0.1 * functional.eps
                else:
                    functional.eps = 0.0001*space.cell_sides[0]
            grad_op = functional.gradient
            ct = 0
            uu = 0
            
            file = open(name_file_output, 'a')
            file.write('iter =  '  + str(i)  + 'stepGD = ' + str(stepGD) + 'stepMom' + str(stepMom) + '\n')
            file.close()

         

### Save results


for i in range(len(X[0])):
    name = path_results + name_exp  + 'GD_Mod_' + str(i)
    np.savetxt(name, X[0][i])
    name = path_results + name_exp + 'MOM_Mod_' + str(i)
    np.savetxt(name, X[1][i])
#          
    

shoot_op = shoot.Shooting(N, Module)
GD_list, Mom_list, Cont_list, field_list = shoot_op._call(X)


import os
path_result_tot = path_results + name_exp + '/'

for j in range(N+1):
    for i in range(len(X[0])):
        name = path_result_tot  + 'GD_Mod_' + str(i) + '__t_' + str(j)
        np.savetxt(name, GD_list[j][i])
#
np.savetxt(path_result_tot + 'template', template)
np.savetxt(path_result_tot + 'ground_truth', ground_truth)
np.savetxt(path_result_tot + 'data', data)

### save image at time t


def restr_op(f):
    return space_init.element(f.interpolation(space_init.points().T))


name_fig = path_result_tot + 'image__t_' + str(t)

fig = restr_op(GD_list[t][0]).show(clim=[mini, maxi])
plt.axis('equal')
plt.axis('off')
fig.delaxes(fig.axes[1])

plt.savefig(path_figure + name_fig + str(t) + '.pdf' )
plt.close('all')

        
