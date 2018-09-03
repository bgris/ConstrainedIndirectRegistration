#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test with SheppLogan, rotation of small structure and 2 other translations
Forward operator: Ray Transform
"""


##%%

name_path = 'gris'
import numpy as np


##%% Give the parameter of forward operator
num_angles = 10
miniangle = '0_3pi'
min_angle = 0.3 * np.pi
maxiangle = '0_7pi'
max_angle = 0.7 * np.pi




name_init = '/home/' + name_path
path_results = name_init + '/Results/DeformationModule/RayTransfo/'
name_exp =  '__minanglangle_' + miniangle+ '__maxangle_' + maxiangle + '__angles_' + str(num_angles) 

name_file_output = path_results + 'output' + name_exp

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
import Modules.Shooting_DefMod as shoot
import Modules.TranslationBased as TranslationBased
import Modules.Silent as Silent
import copy




from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
import scipy




#%%

## Load data
  
space_init = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')
fac_smooth = 0.8
template_init = odl.phantom.shepp_logan(space_init, modified=True)

padded_op = ResizingOperator(
        space_init, ran_shp=[int(2* template_init.shape[0]) for _ in range(space_init.ndim)])
template = padded_op(template_init)
space = template.space


path_data = 'data/'
name_target = 'Trans2Rot'
target = np.loadtxt(path_data + name_target)

ground_truth = space.element(target).copy()

padded_op = ResizingOperator(
        space_init, ran_shp=[int(1.2* template_init.shape[0]) for _ in range(space_init.ndim)])

template = padded_op(template_init)
space = template.space

ground_truth = space.element(ground_truth.interpolation(space.points().T).reshape(space.shape))
template = space.element(np.loadtxt(path_data + 'Template'))



### Define Deformation Module

##%% First Translation
sigma = 2.0
def kernel(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma ** 2))
#

def partialder2kernel(x, y, u):
   return (-2 / (sigma ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma ** 2)) * sum([ (yi - xi) * ui for xi, yi, ui in zip(x, y, u)]) 


def partialderdim1kernel(x, y, d):
   return (-2 / (sigma ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma ** 2)) * (x[d] - y[d]) 


Trans = SumTranslations.SumTranslations(space, 1, kernel, partialderdim1kernel, partialder2kernel)


GD_Trans = Trans.GDspace.element([[2.5, 7.5]])
Cont_Trans = Trans.Contspace.element([[1., 1.]])


##%% Second Translation
sigmaTrans2 = 4.0
def kernelTrans2(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaTrans2 ** 2))
#

def partialder2kernelTrans2(x, y, u):
   return (-2 / (sigmaTrans2 ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaTrans2 ** 2)) * sum([ (yi - xi) * ui for xi, yi, ui in zip(x, y, u)]) 


def partialderdim1kernelTrans2(x, y, d):
   return (-2 / (sigma ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaTrans2 ** 2)) * (x[d] - y[d]) 


Trans2 = SumTranslations.SumTranslations(space, 1, kernelTrans2, partialderdim1kernelTrans2, partialder2kernelTrans2)


GD_Trans2 = Trans.GDspace.element([[-5.5, -1.]])
Cont_Trans2 = Trans.Contspace.element([[1., 1.]])



##%% Rotation
sigmaRot = 0.5

def kernelRot(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaRot ** 2))
#

def partialder2kernelRot(x, y, u):
   return (-2 / (sigmaRot ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaRot ** 2)) * sum([ (yi - xi) * ui for xi, yi, ui in zip(x, y, u)]) 


def partialderdim1kernelRot(x, y, d):
   return (-2 / (sigmaRot ** 2)) * np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigmaRot ** 2)) * (x[d] - y[d]) 


GDspace = odl.ProductSpace(odl.rn(2), 2)
dimGD = 4

#nb_inter = 5

nb_dir = 10
nb_orth = 4
factor = 0.25 * sigmaRot

liste_index = []
for i in range(nb_dir+1):
    for j in range(nb_orth+1):
        liste_index.append([i , j ])

liste_index_der = []
for i in range(nb_dir+1):
    for j in range(nb_orth+1):
        liste_index_der.append([i-0.5 * nb_dir, j- 0.5 * nb_orth] )



def pointfunction(o):
    centre = 0.5*(np.array(o[0]) + np.array(o[1]))
    direc = factor*(np.array(o[0]) - np.array(o[1]))
    direc_orth =  np.array([direc[1], -direc[0]])
    point_init = centre - 0.5 * nb_dir * direc - 0.5 * nb_orth * direc_orth
    mesh = [point_init + i[0]* direc + i[1] *direc_orth for i in liste_index]
    return np.array(mesh).copy()

centre_diff = np.zeros((2,4))
centre_diff[0][0] = 0.5
centre_diff[0][2] = 0.5
centre_diff[1][1] = 0.5
centre_diff[1][3] = 0.5

direc_diff = np.zeros((2,4))
direc_diff[0][0] = factor
direc_diff[0][2] = -factor
direc_diff[1][1] = factor
direc_diff[1][3] = -factor

direc_orth_diff = np.zeros((2,4))
direc_orth_diff[0][1] = factor
direc_orth_diff[0][3] = -factor
direc_orth_diff[1][0] = -factor
direc_orth_diff[1][2] = factor

point_init_diff = centre_diff - 0.5 * nb_dir * direc_diff - 0.5 * nb_orth * direc_orth_diff

mesh_diffpointfunctiondiff = []
for i in range(nb_dir+1):
    for j in range(nb_orth+1):
        mesh_diffpointfunctiondiff.append(point_init_diff + i * direc_diff + j *direc_orth_diff )


def pointfunctiondiff(o):
    return copy.deepcopy(mesh_diffpointfunctiondiff)


def vectorfunction(o):
    direc = factor*(np.array(o[0]) - np.array(o[1]))
    direc_vec = np.array([direc[1], -direc[0]])
    direc_orth =  np.array([direc[1], -direc[0]])
    direc_orth_vec = np.array([direc_orth[1], -direc_orth[0]])
    mesh = [i[0] * direc_vec + i[1] * direc_orth_vec for i in liste_index_der]
    return copy.deepcopy(mesh)


direc_vec_diff = np.zeros((2,4))
direc_vec_diff[0][1] = factor
direc_vec_diff[0][3] = -factor
direc_vec_diff[1][0] = -factor
direc_vec_diff[1][2] = factor

direc_orth_vec_diff = np.zeros((2,4))
direc_orth_vec_diff[0][0] = -factor
direc_orth_vec_diff[0][2] = factor  
direc_orth_vec_diff[1][1] = -factor
direc_orth_vec_diff[1][3] = factor

mesh_diffvectorfunctiondiff = []
for i in range(nb_dir+1):
    for j in range(nb_orth+1):
        mesh_diffvectorfunctiondiff.append((i-0.5 * nb_dir) * direc_vec_diff + (j- 0.5 * nb_orth) *direc_orth_vec_diff )


def vectorfunctiondiff(o):
    return copy.deepcopy(mesh_diffvectorfunctiondiff)


pointfunctionlist = [pointfunction]
pointfunctiondifflist = [pointfunctiondiff]
vectorfunctionlist = [vectorfunction]
vectorfunctiondifflist = [vectorfunctiondiff]
Rot = TranslationBased.TranslationBased(space, GDspace, 2, kernelRot, partialderdim1kernelRot, partialder2kernelRot, pointfunctionlist, pointfunctiondifflist, vectorfunctionlist, vectorfunctiondifflist)

GD_Rot = Rot.GDspace.element([[-2.2,-9.8], [1.8,-9.8]])
Cont_Rot = Rot.Contspace.element(-1)

##%%
N=10
silent = Silent.Silent(space, 1.0/N)

GD_Silent = silent.GDspace.element(template)
Cont_Silent = silent.Contspace.zero()


Module=Compound([silent, Trans, Trans2, Rot])
GD_init = Module.GDspace.element([GD_Silent, GD_Trans, GD_Trans2, GD_Rot])



### Create forward operator

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                    nodes_on_bdry=[(True, True)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-32, 32, int(round(space.shape[0]*np.sqrt(2))))

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)


## Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')

# Create projection data by calling the op on the phantom
data_noiseless = forward_op(ground_truth)

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
noise_level = 0.
noise_level_str = '0'

data = data_noiseless.copy()

#%%
lamb = 1e-6
norm = odl.solvers.L2NormSquared(forward_op.range)

functional_match = shoot.MatchingModule(N, lamb, Module, data, forward_op, norm, 0.0001*space.cell_sides[0])

projXGD = odl.operator.pspace_ops.ComponentProjection(functional_match.domain, 0)
projXMOM = odl.operator.pspace_ops.ComponentProjection(functional_match.domain, 1)

proj_op0 = odl.operator.pspace_ops.ComponentProjection(Module.GDspace, 0)
proj_op1 = odl.operator.pspace_ops.ComponentProjection(Module.GDspace, 1)
proj_op2 = odl.operator.pspace_ops.ComponentProjection(Module.GDspace, 2)
proj_op3 = odl.operator.pspace_ops.ComponentProjection(Module.GDspace, 3)

proj_opPoints0 = odl.operator.pspace_ops.ComponentProjection(Rot.GDspace, 0)
proj_opPoints1 = odl.operator.pspace_ops.ComponentProjection(Rot.GDspace, 1)
proj_opPoints00 = odl.operator.pspace_ops.ComponentProjection(Trans.GDspace, 0)

normrn4 = odl.solvers.L2NormSquared(Rot.GDspace)
normrn20 = odl.solvers.L2NormSquared(Trans.GDspace)
normrn2 = odl.solvers.L2NormSquared(odl.rn(2))
normrn1 = odl.solvers.L2NormSquared(normrn2.range)
cst = odl.solvers.functional.default_functionals.ConstantFunctional(proj_op1.domain, normrn1.range.element(1))

frac30 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints0*proj_op3-25))
frac31 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints1*proj_op3-25))
frac20 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints00*proj_op2-25))
frac10 = odl.solvers.functional.functional.FunctionalQuotient(cst, (normrn2*proj_opPoints00*proj_op1-25))


#
norm_GD = frac10 + frac20 + frac30 + frac31 

norm_MOM = normrn20*proj_op1 + normrn20*proj_op2 + normrn4*proj_op3

gamma = 1e-5
tau = 1e-5

functional = functional_match  + gamma * (norm_GD*projXGD) + tau*(norm_MOM*projXMOM)


##%% Gradient descent


GD = copy.deepcopy(GD_init)

X = functional.domain.element([GD, Module.GDspace.zero()])
grad_op = functional.gradient
energy = functional(X)

niter = 300
stepGD = 5e-8
stepMom = 5e-8


name_exp += 'sigmaRot_' + str(sigmaRot) + '__nb_dir_' + str(nb_dir) + '__nb_orth_' + str(nb_orth) + '__noise_level_' + noise_level_str 


file = open(name_file_output, 'w')
file.write(str(energy) + '\n')
file.close()

uu= 0
ct=0
for i in range(niter):
    if (uu == 0):
        grad = grad_op(X)
#
    if (np.mod(i,5)==0):
       for i in range(len(X[0])):
          name = path_results + name_exp  + 'GD_Mod_' + str(i)
          np.savetxt(name, X[0][i])
          name = path_results + name_exp + 'MOM_Mod_' + str(i)
          np.savetxt(name, X[1][i])

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
            #print('iter =  {}  , energy = {}'.format(i, energy))
            
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
                if functional_match.eps > 1e-13:
                    functional_match.eps = 0.1 * functional_match.eps
                else:
                    functional_match.eps = 0.0001*space.cell_sides[0]
                
                ct = 0
                uu = 0
            grad_op = functional.gradient
            
            file = open(name_file_output, 'a')
            file.write('iter =  '  + str(i)  + 'stepGD = ' + str(stepGD) + 'stepMom' + str(stepMom) + '\n')
            file.close()

### save results
for i in range(len(X[0])):
    name = path_results + name_exp  + 'GD_Mod_' + str(i)
    np.savetxt(name, X[0][i])
    name = path_results + name_exp + 'MOM_Mod_' + str(i)
    np.savetxt(name, X[1][i])

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



