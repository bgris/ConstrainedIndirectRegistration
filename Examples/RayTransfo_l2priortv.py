# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:24:34 2018

@author: gris
"""

# -*- coding: utf-8 -*-
"""

"""

import odl
import numpy as np
from matplotlib import pylab as plt
import os


import copy

from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
import scipy


##%%

##%% Give the parameter of forward operator
num_angles = 10
miniangle = '0_3pi'
min_angle = 0.3 * np.pi
maxiangle = '0_7pi'
max_angle = 0.7 * np.pi

path_init = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/'
path_init += 'Results/DeformationModule/RayTransfo/'



name_exp =  '__minanglangle_' + miniangle+ '__maxangle_' + maxiangle + '__angles_' + str(num_angles) 
sigmaRot = 0.5
nb_dir = 10
nb_orth = 4
noise_level_str = '0'
name_exp += 'sigmaRot_' + str(sigmaRot) + '__nb_dir_' + str(nb_dir) + '__nb_orth_' + str(nb_orth) + '__noise_level_' + noise_level_str 

path_load_result = path_init + name_exp + '/'


## Template  
space_init = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')
fac_smooth = 0.8
template_init = odl.phantom.shepp_logan(space_init, modified=True)



padded_op = ResizingOperator(
        space_init, ran_shp=[int(1.2* template_init.shape[0]) for _ in range(space_init.ndim)])

template = padded_op(template_init)
space = template.space

template = space.element(np.loadtxt(path_load_result + 'template'))
ground_truth = space.element(np.loadtxt(path_load_result + 'ground_truth'))

angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                    nodes_on_bdry=[(True, True)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-32, 32, int(round(space.shape[0]*np.sqrt(2))))

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)


## Ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')

data = ray_trafo(ground_truth)
data = ray_trafo.range.element(np.loadtxt(path_load_result + 'data'))

gradient = odl.Gradient(space)

# Functional to enforce 0 <= x <= 1
f = odl.solvers.IndicatorBox(space, -1.5, 1.5)

# --- Create functionals for solving the optimization problem ---
#%% TV
# Gradient for TV regularization
gradient = odl.Gradient(space)

# Functional to enforce 0 <= x <= 1
f = odl.solvers.IndicatorBox(space, -1.6, 1.6)

lamlist = [0.01, 0.1, 1]
lamlist_str = ['0_01', '0_1', '1']
l2_reg_paramlist = [0.01, 0.1, 1]
l2_reg_paramlist_str = ['0_01', '0_1', '1']

for i in range(3):
    for j in range(3):
        lam = lamlist[i]
        lam_str = lamlist_str[i]
        l2_reg_param = l2_reg_paramlist[j]
        l2_reg_param_str = l2_reg_paramlist_str[j]
        
        indicator_zero = odl.solvers.IndicatorZero(ray_trafo.range)
        indicator_data = indicator_zero.translated(data)
        
        data_func = odl.solvers.IndicatorLpUnitBall(data.space, 2).translated(data) 


        cross_norm = lam * odl.solvers.GroupL1Norm(gradient.range)
        l2_reg_func_l2_and_tv = l2_reg_param * odl.solvers.L2NormSquared(
                                            space).translated(template)
        # --- Create functionals for solving the optimization problem ---
        
        # Assemble operators and functionals for the solver
        lin_ops = [ray_trafo, odl.IdentityOperator(space), gradient]
        #g = [indicator_data, l2_reg_func_l2_and_tv, cross_norm]
        g = [data_func, l2_reg_func_l2_and_tv, cross_norm]
        
        # Create callback that prints the iteration number and shows partial results
    #    callback = (odl.solvers.CallbackShow('iterates', step=5, clim=[-0.3, 1]) &
    #                odl.solvers.CallbackPrintIteration())
        callback = (odl.solvers.CallbackPrintIteration())
        
        # Solve with initial guess x = 0.
        # Step size parameters are selected to ensure convergence.
        # See douglas_rachford_pd doc for more information.
        x = ray_trafo.domain.zero()
        odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                        tau=0.1, sigma=[0.1, 0.1, 0.02], lam=1.5,
                                        niter=100, callback=callback)
        
        name = 'reco_l2TV_datafunc__lam_' + lam_str + '__l2reg_' + l2_reg_param_str
        np.savetxt(path_load_result + name , x)
        
        fig = x.show(clim=[-1, 1])
        plt.axis('off')
        fig.delaxes(fig.axes[1])
        plt.autoscale(False)
        typefig = 'png'
        plt.savefig(path_load_result + name + '.' + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0, format=typefig)
        plt.close('all')



