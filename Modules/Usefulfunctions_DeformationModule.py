#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:10:24 2018

@author: bgris
"""


# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

__all__ = ('Dirac', )

def Dirac(o, u, space):
    """
    o is a list of points (arrays)
    u is a list of vectors
    space is a space
    
    Dirac(x, u, space) is a space.tangent_bundle.element() such that
    for v a vector field, Dirac(x, u, space).inner(v) = \sum_i v(x_i).u_i
    
    """
    
    dis = space.cell_sides[0]
    vol = space.cell_volume
    maxi = space.max_pt
    mini = space.min_pt
    vect_field = space.tangent_bundle.zero()
    for i in range(len(o)):
        # checks that o[i] is in space
        if (min([oii < maxii and minii < oii  for oii, maxii, minii in zip(o[i], maxi, mini)])):
            dirac_temp = space.element(lambda x: (np.sqrt(sum((x-o[i])**2)) < dis) )
            fac = sum(sum(dirac_temp.asarray() == 1)) * vol
            vect_field += [(uu / fac) * dirac_temp for uu in u[i]]
        
    return vect_field.copy()
    
    
def Make_der_CovMat(x, y, u, partialder2kernel):
    # x and y a 'structured points' : size =dim x nb_points
    # u is 'structured vectors '  :size =dim x nb_vectors
    # u and y have the same number of elements
    dim = len(x)
    p1 = np.reshape(x, (dim, 1, -1))
    p2 = np.reshape(y, (dim, -1, 1))
    p3 = np.reshape(u, (dim, -1, 1))
    return partialder2kernel(p1, p2, p3)

def create_structured(points, vectors):
    return np.vstack([points, vectors])



def make_covariance_matrix(points, kernel):
    """ creates the covariance matrix of the kernel for the given points"""

    dim = len(points)
    p1 = np.reshape(points, (dim, 1, -1))
    p2 = np.reshape(points, (dim, -1, 1))

    return kernel(p1, p2)



def make_covariance_mixte_matrix(points1, points2, kernel):
    """ creates the covariance matrix of the kernel for the given points"""

    dim = len(points1)
    p1 = np.reshape(points1, (dim, -1, 1))
    p2 = np.reshape(points2, (dim, 1, -1))
    
    return kernel(p1, p2)

def get_points(structured_field):
    dim_double, nb_points = structured_field.shape
    dim = int(dim_double/2)

    return structured_field[0:dim].copy()


def get_vectors(structured_field):
    dim_double, nb_points = structured_field.shape
    dim = int(dim_double/2)

    return structured_field[dim:2*dim].copy()


def get_from_structured_to_unstructured(space, kernel):
    mg = space.meshgrid
    nb_pts_mg0 = mg[0].shape[0]
    nb_pts_mg1 = mg[1].shape[1]
    mg_reshaped = []
    mg_reshaped.append(mg[0].reshape([nb_pts_mg0,1,1]))
    mg_reshaped.append(mg[1].reshape([1,nb_pts_mg1,1]))

    def from_structured_to_unstructured(structured_field):
        dim_double, nb_points = structured_field.shape
        dim = int(dim_double/2)
        points = get_points(structured_field)
        vectors = get_vectors(structured_field)
        unstructured = space.tangent_bundle.zero()
        pt0 = points[0].reshape(1,1,nb_points)
        pt1 = points[1].reshape(1,1,nb_points)
        points_reshaped = [pt0, pt1]
        vectors_reshaped = np.transpose(vectors.reshape(dim,nb_points,1), (0,2,1))
        kern_discr = kernel(mg_reshaped, points_reshaped)
        #unstructured = space.tangent_bundle.zero()
        unstructured = space.tangent_bundle.element([(vectors_reshaped[u] * kern_discr).sum(2) for u in range(dim)])

        return unstructured

    return from_structured_to_unstructured






#
