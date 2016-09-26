# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:54:41 2016

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.stats import poisson

from Tsallis import TROT
from Generators import euc_costs  



#Grid visualization of the impact of parameters q, lambda on TROT's solution
#arrays q,l contain the values we wish to test
#arrays mu1, mu2 contain the modes of marginals r,c (between 0 and n)
#arrays t1, t2 are coefficients for mu1, mu2
def TROT_grid(q,l,mu1,mu2,t1,t2,n,output):


    #Create marginals r,c
    
    x = range(n)
    
    r_tmp = []    
    for mode in mu1:
        r_tmp.append(poisson.pmf(x,mode))
        
    c_tmp = []    
    for mode in mu2:
        c_tmp.append(poisson.pmf(x,mode))
        
    r = np.dot(t1,r_tmp)
    r = r/r.sum()
    
    c = np.dot(t2,c_tmp)
    c = c/c.sum()
    
    
    M = euc_costs(n,n)
    
    P = []
    
    nq = len(q)
    nl = len(l)
    
        
    for j in range(nq):
         for i in range(nl):
            P_tmp = TROT(q[j],M,r,c,l[i],1E-2)
            P.append(P_tmp)
    
    
    fig = plt.figure(figsize=(8, 8))
    
    outer_grid = gridspec.GridSpec(2, 2, width_ratios=[1,5], height_ratios=[1,5])
    outer_grid.update(wspace=0.01, hspace=0.01)
    # gridspec inside gridspec
    outer_joint = gridspec.GridSpecFromSubplotSpec(nq,nl, subplot_spec=outer_grid[1,1],wspace=0.02, hspace=0.02)
    outer_row_marg = gridspec.GridSpecFromSubplotSpec(nq,1, subplot_spec=outer_grid[1,0],wspace=0.02, hspace=0.02)
    outer_col_marg = gridspec.GridSpecFromSubplotSpec(1,nl, subplot_spec=outer_grid[0,1],wspace=0.02, hspace=0.02)
    
    
    for b in range(nl):
        for a in range (nq):
            ax = plt.Subplot(fig, outer_joint[a,b])
            ax.imshow(P[nl*a + b], origin='upper', interpolation = None, aspect = 'auto', cmap = 'Greys')
            rect = Rectangle((0, 0), n-1, n-1, fc='none', ec='black')     
            rect.set_width(0.8)
            rect.set_bounds(0,0,n-1,n-1)
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            ax.set_axis_bgcolor('white')
        
    for i in range(nq):
        ax_row = plt.Subplot(fig,outer_row_marg[i], sharey = ax)
        ax_row.plot(1-r, x)
        fig.add_subplot(ax_row)
    
        ax_row.axes.get_xaxis().set_visible(False)
        ax_row.axes.get_yaxis().set_visible(False)
        bottom, height = .25, .5
        top = bottom + height
        ax_row.text(-0.05, 0.5*(bottom+top), 'q = {0}'.format(q[i]), horizontalalignment='right', verticalalignment='center', rotation='vertical',transform=ax_row.transAxes, fontsize='medium')
    
        ax_row.set_axis_bgcolor('white')
    
    for j in range(nl):
        ax_col = plt.Subplot(fig,outer_col_marg[j], sharex = ax)
        ax_col.plot(x,c)
        fig.add_subplot(ax_col)    
        bottom, height = .25, .5
        ax_col.axes.get_xaxis().set_visible(False)
        ax_col.axes.get_yaxis().set_visible(False)
        ax_col.set_title(r'$\lambda$'+' = {0}'.format(l[j]),fontsize='medium')
        ax_col.set_axis_bgcolor('white')
        
    plt.show()
    plt.savefig('{0}.pdf'.format(output))