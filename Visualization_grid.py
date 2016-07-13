# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:54:41 2016

@author: boris
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from Regularized_OT import  KL_proj_descent
from Tsallis import second_order_sinkhorn
from Generators import rand_marginal, rand_costs, euc_costs  
from matplotlib import gridspec
from scipy.stats import poisson
from matplotlib.patches import Rectangle
from Projections import Sinkhorn


k=1
n = k*100



mu1 = k*10
mu2 = k*60
mu3 = k*20
mu4 = k*70
t1 = 0.1
t2 = 0.7
    

x = range(n)
y = range(n-1,-1,-1)

r = t1*poisson.pmf(x, mu1) + (1-t1)*poisson.pmf(x,mu2)
r = r/r.sum()
c = t2*poisson.pmf(x, mu3) + (1-t2)*poisson.pmf(x,mu4)
c = c/c.sum()




M = euc_costs(n,n)

P = []

q = np.array([2,1,0.8, 0.5])
l = np.array([50,25,10,1])

nq = len(q)
nl = len(l)


for i in range(nl):
    for j in range(nq):
        if (q[j] <1):
            P_tmp,_,_ = second_order_sinkhorn(q[j],M,r,c,l[i],1E-2)
        elif (q[j]==1):
            P_tmp = Sinkhorn(np.exp(-l[i]*np.matrix(M)),r,c,1E-2)
        else:
            P_tmp,_ = KL_proj_descent(q[j],M,r,c,l[i],1E-2, 50, rate = 1, rate_type = "square_summable")
        P.append(P_tmp)


fig = plt.figure(figsize=(8, 8))

outer_grid = gridspec.GridSpec(2, 2, width_ratios=[1,5], height_ratios=[1,5])
outer_grid.update(wspace=0.01, hspace=0.01)
# gridspec inside gridspec
outer_joint = gridspec.GridSpecFromSubplotSpec(nq,nl, subplot_spec=outer_grid[1,1],wspace=0.02, hspace=0.02)
outer_row_marg = gridspec.GridSpecFromSubplotSpec(nq,1, subplot_spec=outer_grid[1,0],wspace=0.02, hspace=0.02)
outer_col_marg = gridspec.GridSpecFromSubplotSpec(1,nl, subplot_spec=outer_grid[0,1],wspace=0.02, hspace=0.02)


for a in range(nl):
    for b in range (nq):
        ax = plt.Subplot(fig, outer_joint[a,b])
        ax.imshow(P[a + 3*b], origin='upper', interpolation = None, aspect = 'auto', cmap = 'Greys')
        rect = Rectangle((0, 0), k*n-1, k*n-1, fc='none', ec='black')     
        rect.set_width(0.8)
        rect.set_bounds(0,0,k*n-1,k*n-1)
        ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
        ax.set_axis_bgcolor('white')
    
for i in range(nl):
    ax_row = plt.Subplot(fig,outer_row_marg[i], sharey = ax)
    ax_row.plot(1-r, x)
    fig.add_subplot(ax_row)
    ax_col = plt.Subplot(fig,outer_col_marg[i], sharex = ax)
    ax_col.plot(x,c)
    fig.add_subplot(ax_col)
    ax_row.axes.get_xaxis().set_visible(False)
    ax_row.axes.get_yaxis().set_visible(False)
    left, width = .25, .5
    bottom, height = .25, .5
    top = bottom + height
    ax_row.text(-0.05, 0.5*(bottom+top), 'q = {0}'.format(q[i]), horizontalalignment='right', verticalalignment='center', rotation='vertical',transform=ax_row.transAxes, fontsize='medium')
    ax_col.axes.get_xaxis().set_visible(False)
    ax_col.axes.get_yaxis().set_visible(False)
    ax_col.set_title(r'$\lambda$'+' = {0}'.format(l[i]),fontsize='medium')
    ax_row.set_axis_bgcolor('white')
    ax_col.set_axis_bgcolor('white')
    

plt.show()
plt.savefig('/home/boris/Documents/Stage NICTA/grid_OT.pdf')