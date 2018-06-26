# -*- coding: utf-8 -*-

"""
Probability to have a scene chosen by random at least k times on data from f folds 
when the individual prob. is p=5/n and n=5*f (5 is the min. no of waves per class)

Created on Mon Jun 25 20:21:25 2018

@author: Moritz Augustin
"""

#from pyplot import *
import numpy as np
import matplotlib.pyplot as plt


lightgray = [0.85, 0.85, 0.85]

def plot_probability_over_k(f, k_max, legend=False):

    k = np.linspace(1, k_max, num=200)
    
    p = k/80.
    n = 5*f
    
    prob_at_least_once = 1 - (1-p)**n # not 0 hits

    plt.plot(k, np.ones_like(k), '--', color=lightgray)

    plt.title('{} fold{}'.format(f, 's' if f>1 else ''))
    plt.plot(k, prob_at_least_once, label='once')
    
    prob_at_least_twice = prob_at_least_once - n*p*(1-p)**(n-1) # not 0 and not exactly one hit
    plt.plot(k, prob_at_least_twice, label='twice')
    
    if legend: 
        plt.legend()
    
    plt.ylabel('probability')
    plt.xlabel('k (scenes from n=80)')
    
    plt.yticks([0, 0.5, 0.8, 0.9, 0.95, 1.])
    
    return k, prob_at_least_once, prob_at_least_twice


plt.figure()
plt.suptitle('probability of choosing any scene at least once/twice')


# 5 folds (training)

plt.subplot(1, 2, 1)
k, p1, p2 = plot_probability_over_k(f=5, k_max=24, legend=True)

# chosen points:
k1_5f = 12
ind1 =  np.argmin(np.abs(k-k1_5f))
plt.plot([k[ind1], k[ind1]], [0, p1[ind1]], '--', color=lightgray, zorder=1)
plt.plot(k[ind1], p1[ind1], 'rx')

k2_5f = 20
ind2 = np.argmin(np.abs(k-k2_5f))
plt.plot([k[ind2], k[ind2]], [0, p2[ind2]], '--', color=lightgray, zorder=1)
plt.plot(k[ind2], p1[ind2], 'rx')

plt.xticks([0, 4,8,12,16,20,24])


# 1 fold (validation)

plt.subplot(1, 2, 2)
k, p1, p2 = plot_probability_over_k(f=1, k_max=80)

# chosen points:
k1_1f = 50
ind1 =  np.argmin(np.abs(k-k1_1f))
plt.plot([k[ind1], k[ind1]], [0, p1[ind1]], '--', color=lightgray, zorder=1)
plt.plot(k[ind1], p1[ind1], 'rx')

k2_1f = 60
ind2 =  np.argmin(np.abs(k-k2_1f))
plt.plot([k[ind2], k[ind2]], [0, p2[ind2]], '--', color=lightgray, zorder=1)
plt.plot(k[ind2], p2[ind2], 'rx')

plt.ylabel('')

plt.xticks([1, 10, 20, 30, 40, 50,60,70, 80])
plt.savefig('scene_subsampling_criterion.png')
