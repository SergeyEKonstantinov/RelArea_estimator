# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:29:27 2024

@author: konstse
"""




from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from numpy.random import uniform as rand, randint, choice as ch
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde



# Some latex for better text visuals. 
# comment this block if you doesn't use it or care
plt.rcParams.update({
    "text.usetex": True,
    "font.family":  "Times New Roman",
    "font.sans-serif":  "Times New Roman",
})



def distr_Interpolator(bins_num):
    '''
    INTERPOLATOR FOR DISTRIBUTION OVER NUMERICAL DATA (UNIT CUBES)
    
    bins_num - number of bins that are going to be used for the distribution.
               The distribution is precalculated and accessible only for 
               bins_num = 12, 60 or 140 bins.
    '''
    
    distr = np.load(r'bars_{}.npz'.format(bins_num)) 
    bins, ret = distr['bars'], distr['hist']
        
    return interp1d(bins, ret, kind = 1)
    




def sizes_Interpolator(dots_num, mode = 0):
    '''
    INTERPOLATOR FOR DISTRIBUTION OVER EXPERIMENTAL DATA 
    
    Requires supplementary file "Areas Pt together and separately.xlsx" 
    in the same folder
    
    dots_num - number of linearly spaced points that 
               gaussian Kernal density approximator 
               is going to approximate the distribution
    
    mode = 0 - all samples
    mode = 1 - 17 samples
    mode = 2 - 9c samples
    '''
    
    sizes = pd.read_excel('Areas Pt together and separately.xlsx', index_col = 0)  
    
    ret17 = sizes.loc[sizes.index < 10]['Area mkm2']
    ret9 = sizes.loc[sizes.index >= 10]['Area mkm2']
    retAll = sizes['Area mkm2']
    
    if mode == 0: 
        
        sa = np.linspace(0, retAll.max(), dots_num)
        da = gaussian_kde(retAll)(sa)# * 2*sa
        
    elif mode == 1: 
        
        sa = np.linspace(0, ret17.max(), dots_num)
        da = gaussian_kde(ret17)(sa)# * 2*sa
        
    else: 
        
        sa = np.linspace(0, ret9.max(), dots_num)
        da = gaussian_kde(ret9)(sa)# * 2*sa
      
    da /= da.sum()
        
    return da, sa



def simulate_matrix(M = 1000, p = 0.001, samples_num = 1000,inclusions_num = 100,
                    size_distr = True):
    '''
    THE FLAT UNFOLDING GENERATOR
    
    M - edge size of the unfolding
    p - probability of inclusion
    samples_num - number of randomly picked lxl sections
    inclusions_num - maximum number of inclusions used for error estimation
    size_distr - include experimental distribution or not
    '''

    # Keyword classifier for numerical distribution
    pallet = {
        12 : rand(0.1177067572330835, 1.4124810867970021, M**2),
        60 : rand(0.0235413514466167, 1.4124810867970021, M**2),
        140 : rand(0.010089150619978587, 1.4124810867970021, M**2),
    }
    
    # The bumber of bins in numerical distribution - 12/60/140.
    hist_num = 12
    
    # Creation, interpolation and normalization of the unit cube distribution
    p_ = distr_Interpolator(hist_num)(pallet[hist_num])
    p_ /= p_.sum()
    
    
    # The unfolding is generated as 1d M^2 array from 
    # unit cube of experimental distribution with probability 
    # p for each instance 
    samples = ch(pallet[hist_num], size = M**2, p = p_)
    
    # The mask decide whether the picked inclusion 
    # should be included in the unfolding
    mask = ch([0, 1], size = (M, M), p = [1 - p, p])
    
    # Including size distribution if size_distr is True
    if size_distr:
        
        s_ = sizes_Interpolator(1000, 2)
        samples =  ch(s_[1], size = M**2, p = s_[0])
    
    # Reshaping the unfolding into 2d MxM matrix
    samples = samples.reshape((M, M))
    
    # The unfolding content is a product of 
    # generated samples with their positions
    matrix = samples*mask
    
    # The true relative area is just the 
    # sum over the matrix over its area.
    trueRelArea = matrix.sum() / M**2
        

    # THIS FUNCTION RETURNS RANDOM lxl SECTION WITH EXACTLY Dnum INCLUSIONS 
    # poss list is here for insuring that randomly picked lxl section is unique. 
    poss = []
    def pickMeshDots_(Dnum):

        # this infinite loop works faster that you expect
        while True:
            
            N = int(np.sqrt(Dnum / p))
            
            N2 = int(N/2)
            
            RS_position = randint(N2 + 1, M - N2 + 1, 2)
            up, down = RS_position[0] - N2, RS_position[0] + N2
            left, right = RS_position[1] - N2, RS_position[1] + N2
        
            section_mask = matrix[up : down, left : right] 
            
            if np.count_nonzero(section_mask) == Dnum:
        
                poss.append(RS_position)
                return section_mask, N
    
    # Percentile storage    
    quants = np.zeros((inclusions_num, 10))
    
    # The main nested cycle provided with time profiler, 
    # so you know how many coffees you you can expect to consume before the script is finished.
    #
    # Calculation of percentiles for all inclusions from 1 to their max number
    # averaged over number of randomly picked lxl sections for each inclusion number
    for k in tqdm(range(1, inclusions_num + 1)):
        
        areas_prom = np.zeros(samples_num)
    
        for N in range(1, samples_num + 1):
        
            # points from random section
            section_mask, N_ = pickMeshDots_(k)
            areas_prom[N - 1] = section_mask.sum() / N_**2
            
        
        # using database is useful for faster statistics
        fullDB = pd.DataFrame({'areas' : areas_prom})
        fullDB.areas = (fullDB.areas - trueRelArea) / fullDB.areas * 100
        d = fullDB.areas.describe(percentiles=[0, .05, .125, .5, .45, .55, .875, .95, 1.]) 
        
        quants[k - 1, :] = np.array([d['0%'], d['100%'], 
                                     d['5%'], d['95%'], 
                                     d['12.5%'], d['87.5%'], 
                                     d['45%'], d['55%'], 
                                     d['50%'], d['50%']]) 
        
        if k == 5:
            
            distr5 = fullDB.areas.to_list()
            
        if k == 20:
            
            distr20 = fullDB.areas.to_list()
            
        if k == 60:
            
            distr60 = fullDB.areas.to_list()
            
                        
    return quants, distr5, distr20, distr60


# plotting    
f, (a, b) = plt.subplots(1, 2, dpi = 200, figsize = (12, 5), 
                         sharey = True, sharex = True)
f.subplots_adjust(wspace = 0.05)


# This cycle performes averaging over 10 randomly generated unfoldings.
# you can vary any of {M, p, samples_num, inclusions_num, matrix_nums}
for ax, type_ in zip([a, b], [0, 1]):
    
    
    M = 1000 
    p = 0.001
    samples_num = 100
    inclusions_num = 100
    matrix_nums = 10       
    
    quantiles = np.zeros((inclusions_num, 10))
    section5 = []
    section20 = []
    section60 = []
    for matrix in range(matrix_nums):
    
        ret = simulate_matrix(M = M, p = p, 
                              samples_num = samples_num,
                              inclusions_num = inclusions_num,
                              size_distr = type_)
    
        quantiles += ret[0]
        section5.append(ret[1])
        section20.append(ret[2])
        section60.append(ret[3])
        
    quantiles /= matrix_nums
    
    x_approx = np.linspace(-100, 100, 10000)
     
    promsec5 = np.array(section5).ravel()
    section5 = gaussian_kde(promsec5)(x_approx)
        
    promsec20 = np.array(section20).ravel()
    section20 = gaussian_kde(promsec20)(x_approx)
        
    promsec40 = np.array(section60).ravel()
    section60 = gaussian_kde(promsec40)(x_approx)
    
        
    dots_ = range(1, inclusions_num + 1)
    ax.set_ylim((-100, 100))
    ax.set_xlim((1, 100))
    ax.set_xticks((1, 20, 40, 60, 80, 100), (1, 20, 40, 60, 80, 100))
    
    # Ticks inside the plot
    # ax.tick_params(axis="y",direction="in")
    # ax.tick_params(axis="x",direction="in")

    ax.fill_between(dots_, quantiles.T[0], quantiles.T[1], alpha = 0.25, color = 'k')
    ax.fill_between(dots_, quantiles.T[2], quantiles.T[3], alpha = 0.25, color = 'k')
    ax.fill_between(dots_, quantiles.T[4], quantiles.T[5], alpha = 0.25, color = 'k')
    ax.fill_between(dots_, quantiles.T[6], quantiles.T[7], alpha = 0.25, color = 'k')
    ax.fill_between(dots_, quantiles.T[8], quantiles.T[8], alpha = 1, color = 'r')
    
    ax.grid(alpha = 0.1)
    
    if type_ == 1:
    
        fig, aig = plt.subplots(1, 1, dpi = 200)
    
        aig.set_ylim((-100, 100))
        aig.set_xlim((1, 100))
        aig.set_xticks((1, 20, 40, 60, 80, 100), (1, 20, 40, 60, 80, 100))
        
        # Ticks inside the plot
        # aig.tick_params(axis="y",direction="in")
        # aig.tick_params(axis="x",direction="in")
    
        aig.fill_between(dots_, quantiles.T[0], quantiles.T[1], alpha = 0.25, color = 'k')
        aig.fill_between(dots_, quantiles.T[2], quantiles.T[3], alpha = 0.25, color = 'k')
        aig.fill_between(dots_, quantiles.T[4], quantiles.T[5], alpha = 0.25, color = 'k')
        aig.fill_between(dots_, quantiles.T[6], quantiles.T[7], alpha = 0.25, color = 'k')
        aig.fill_between(dots_, quantiles.T[8], quantiles.T[8], alpha = 1, color = 'r')
        
        aig.grid(alpha = 0.1)
        red_patch1 = Patch(color = 'k', label = r'$100\%$', alpha = 0.2)
        red_patch2 = Patch(color = 'k', label = r'$90\%$', alpha = 0.3)
        red_patch3 = Patch(color = 'k', label = r'$75\%$', alpha = 0.4)
        red_patch4 = Patch(color = 'k', label = r'$10\%$', alpha = 0.5)
        red_patch5 = Line2D([0], [0], color = 'r', lw = 1, linestyle = '-', 
                            label = r'$mean$')
        legend = aig.legend(handles=[red_patch1, red_patch2, red_patch3, red_patch4, red_patch5], 
                  ncol = 5, mode = 'strech', fontsize = 11, 
                  handleheight = 0.5, handlelength = 1)
        legend.get_frame().set_alpha(None)
        aig.set_ylabel(r'$\delta(N),$ $100 \%$', fontsize = 20)
        aig.set_xlabel(r'$N$', fontsize = 20)
        
    
    f.text(0.5, 0.04, r'$N$', va = 'center', ha = 'center', fontsize = 20)
red_patch1 = Patch(color = 'k', label = r'$100\%$', alpha = 0.2)
red_patch2 = Patch(color = 'k', label = r'$90\%$', alpha = 0.3)
red_patch3 = Patch(color = 'k', label = r'$75\%$', alpha = 0.4)
red_patch4 = Patch(color = 'k', label = r'$10\%$', alpha = 0.5)
red_patch5 = Line2D([0], [0], color = 'r', lw = 1, linestyle = '-', 
                    label = r'$mean$')
f.legend(handles=[red_patch1, red_patch2, red_patch3, red_patch4, red_patch5], 
          ncol = 5, mode = 'center', fontsize = 15, bbox_to_anchor=(0.8, 1.005))
a.set_ylabel(r'$\delta(N),$ $\%$', fontsize = 20)
    
# This line saves the plot. You can choose extension to your purpose (pdf, png, esp...)
# f.savefig('methods_graph.pdf', bbox_inches = 'tight')

