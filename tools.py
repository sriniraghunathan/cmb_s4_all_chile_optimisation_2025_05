import numpy as np, sys, os, copy, scipy as sc
import astropy
from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
from pylab import *
import copy

def combine_fisher( F_mat_arr, param_names_arr, small_diag_element = 1e-3):

    param_names_diff_dic = {'a_s': 'As', 'n_s': 'ns', 'omega_c_h2': 'omch2',  'omega_b_h2': 'ombh2', 'w_0': 'ws', 'w_a': 'wa'}
    param_names_arr_mod = []
    for curr_param_names in param_names_arr:
        curr_param_names_mod = []
        for p in curr_param_names:
            if p in param_names_diff_dic:
                curr_param_names_mod.append( param_names_diff_dic[p] )
            else:
                curr_param_names_mod.append(p)
        param_names_arr_mod.append(curr_param_names_mod)
    param_names_arr = np.copy(param_names_arr_mod)
    combined_param_names = np.unique(np.concatenate( param_names_arr ) )
    nparams = len(combined_param_names)
    ##print(combined_param_names, nparams); sys.exit()

    combined_F_mat = np.zeros((nparams, nparams))
    for curr_param_names, curr_F_mat in zip(param_names_arr, F_mat_arr):
        for pcntr1, p1 in enumerate( curr_param_names ):            
            for pcntr2, p2 in enumerate( curr_param_names ):
                curr_pind1 = np.where(combined_param_names == p1)[0][0]
                curr_pind2 = np.where(combined_param_names == p2)[0][0]
                combined_F_mat[curr_pind2, curr_pind1] += curr_F_mat[pcntr2, pcntr1]
    
    F_mat_mod_arr = []
    for curr_param_names, curr_F_mat in zip(param_names_arr, F_mat_arr):
        F_mat_mod = np.zeros((nparams, nparams))
        if small_diag_element is not None:
            small_diag_mat = np.zeros_like( F_mat_mod )
            np.fill_diagonal(small_diag_mat, small_diag_element)
            F_mat_mod = F_mat_mod + small_diag_mat
            ##print(F_mat_mod); sys.exit()

        for pcntr1, p1 in enumerate( curr_param_names ):            
            for pcntr2, p2 in enumerate( curr_param_names ):
                curr_pind1 = np.where(combined_param_names == p1)[0][0]
                curr_pind2 = np.where(combined_param_names == p2)[0][0]
                F_mat_mod[curr_pind2, curr_pind1] += curr_F_mat[pcntr2, pcntr1]
        F_mat_mod_arr.append(F_mat_mod)

    return combined_F_mat, combined_param_names, F_mat_mod_arr

def fix_params(F_mat, param_names, fix_params):

    #remove parameters that must be fixed    
    F_mat_refined = []
    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 in fix_params or p2 in fix_params: continue
            F_mat_refined.append( (F_mat[pcntr2, pcntr1]) )

    totparamsafterfixing = int( np.sqrt( len(F_mat_refined) ) )
    F_mat_refined = np.asarray( F_mat_refined ).reshape( (totparamsafterfixing, totparamsafterfixing) )

    param_names_refined = []
    for p in param_names:
        if p in fix_params: continue
        param_names_refined.append(p)

    return F_mat_refined, param_names_refined

def get_sigma_of_a_parameter(F_mat, param_names, desired_param_arr, prior_dic = None, fix_params_arr = None):

    F_mat_mod = np.copy(F_mat)
    if np.sum(F_mat_mod) == 0:
        sigma_vals = {}
        if desired_param_arr is not None:
            for desired_param in desired_param_arr:
                sigma_vals[desired_param] = 0.
        return sigma_vals

    param_names_mod = np.copy(param_names)

    param_names_mod = np.asarray( param_names_mod )
    fix_params_arr = np.asarray( fix_params_arr )

    if prior_dic is not None: #add priors.
        F_mat_mod = fn_add_prior(F_mat_mod, param_names_mod, prior_dic)

    if fix_params_arr is not None:
        F_mat_mod, param_names_mod = fix_params(F_mat_mod, param_names_mod, fix_params_arr)

    cov_mat = np.linalg.inv(F_mat_mod)
    param_names_mod = np.asarray( param_names_mod )

    sigma_vals = {}
    if desired_param_arr is not None:
        for desired_param in desired_param_arr:
            #print('\textract sigma(%s)' %(desired_param))
            pind = np.where(param_names_mod == desired_param)
            pcntr1, pcntr2 = pind, pind
            cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]
            cov_extract = np.asarray( [cov_mat[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
            sigma = cov_extract[0,0]**0.5
            
            sigma_vals[desired_param] = sigma

    return sigma_vals      

def format_axis(ax, fx, fy, maxxloc=None, maxyloc = None):
    """
    function to format axis fontsize.


    Parameters
    ----------
    ax: subplot axis.
    fx: fontsize for xaxis.
    fy: fontsize for yaxis.
    maxxloc: total x ticks.
    maxyloc: total y ticks.

    Returns
    -------
    formatted axis "ax".
    """
    for label in ax.get_xticklabels(): label.set_fontsize(fx)
    for label in ax.get_yticklabels(): label.set_fontsize(fy)
    if maxyloc is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=maxxloc))
    if maxxloc is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=maxxloc))

    return ax

def convert_param_to_latex(param):
    greek_words_small = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 
                        'lambda', 'mu', 'nu', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
    greek_words_captial = [w.capitalize() for w in greek_words_small]
    greek_words = greek_words_small + greek_words_captial
    math_words = ['z']

    tmp_param_split = param.split('_')
    if len( tmp_param_split ) == 1:
        latex_param = r'$%s$' %(param)
    else:
        tmpval = tmp_param_split[0]
        if tmpval in greek_words:
            tmpval = '\%s' %(tmpval)
        latex_param = '%s' %(tmpval)
        braces_arr = ''
        for tmpval in tmp_param_split[1:]:
            if tmpval in greek_words:
                tmpval = '\%s' %(tmpval)
            if tmpval in math_words:
                latex_param = '%s_{%s' %(latex_param, tmpval)
            else:
                latex_param = '%s_{\\rm %s' %(latex_param, tmpval)
            braces_arr = '%s}' %(braces_arr)
        latex_param = '%s%s' %(latex_param, braces_arr)
        latex_param  = r'$%s$' %(latex_param)

    return latex_param

def get_latex_param_str(param):
    params_str_dic= {\
    'norm_YszM': r'${\rm log}(Y_{\ast})$', 'alpha_YszM': r'$\alpha_{_{Y}}$',\
    'beta_YszM': r'$\beta_{_{Y}}$', 'gamma_YszM': r'$\gamma_{_{Y}}$', \
    'alpha': r'$\eta_{\rm v}$', 'sigma_8': r'$\sigma_{\rm 8}$', \
    'one_minus_hse_bias': r'$1-b_{\rm SZ}$', 
    'omega_m': r'$\Omega_{\rm m}$', 'omegam': r'$\Omega_{\rm m}$', \
    'h0':r'$h$', 'm_nu':r'$\sum m_{\nu}$', \
    'ombh2': r'$\Omega_{b}h^{2}$', 'omch2': r'$\Omega_{c}h^{2}$', 'omega_lambda': r'$\Omega_{\Lambda}$',
    'omega_b_h2': r'$\Omega_{b}h^{2}$', 'omega_c_h2': r'$\Omega_{c}h^{2}$',
    'omega_k': r'$\Omega_{k}$',
    'w0': r'$w_{0}$', 'wa': r'$w_{a}$', \
    'tau': r'$\tau_{\rm re}$', 
    'As': r'$A_{\rm s}$', 
    #'As': r'log$A_{\rm s}$', 
    'ns': r'$n_{\rm s}$', 'neff': r'$N_{\rm eff}$', \
    'mnu': r'$\sum m_{\nu}$', 'thetastar': r'$\theta_{\ast}$', \
    'h': r'$h$', 'omk': r'$\Omega_{k}$', 'ws': r'$w_{0}$', \
    'w_0': r'$w_{0}$', 'w_a': r'$w_{a}$', \
    'yhe': r'$Y_{P}$','nnu': r'N$_{\rm eff}$','omegak': r'$\Omega_{k}$',\
    'w': r'$w_{0}$', 'nrun': r'$n_{run}$', 'Aphiphi':r'$A^{\phi\phi}$', \
    'nnu': r'$N_{\rm eff}$', 'H0': r'$H_0$', \
    #adding more
    'a_s': r'$A_{\rm s}$', 'h': r'$h$', 'n_s': r'$n_{\rm s}$', \
    'omega_m': r'$\Omega_{m}$', 
    'omega_b': r'$\Omega_{b}$', 'omegab': r'$\Omega_{b}$',\
    #SNe
    'M': r'$M$', 'alpha': r'$\alpha$', 'beta': r'$\beta$',\
    }

    if param not in params_str_dic:
        return convert_param_to_latex(param)
    else:
        return params_str_dic[param]

def add_prior(F_mat, param_names, prior_dic):

    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 == p2 and p1 in prior_dic:
                prior_val = prior_dic[p1]
                F_mat[pcntr2, pcntr1] += 1./prior_val**2.

    return F_mat

def get_gaussian(mean, sigma, minx, maxx, delx = None):

    if delx is None: 
        delx = (maxx - minx)/1000000.

    ##print(mean, sigma, minx, maxx, delx)#; sys.exit()
    x = np.arange(minx, maxx, delx)

    #return x, 1./(2*np.pi*sigma)**0.5 * np.exp( -(x - mean)**2. / (2 * sigma**2.)  )
    return x, np.exp( -(x - mean)**2. / (2 * sigma**2.)  )

def get_ellipse_specs(COV, howmanysigma = 1):
    """
    Refer https://arxiv.org/pdf/0906.4123.pdf
    """
    assert COV.shape == (2,2)
    confsigma_dic = {1:2.3, 2:6.17, 3: 11.8}

    sig_x2, sig_y2 = COV[0,0], COV[1,1]
    sig_xy = COV[0,1]
    
    t1 = (sig_x2 + sig_y2)/2.
    t2 = np.sqrt( (sig_x2 - sig_y2)**2. /4. + sig_xy**2. )
    
    a2 = t1 + t2
    b2 = t1 - t2

    a = np.sqrt(abs(a2))
    b = np.sqrt(abs(b2))

    t1 = 2 * sig_xy
    t2 = sig_x2 - sig_y2
    theta = np.arctan2(t1,t2) / 2.
    
    alpha = np.sqrt(confsigma_dic[howmanysigma])
    
    #return (a*alpha, b*alpha, theta)
    return (a*alpha, b*alpha, theta, alpha*(sig_x2**0.5), alpha*(sig_y2**0.5))
