# utility functions to run period searches

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style("whitegrid")
sns.set_palette('colorblind')

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize as op

from stingray import Lightcurve
from stingray import Powerspectrum

from stingray.pulse import z2_n_probability, z2_n_detection_level, z_n

import emcee
import george
from george import kernels
import corner


class GaussLikelihood(object):
    
    def __init__(self, x, y, yerr, model):
        """
        Gaussian Log-likelihood.

        Parameters
        ----------
        x : iterable
             The independent variable

        y : iterable
             The dependent variable

        yerr : iterable
             The uncertainty on y

        model : function
             A model definition of type `fun(x, *args)`. The first parameter must be 
             an iterable defining the dependent variable, the 
             remaining parameters are the parameters of the model
             Should return an iterable with the model values.
        """

        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
    def evaluate(self, pars, neg=False):
        """
        Evaluate the Gaussian log-likelihood. 

        Parameters
        ----------
        pars : iterable
            A list with parameter values

        neg : bool, default False
            If `True`, return the *negative* log-likelihood.
            This is useful for optimization.

        Returns
        -------
        loglike : float
            The value of the (negative) log-likelihood

        """
        mean_model = self.model(self.x, *pars)

        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) -
                         (self.y-mean_model)**2/(2.*self.yerr**2))

        if not np.isfinite(loglike):
            loglike = -np.inf

        if neg:
            return -loglike
        else:
            return loglike
        
    def __call__(self, parameters, neg=False):
        """
        Evaluate the Gaussian log-likelihood. 

        Parameters
        ----------
        pars : iterable
            A list with parameter values

        neg : bool, default False
            If `True`, return the *negative* log-likelihood.
            This is useful for optimization.

        Returns
        -------
        loglike : float
            The value of the (negative) log-likelihood
        """
        return self.evaluate(parameters, neg)


def sinusoid(t, logamp, period, bkg):
    """
    A sinusoidal model.
    
    Parameters
    ----------
    t : iterable
        The dependent coordinate
        
    logamp : float
        The logarithm of the sinusoidal amplitude
        
    period : float
        The period of the sinusoid
        
    phase : float [0, 2*pi]
        The phase of the sinusoidal signal
        
    bkg : float
        The mean magnitude
    
    Returns
    -------
    res : numpy.ndarray
        The result
    """
    res = np.exp(logamp) * np.sin(2.*np.pi*t/period) + bkg
    return res


class GaussPosterior(object):
    
    def __init__(self, x, y, yerr, model):
        """
        Gaussian Log-likelihood.

        Parameters
        ----------
        x : iterable
             The independent variable

        y : iterable
             The dependent variable

        yerr : iterable
             The uncertainty on y

        model : function
             A model definition of type `fun(x, *args)`. The first parameter must be 
             an iterable defining the dependent variable, the 
             remaining parameters are the parameters of the model
             Should return an iterable with the model values.
        """
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
        self.loglikelihood = GaussLikelihood(x, y, yerr, model)
        #self.vm = scipy.stats.vonmises(kappa=0.2, loc=0.0)
        self.flat_prior = np.log(1/20.0) + np.log(1/(1 - 1/24.0)) + \
                          np.log(1/5.0) #+ np.log(1.0)
        
    def logprior(self, pars):
        """
        Baked-in priors for the sinusoidal model:
 
        log-amplitude: uniform prior between -20 and 20
        period : uniform prior between 1 hour and 1 day
        mean brightness level: uniform prior between 20 and 25

        Parameters
        ----------
        pars : iterable
            A list with parameter values

        Returns
        -------
        lprior : float
            The value of the log-prior 
        """
        logamp = pars[0]
        period = pars[1]
        #phase = pars[2]
        bkg = pars[2]
        
        if logamp < -20 or logamp > 20:
            return -np.inf
        elif period < 1/24.0 or period > 1.0:
            return -np.inf
        elif bkg < 20 or bkg > 25:
            return -np.inf 
        #elif phase < 0 or phase > 1.0:
        #    return -np.inf
        else:
            return self.flat_prior
        
    def logposterior(self, pars, neg=False):
        """
        The log-posterior for a Gaussian log-likelihood

        Parameters
        ----------
        pars : iterable
            A list with parameter values

        neg : bool, default False
            If `True`, return the *negative* log-likelihood.
            This is useful for optimization.

        Returns
        -------
        lpost : float
            The value of the (negative) log-posterior

        """
        lpost = self.logprior(pars) + self.loglikelihood(pars, neg=False)
        
        if not np.isfinite(lpost):
            lpost = -np.inf
            
        if neg:
            return -lpost
        else:
            return lpost
        
    def __call__(self, pars, neg=False):
        return self.logposterior(pars, neg)



def model_sine_curve(data, start_pars, nwalkers=200, niter=10000, nsim=500, namestr="test", fitmethod="powell", threads=1):
    """
    Model asteroid or comet data with a sine curve.

    Produces a bunch of plots

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame with the data. 
        Should have the following columns:
            * `time`: the time stamps of the data
            * `mag`: the corresponding magnitudes
            * `mag_err`: The uncertainties in the magnitudes
            * `ObsID`: the Minor Planet Center ID of the observatory that took the data

    start_pars : iterable
        A list of starting parameters for the optimization run. 

    nwalkers : int
        The number of walkers for the MCMC run

    niter : int
        The length of iterations to run the MCMC chains for

    nsim : int
        The number of iterations to extract from each chain for posterior inference

    namestr : str
        A string containing the absolute path to a target directory as well as an identifier 
        for storing data products and figures.

    fitmethod : str
        A valid method for `scipy.optimize.minimize`

    Returns
    -------


    """
    time = np.array(data.time)
    mag = np.array(data.mag)
    mag_err = np.array(data.mag_err)
    obsid = np.array(data.ObsID)

    # define a Posterior object
    lpost = GaussPosterior(time, mag, mag_err, sinusoid)

    # do a maximum-a-posteriori fit of the model to the data
    res = op.minimize(lpost, start_pars, args=(True), method=fitmethod)

    # define a set of model time stamps for plotting
    model_time = np.linspace(time[0], time[-1], 2000)
    # evaluate the MAP model at the time stamps
    m = sinusoid(model_time, *res.x)

    # plot the data and the MAP model
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.errorbar(time, mag, yerr=mag_err, fmt="o", markersize=5, color="black", label="data")
    ax.plot(model_time, m, color="red", lw=2, label="best-fit model")
    ax.set_xlim(time[0]-0.1, time[-1]+0.1)
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig("%s_lc_mapfit.pdf"%namestr, format="pdf")
    plt.close()

    # Set up the sampler.
    ndim = 3
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, threads=threads)

    # Initialize the walkers.
    p0 = res.x + 0.01 * np.random.randn(nwalkers, ndim)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, niter)

    # plot trace plots of the parameter values
    par_names = ["logamp", "period", "bkg"]
    for i, pn in enumerate(par_names):
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
        ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter value")
        ax.set_title(pn)
        plt.tight_layout()
        plt.savefig("%s_sine_mcmc_%s.pdf"%(namestr, pn), format="pdf")
        plt.close()

    # extract the last `nsim` iterations from the sampler:
    flatchain = np.concatenate(sampler.chain[:,-nsim:, :], axis=0)
    names = [r"$\log(A)$", r"$P$ [days]", r"$\lambda_{\mathrm{bkg}}$"]
    
    fig = corner.corner(flatchain, labels=names)
    plt.savefig("%s_sine_corner.pdf"%namestr, format="pdf")
    plt.close()

    # let's extract posterior means and percentiles:
    logamp_pmean = np.mean(flatchain[:,0])
    logamp_percentile = np.percentile(flatchain[:,0], [50-(68.27/2.0), 50, 50+(68.27/2.)], axis=0) 
    
    logamp_neg = logamp_pmean - logamp_percentile[0]
    logamp_plus = logamp_percentile[-1] - logamp_pmean

    print("The posterior mean and uncertainty for the log-amplitude is $\log(A) = $ %.3f (+%.3f/-%.3f)"%(logamp_pmean, logamp_plus, logamp_neg))

    # let's extract posterior means and percentiles:
    period_pmean = np.mean(flatchain[:,1])
    period_percentile = np.percentile(flatchain[:,1], [50-(68.27/2.0), 50, 50+(68.27/2.)], axis=0)
    
    period_neg = period_pmean - period_percentile[0]
    period_plus = period_percentile[-1] - period_pmean
 
    print("The posterior mean and uncertainty for the period is $P = $ %.3f (+%.3f/-%.3f)"%(period_pmean, period_plus, period_neg))

    # let's extract posterior means and percentiles:
    bkg_pmean = np.mean(flatchain[:,-1])
    bkg_percentile = np.percentile(flatchain[:,-1], [50-(68.27/2.0), 50, 50+(68.27/2.)], axis=0)
    
    bkg_neg = bkg_pmean - bkg_percentile[0]
    bkg_plus = bkg_percentile[-1] - bkg_pmean
 
    print("The posterior mean and uncertainty for the mean magnitude is $\lambda =$ %.3f (+%.3f/-%.3f)"%(bkg_pmean, bkg_plus, bkg_neg))

    # let's phase-fold at the posterior mean!
    phase = time/period_pmean % 1
    phase *= (2.*np.pi)

    labels = ["0", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"]
    ticks = [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]

    data["phase"] = phase

    # now we can make a figure!    
    fig = plt.figure(figsize=(8,6))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    
    # find all unique observatories in the data set
    unique_obs = np.unique(obsid)

    # get a list of colours:
    colors = sns.color_palette("colorblind", n_colors=len(unique_obs))
    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)
 
        ax1.errorbar(t, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)
    
    for i in range(30):
        # Choose a random walker and step.
        w = np.random.randint(flatchain.shape[0])
        p = flatchain[w]
        #ph = (p[2]/(2.0*np.pi) - int(p[2]/(2.0*np.pi))) * 2 * np.pi
        m = sinusoid(model_time, p[0], p[1], p[2])
        # Plot a single sample.
        if i == 0:
            ax1.plot(model_time, m, alpha=0.2, color="black", 
                     label="posterior draws", zorder=0)
        else:
            ax1.plot(model_time, m, alpha=0.2, color="black", zorder=0)
    
    #ax1.set_xlim(58055.2, 58056.32)
    ax1.set_xlabel("Time [MJD]")
    ax1.set_ylabel("Magnitude");
    leg = ax1.legend(frameon=True)
    leg.get_frame().set_edgecolor('grey')
    
    ax1.set_ylim(22, 25.5)
    ax1.set_ylim(ax1.get_ylim()[::-1])
  
    # find all unique observatories in the data set
    unique_obs = np.unique(obsid)

    # get a list of colours:
    colors = sns.color_palette("colorblind", n_colors=len(unique_obs))
 
    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        ph = np.array(d.phase)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax3.errorbar(ph, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)

    ax3.legend()
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels)
    ax3.set_title(r"Folded light curve, $P = 4.07$ hours")
    ax3.set_ylim(ax3.get_ylim()[::-1])
    plt.tight_layout()
    
    plt.savefig("%s_sine_posterior.pdf"%namestr, format="pdf")
    plt.close()

    # fit all posterior draws individually

    unique_obs = data.ObsID.unique()

    for i, obs in enumerate(np.array(unique_obs)):
        fig, (ax1) = plt.subplots(1, 1, figsize=(8,4))
    
        d = data.loc[data["ObsID"] == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)
        
        ax1.errorbar(t,m, yerr=me,
                     color=sns.color_palette("colorblind", n_colors=len(unique_obs))[i], fmt="o", markersize=4, label=obs)
    
        flatchain = np.concatenate(sampler.chain[:,-1000:, :], axis=0)

        for i in range(30):
            # Choose a random walker and step.
            w = np.random.randint(flatchain.shape[0])
            p = flatchain[w]
            m = sinusoid(model_time, p[0], p[1], p[2])
        #    # Plot a single sample.
            if i == 0:
                ax1.plot(model_time, m, alpha=0.2, color="black", 
                         label="posterior draws", zorder=0)
            else:
                ax1.plot(model_time, m, alpha=0.2, color="black", zorder=0)
    
        #ax1.set_xlim(58055.2, 58056.32)
        ax1.set_xlabel("Time [MJD]")
        ax1.set_ylabel("Magnitude");
        leg = ax1.legend(frameon=True)
        leg.get_frame().set_edgecolor('grey')
    
        ax1.set_ylim(22, 25.5)
        ax1.set_ylim(ax1.get_ylim()[::-1])
    
        ax1.set_xlim(t[0], t[-1])
        ax1.set_title(obs)
        plt.tight_layout()

        plt.savefig("%s_sine_obs%s.pdf"%(namestr, obs), format="pdf")
        plt.close()

    return lpost, res, sampler
    

def model_gp(data, start_pars, nwalkers=200, niter=10000, nsim=500, namestr="test", fitmethod="powell", threads=1):
    """
    Model asteroid/comet data with a Gaussian Process

    Produces a bunch of plots.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame with the data. 
        Should have the following columns:
            * `time`: the time stamps of the data
            * `mag`: the corresponding magnitudes
            * `mag_err`: The uncertainties in the magnitudes
            * `ObsID`: the Minor Planet Center ID of the observatory that took the data

    start_pars : iterable
        A list of starting parameters for the optimization run. 

    nwalkers : int
        The number of walkers for the MCMC run

    niter : int
        The length of iterations to run the MCMC chains for

    nsim : int
        The number of iterations to extract from each chain for posterior inference

    namestr : str
        A string containing the absolute path to a target directory as well as an identifier 
        for storing data products and figures.

    fitmethod : str
        A valid method for `scipy.optimize.minimize`
 
    Returns
    -------
    data : pd.DataFrame
         Same as input data frame, but with new column `phase`

    kernel : george.kernel object
         The object containing the GP kernel

    gp : george.GP object
         The Gaussian Process object

    res : scipy.optimize.OptimizationResult object
         The object with the results of the optimizer

    sampler : emcee.Sampler
         The emcee sampler with the MCMC chains

    """
    time = np.array(data.time)
    mag = np.array(data.mag)
    mag_err = np.array(data.mag_err)
    obsid = np.array(data.ObsID)

    # use a exponential sin**2 kernel:
    kernel = start_pars[1] * kernels.ExpSine2Kernel(gamma=np.exp(start_pars[2]), log_period=start_pars[3])
 
    # initialize the Gaussian Process object
    gp = george.GP(kernel, mean=np.mean(mag), fit_mean=True,
                   white_noise=np.mean(np.log(mag_err)), fit_white_noise=False)

    # compute the GP for the data points
    gp.compute(time)

    # make a regular grid for the model values
    x = np.linspace(np.min(time), np.max(time), 5000)

    # predict mean and standard deviation for the GP
    mu, var = gp.predict(mag, x, return_var=True)
    std = np.sqrt(var)
    
    # plot the data and the mean/variance for the GP
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.errorbar(time, mag, yerr=mag_err,
                 color="black", fmt="o", markersize=4)
    ax.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)
    
    ax.set_xlabel("Time [MJD]")
    ax.set_ylabel("Magnitude")

    plt.savefig("%s_gp_beforefit.pdf"%namestr, format="pdf")
    plt.close()
     
    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(mag, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    
    # And the gradient of the objective function.
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(mag, quiet=True)
    
    # You need to compute the GP once before starting the optimization.
    gp.compute(time)
    
    # Print the initial ln-likelihood.
    print(gp.log_likelihood(mag))
    
    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    
    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    print(results.x)
    print(gp.log_likelihood(mag))
    
    x = np.linspace(np.min(time), np.max(time), 5000)
    mu, var = gp.predict(mag, x, return_var=True)
    std = np.sqrt(var)
    
    # plot the best-fit model:
    fig, ax = plt.subplots(1, 1, figsize=(8,4))

    # find all unique observatories in the data set
    unique_obs = np.unique(obsid)

    # get a list of colours:
    colors = sns.color_palette("colorblind", n_colors=len(unique_obs))

    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax.errorbar(t, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)

    ax.fill_between(x, mu+std, mu-std, color="black", alpha=0.3)
    
    ax.set_xlabel("Time [MJD]")
    ax.set_ylabel("Magnitude");
    ax.legend()
    
    plt.tight_layout()        
    plt.savefig("%s_gp_maxlike.pdf"%namestr, format="pdf")
    plt.close()
    
    # posterior: 
    def lnprob(p):
        
        mean = p[0]
        logamplitude = p[1]
        loggamma = p[2]
        period = np.exp(p[3])

        if mean < -100 or mean > 100:
            #print("boo! 0")
            return -np.inf
        
        # prior on log-amplitude: flat and uninformative
        elif logamplitude < -100 or logamplitude > 100:
            #print("boo! 1")
            return -np.inf
        
        # prior on log-gamma of the periodic signal: constant and uninformative
        elif loggamma < -20 or loggamma > 20:
            #print("boo! 2")
            return -np.inf
            
        # prior on the period: somewhere between 30 minutes and 2 days
        elif period < (1./24) or period > (23/24.0):
            #print("boo! 4")
            return -np.inf

        else:
            pnew = np.array([mean, logamplitude, np.exp(loggamma), p[3]])
            #print("yay!")
            # Update the kernel and compute the lnlikelihood.
            gp.set_parameter_vector(pnew)
            return gp.lnlikelihood(mag, quiet=True)
    
    gp.compute(time)

    # Set up the sampler.
    ndim = len(gp)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads)
    
    # Initialize the walkers.
    p0 = start_pars + 0.01 * np.random.randn(nwalkers, ndim)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, niter) 

    labels=["mean_magnitude", "log_amplitude", "log_gamma", "period"]

    # make trace plots:
    for i,l in enumerate(labels):
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
        ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter value")
        ax.set_title(l)
        plt.tight_layout()
        plt.savefig("%s_gp_trace_%s.pdf"%(namestr, l), format="pdf")
        plt.close()

    flatchain = np.concatenate(sampler.chain[:,-nsim:,:], axis=0)
    flatchain[:,3] = np.exp(flatchain[:,3]) * 24.0

    fig = corner.corner(flatchain[:,1:], labels=["$\log(C)$", r"$\log(1/d^2)$", "$P$ [hours]", "$\log(M)$"], bins=60, smooth=1);
    plt.savefig("%s_gp_corner.pdf"%namestr, format="pdf", frameon=True)
    plt.close()

    # let's find the posterior maximum:
    lnprobability = np.concatenate(sampler.lnprobability[:,-nsim:], axis=0)
    max_ind = np.argmax(lnprobability)

    # parameters at the posterior maximum:
    max_pars = flatchain[max_ind, :]

    # Update the kernel and print the final log-likelihood.
    max_pars = [max_pars[0], max_pars[1], np.exp(max_pars[2]), np.log(max_pars[3]/24.0)]
   
    max_period = max_pars[3]/24.0

    # let's phase-fold at the posterior mean!
    phase = time/max_period % 1
    phase *= (2.*np.pi)

    labels = ["0", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"]
    ticks = [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]

    data["phase"] = phase
     
    gp.set_parameter_vector(max_pars)
    print(gp.log_likelihood(mag))
    
    x = np.linspace(np.min(time), np.max(time), 5000)
    mu, var = gp.predict(mag, x, return_var=True)
    std = np.sqrt(var)
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8,6))

    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax1.errorbar(t, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)

    ax1.fill_between(x, mu+2*std, mu-2*std, color="black", alpha=0.3)
    
    ax1.set_xlabel("Time [MJD]")
    ax1.set_ylabel("Magnitude");
    ax1.legend()

    # get a list of colours:
    colors = sns.color_palette("colorblind", n_colors=len(unique_obs))
 
    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        ph = np.array(d.phase)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax3.errorbar(ph, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)

    ax3.legend()
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels)
    ax3.set_title(r"Folded light curve, $P = %.3f $ hours"%(max_period*24.0))
    ax3.set_ylim(ax3.get_ylim()[::-1])


    plt.tight_layout()
    plt.savefig("%s_gp_postmax.pdf"%namestr, format="pdf")
    plt.close()


    unique_obs = np.unique(obsid)

    # fit all posterior draws individually

    for i, obs in enumerate(np.array(unique_obs)):
        fig, (ax1) = plt.subplots(1, 1, figsize=(8,4))

        d = data.loc[data["ObsID"] == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax1.errorbar(t,m, yerr=me,
                     color=sns.color_palette("colorblind", n_colors=len(unique_obs))[i], fmt="o", markersize=4, label=obs)
        ax1.fill_between(x, mu+2*std, mu-2*std, color="black", alpha=0.3)

        #ax1.set_xlim(58055.2, 58056.32)
        ax1.set_xlabel("Time [MJD]")
        ax1.set_ylabel("Magnitude");
        leg = ax1.legend(frameon=True)
        leg.get_frame().set_edgecolor('grey')

        ax1.set_ylim(22, 25.5)
        ax1.set_ylim(ax1.get_ylim()[::-1])

        ax1.set_xlim(t[0], t[-1])
        ax1.set_title(obs)
        plt.tight_layout()

        plt.savefig("%s_gp_obs%s.pdf"%(namestr, obs), format="pdf")
        plt.close()


    return data, kernel, gp, results, sampler


def model_gp_with_qpo(data, start_pars, nwalkers=200, niter=10000, nsim=500, namestr="test", fitmethod="powell", threads=1):
    """
    Model asteroid/comet data with a Gaussian Process

    Produces a bunch of plots.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame with the data. 
        Should have the following columns:
            * `time`: the time stamps of the data
            * `mag`: the corresponding magnitudes
            * `mag_err`: The uncertainties in the magnitudes
            * `ObsID`: the Minor Planet Center ID of the observatory that took the data

    start_pars : iterable
        A list of starting parameters for the optimization run. 

    nwalkers : int
        The number of walkers for the MCMC run

    niter : int
        The length of iterations to run the MCMC chains for

    nsim : int
        The number of iterations to extract from each chain for posterior inference

    namestr : str
        A string containing the absolute path to a target directory as well as an identifier 
        for storing data products and figures.

    fitmethod : str
        A valid method for `scipy.optimize.minimize`
 
    Returns
    -------
    data : pd.DataFrame
         Same as input data frame, but with new column `phase`

    kernel : george.kernel object
         The object containing the GP kernel

    gp : george.GP object
         The Gaussian Process object

    res : scipy.optimize.OptimizationResult object
         The object with the results of the optimizer

    sampler : emcee.Sampler
         The emcee sampler with the MCMC chains

    """
    time = np.array(data.time)
    mag = np.array(data.mag)
    mag_err = np.array(data.mag_err)
    obsid = np.array(data.ObsID)

    # use a exponential sin**2 kernel:
    kernel = start_pars[1] *kernels.ExpSquaredKernel(np.exp(start_pars[4])) * kernels.ExpSine2Kernel(gamma=np.exp(start_pars[2]), log_period=start_pars[3])
 
    # initialize the Gaussian Process object
    gp = george.GP(kernel, mean=np.mean(mag), fit_mean=True,
                   white_noise=np.mean(np.log(mag_err)), fit_white_noise=False)

    # compute the GP for the data points
    gp.compute(time)

    # make a regular grid for the model values
    x = np.linspace(np.min(time), np.max(time), 5000)

    # predict mean and standard deviation for the GP
    mu, var = gp.predict(mag, x, return_var=True)
    std = np.sqrt(var)
    
    # plot the data and the mean/variance for the GP
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.errorbar(time, mag, yerr=mag_err,
                 color="black", fmt="o", markersize=4)
    ax.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)
    
    ax.set_xlabel("Time [MJD]")
    ax.set_ylabel("Magnitude")

    plt.savefig("%s_gp_beforefit.pdf"%namestr, format="pdf")
    plt.close()
     
    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(mag, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    
    # And the gradient of the objective function.
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(mag, quiet=True)
    
    # You need to compute the GP once before starting the optimization.
    gp.compute(time)
    
    # Print the initial ln-likelihood.
    print(gp.log_likelihood(mag))
    
    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    
    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    print(results.x)
    print(gp.log_likelihood(mag))
    
    x = np.linspace(np.min(time), np.max(time), 5000)
    mu, var = gp.predict(mag, x, return_var=True)
    std = np.sqrt(var)
    
    # plot the best-fit model:
    fig, ax = plt.subplots(1, 1, figsize=(8,4))

    # find all unique observatories in the data set
    unique_obs = np.unique(obsid)

    # get a list of colours:
    colors = sns.color_palette("colorblind", n_colors=len(unique_obs))

    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax.errorbar(t, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)

    ax.fill_between(x, mu+std, mu-std, color="black", alpha=0.3)
    
    ax.set_xlabel("Time [MJD]")
    ax.set_ylabel("Magnitude");
    ax.legend()
    
    plt.tight_layout()        
    plt.savefig("%s_gp_maxlike.pdf"%namestr, format="pdf")
    plt.close()
    
    # posterior: 
    def lnprob(p):
        
        mean = p[0]
        logamplitude = p[1]
        loggamma = p[2]
        period = np.exp(p[3])
        logmetric = p[4]
    

        if mean < -100 or mean > 100:
            #print("boo! 0")
            return -np.inf
        
        # prior on log-amplitude: flat and uninformative
        elif logamplitude < -100 or logamplitude > 100:
            #print("boo! 1")
            return -np.inf
        
        # prior on log-gamma of the periodic signal: constant and uninformative
        elif loggamma < -20 or loggamma > 20:
            #print("boo! 2")
            return -np.inf
            
        # prior on the period: somewhere between 30 minutes and 2 days
        elif period < (1./24) or period > (23/24.0):
            #print("boo! 4")
            return -np.inf
        elif logmetric < -20 or logmetric > 20:
            return -np.inf

        else:
            pnew = np.array([mean, logamplitude, np.exp(loggamma), p[3], np.exp(logmetric)])
            #print("yay!")
            # Update the kernel and compute the lnlikelihood.
            gp.set_parameter_vector(pnew)
            return gp.lnlikelihood(mag, quiet=True)
    
    gp.compute(time)

    # Set up the sampler.
    ndim = len(gp)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads)
    
    # Initialize the walkers.
    p0 = start_pars + 0.01 * np.random.randn(nwalkers, ndim)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, niter) 

    labels=["mean_magnitude", "log_amplitude", "log_gamma", "period"]

    # make trace plots:
    for i,l in enumerate(labels):
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
        ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter value")
        ax.set_title(l)
        plt.tight_layout()
        plt.savefig("%s_gp_trace_%s.pdf"%(namestr, l), format="pdf")
        plt.close()

    flatchain = np.concatenate(sampler.chain[:,-nsim:,:], axis=0)
    flatchain[:,3] = np.exp(flatchain[:,3]) * 24.0

    fig = corner.corner(flatchain[:,1:], labels=["$\log(C)$", r"$\log(1/d^2)$", "$P$ [hours]", "$\log(M)$"], bins=60, smooth=1);
    plt.savefig("%s_gp_corner.pdf"%namestr, format="pdf", frameon=True)
    plt.close()

    # let's find the posterior maximum:
    lnprobability = np.concatenate(sampler.lnprobability[:,-nsim:], axis=0)
    max_ind = np.argmax(lnprobability)

    # parameters at the posterior maximum:
    max_pars = flatchain[max_ind, :]

    # Update the kernel and print the final log-likelihood.
    max_pars = [max_pars[0], max_pars[1], np.exp(max_pars[2]), np.log(max_pars[3]/24.0), np.exp(max_pars[4])]
   
    max_period = max_pars[3]/24.0

    # let's phase-fold at the posterior mean!
    phase = time/max_period % 1
    phase *= (2.*np.pi)

    labels = ["0", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"]
    ticks = [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]

    data["phase"] = phase
     
    gp.set_parameter_vector(max_pars)
    print(gp.log_likelihood(mag))
    
    x = np.linspace(np.min(time), np.max(time), 5000)
    mu, var = gp.predict(mag, x, return_var=True)
    std = np.sqrt(var)
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8,6))

    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax1.errorbar(t, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)

    ax1.fill_between(x, mu+2*std, mu-2*std, color="black", alpha=0.3)
    
    ax1.set_xlabel("Time [MJD]")
    ax1.set_ylabel("Magnitude");
    ax1.legend()

    # get a list of colours:
    colors = sns.color_palette("colorblind", n_colors=len(unique_obs))
 
    for i, obs in enumerate(unique_obs):
        d = data.loc[data.ObsID == obs]
        ph = np.array(d.phase)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax3.errorbar(ph, m, yerr=me,
                     color=colors[i], fmt="o", markersize=4, label=obs)

    ax3.legend()
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels)
    ax3.set_title(r"Folded light curve, $P = %.3f $ hours"%(max_period*24.0))
    ax3.set_ylim(ax3.get_ylim()[::-1])


    plt.tight_layout()
    plt.savefig("%s_gp_postmax.pdf"%namestr, format="pdf")
    plt.close()


    unique_obs = np.unique(obsid)

    # fit all posterior draws individually

    for i, obs in enumerate(np.array(unique_obs)):
        fig, (ax1) = plt.subplots(1, 1, figsize=(8,4))

        d = data.loc[data["ObsID"] == obs]
        t = np.array(d.time)
        m = np.array(d.mag)
        me = np.array(d.mag_err)

        ax1.errorbar(t,m, yerr=me,
                     color=sns.color_palette("colorblind", n_colors=len(unique_obs))[i], fmt="o", markersize=4, label=obs)
        ax1.fill_between(x, mu+2*std, mu-2*std, color="black", alpha=0.3)

        #ax1.set_xlim(58055.2, 58056.32)
        ax1.set_xlabel("Time [MJD]")
        ax1.set_ylabel("Magnitude");
        leg = ax1.legend(frameon=True)
        leg.get_frame().set_edgecolor('grey')

        ax1.set_ylim(22, 25.5)
        ax1.set_ylim(ax1.get_ylim()[::-1])

        ax1.set_xlim(t[0], t[-1])
        ax1.set_title(obs)
        plt.tight_layout()

        plt.savefig("%s_gp_obs%s.pdf"%(namestr, obs), format="pdf")
        plt.close()


    return data, kernel, gp, results, sampler


