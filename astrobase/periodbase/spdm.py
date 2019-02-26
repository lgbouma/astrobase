#!/usr/bin/env python
# -*- coding: utf-8 -*-
# spdm.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''
Contains the Stellingwerf (1978) phase-dispersion minimization period-search
algorithm implementation for periodbase.

'''

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

from multiprocessing import Pool, cpu_count

from numpy import (
    nan as npnan, arange as nparange, array as nparray, isfinite as npisfinite,
    digitize as npdigitize, median as npmedian, std as npstd,
    argsort as npargsort, unique as npunique, sum as npsum, var as npvar,
    argmin as npargmin
)


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import phase_magseries, sigclip_magseries
from . import get_frequency_grid


############
## CONFIG ##
############

NCPUS = cpu_count()


####################################################################
## PHASE DISPERSION MINIMIZATION (Stellingwerf+ 1978, 2011, 2013) ##
####################################################################

def stellingwerf_pdm_theta(times, mags, errs, frequency,
                           binsize=0.05, minbin=9):
    '''
    This calculates the Stellingwerf PDM theta value at a test frequency.

    Parameters
    ----------

    times,mags,errs : np.array
        The input time-series and associated errors.

    frequency : float
        The test frequency to calculate the theta statistic at.

    binsize : float
        The phase bin size to use.

    minbin : int
        The minimum number of items in a phase bin to consider in the
        calculation of the statistic.

    Returns
    -------

    theta_pdm : float
        The value of the theta statistic at the specified `frequency`.


    '''

    period = 1.0/frequency
    fold_time = times[0]

    phased = phase_magseries(times,
                             mags,
                             period,
                             fold_time,
                             wrap=False,
                             sort=True)

    phases = phased['phase']
    pmags = phased['mags']
    bins = nparange(0.0, 1.0, binsize)

    binnedphaseinds = npdigitize(phases, bins)

    binvariances = []
    binndets = []
    goodbins = 0

    for x in npunique(binnedphaseinds):

        thisbin_inds = binnedphaseinds == x
        thisbin_mags = pmags[thisbin_inds]

        if thisbin_mags.size > minbin:
            thisbin_variance = npvar(thisbin_mags,ddof=1)
            binvariances.append(thisbin_variance)
            binndets.append(thisbin_mags.size)
            goodbins = goodbins + 1

    # now calculate theta
    binvariances = nparray(binvariances)
    binndets = nparray(binndets)

    theta_top = npsum(binvariances*(binndets - 1)) / (npsum(binndets) -
                                                      goodbins)
    theta_bot = npvar(pmags,ddof=1)
    theta = theta_top/theta_bot

    return theta



def _stellingwerf_pdm_worker(task):
    '''
    This is a parallel worker for the function below.

    Parameters
    ----------

    task : tuple
        This is of the form below::

            task[0] = times
            task[1] = mags
            task[2] = errs
            task[3] = frequency
            task[4] = binsize
            task[5] = minbin

    Returns
    -------

    theta_pdm : float
        The theta value at the specified frequency. nan if the calculation
        fails.

    '''

    times, mags, errs, frequency, binsize, minbin = task

    try:

        theta = stellingwerf_pdm_theta(times, mags, errs, frequency,
                                       binsize=binsize, minbin=minbin)

        return theta

    except Exception as e:

        return npnan



def stellingwerf_pdm(times,
                     mags,
                     errs,
                     magsarefluxes=False,
                     startp=None,
                     endp=None,
                     stepsize=1.0e-4,
                     autofreq=True,
                     normalize=False,
                     phasebinsize=0.05,
                     mindetperbin=9,
                     nbestpeaks=5,
                     periodepsilon=0.1,
                     sigclip=10.0,
                     nworkers=None,
                     verbose=True):

    '''This runs a parallelized Stellingwerf phase-dispersion minimization (PDM)
    period search.

    Parameters
    ----------

    times,mags,errs : np.array
        The mag/flux time-series with associated measurement errors to run the
        period-finding on.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    startp,endp : float or None
        The minimum and maximum periods to consider for the transit search.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    autofreq : bool
        If this is True, the value of `stepsize` will be ignored and the
        :py:func:`astrobase.periodbase.get_frequency_grid` function will be used
        to generate a frequency grid based on `startp`, and `endp`. If these are
        None as well, `startp` will be set to 0.1 and `endp` will be set to
        `times.max() - times.min()`.

    normalize : bool
        This sets if the input time-series is normalized to 0.0 and rescaled
        such that its variance = 1.0. This is the recommended procedure by
        Schwarzenberg-Czerny 1996.

    phasebinsize : float
        The bin size in phase to use when calculating the PDM theta statistic at
        a test frequency.

    mindetperbin : int
        The minimum number of elements in a phase bin to consider it valid when
        calculating the PDM theta statistic at a test frequency.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    nworkers : int
        The number of parallel workers to use when calculating the periodogram.

    verbose : bool
        If this is True, will indicate progress and details about the frequency
        grid used for the period search.

    Returns
    -------

    dict
        This function returns a dict, referred to as an `lspinfo` dict in other
        astrobase functions that operate on periodogram results. This is a
        standardized format across all astrobase period-finders, and is of the
        form below::

            {'bestperiod': the best period value in the periodogram,
             'bestlspval': the periodogram peak associated with the best period,
             'nbestpeaks': the input value of nbestpeaks,
             'nbestlspvals': nbestpeaks-size list of best period peak values,
             'nbestperiods': nbestpeaks-size list of best periods,
             'lspvals': the full array of periodogram powers,
             'periods': the full array of periods considered,
             'method':'pdm' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # get the frequencies to use
        if startp:
            endf = 1.0/startp
        else:
            # default start period is 0.1 day
            endf = 1.0/0.1

        if endp:
            startf = 1.0/endp
        else:
            # default end period is length of time series
            startf = 1.0/(stimes.max() - stimes.min())

        # if we're not using autofreq, then use the provided frequencies
        if not autofreq:
            frequencies = nparange(startf, endf, stepsize)
            if verbose:
                LOGINFO(
                    'using %s frequency points, start P = %.3f, end P = %.3f' %
                    (frequencies.size, 1.0/endf, 1.0/startf)
                )
        else:
            # this gets an automatic grid of frequencies to use
            frequencies = get_frequency_grid(stimes,
                                             minfreq=startf,
                                             maxfreq=endf)
            if verbose:
                LOGINFO(
                    'using autofreq with %s frequency points, '
                    'start P = %.3f, end P = %.3f' %
                    (frequencies.size,
                     1.0/frequencies.max(),
                     1.0/frequencies.min())
                )

        # map to parallel workers
        if (not nworkers) or (nworkers > NCPUS):
            nworkers = NCPUS
            if verbose:
                LOGINFO('using %s workers...' % nworkers)

        pool = Pool(nworkers)

        # renormalize the working mags to zero and scale them so that the
        # variance = 1 for use with our LSP functions
        if normalize:
            nmags = (smags - npmedian(smags))/npstd(smags)
        else:
            nmags = smags

        tasks = [(stimes, nmags, serrs, x, phasebinsize, mindetperbin)
                 for x in frequencies]

        lsp = pool.map(_stellingwerf_pdm_worker, tasks)

        pool.close()
        pool.join()
        del pool

        lsp = nparray(lsp)
        periods = 1.0/frequencies

        # find the nbestpeaks for the periodogram: 1. sort the lsp array by
        # lowest value first 2. go down the values until we find five values
        # that are separated by at least periodepsilon in period

        # make sure to filter out the non-finite values of lsp
        finitepeakind = npisfinite(lsp)
        finlsp = lsp[finitepeakind]
        finperiods = periods[finitepeakind]

        # finlsp might not have any finite values if the period finding
        # failed. if so, argmin will return a ValueError.
        try:

            bestperiodind = npargmin(finlsp)

        except ValueError:

            LOGERROR('no finite periodogram values for '
                     'this mag series, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'pdm',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'normalize':normalize,
                              'phasebinsize':phasebinsize,
                              'mindetperbin':mindetperbin,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip}}

        sortedlspind = npargsort(finlsp)
        sortedlspperiods = finperiods[sortedlspind]
        sortedlspvals = finlsp[sortedlspind]

        # now get the nbestpeaks
        nbestperiods, nbestlspvals, peakcount = (
            [finperiods[bestperiodind]],
            [finlsp[bestperiodind]],
            1
        )
        prevperiod = sortedlspperiods[0]

        # find the best nbestpeaks in the lsp and their periods
        for period, lspval in zip(sortedlspperiods, sortedlspvals):

            if peakcount == nbestpeaks:
                break
            perioddiff = abs(period - prevperiod)
            bestperiodsdiff = [abs(period - x) for x in nbestperiods]

            # print('prevperiod = %s, thisperiod = %s, '
            #       'perioddiff = %s, peakcount = %s' %
            #       (prevperiod, period, perioddiff, peakcount))

            # this ensures that this period is different from the last
            # period and from all the other existing best periods by
            # periodepsilon to make sure we jump to an entire different peak
            # in the periodogram
            if (perioddiff > (periodepsilon*prevperiod) and
                all(x > (periodepsilon*prevperiod) for x in bestperiodsdiff)):
                nbestperiods.append(period)
                nbestlspvals.append(lspval)
                peakcount = peakcount + 1

            prevperiod = period


        return {'bestperiod':finperiods[bestperiodind],
                'bestlspval':finlsp[bestperiodind],
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':nbestlspvals,
                'nbestperiods':nbestperiods,
                'lspvals':lsp,
                'periods':periods,
                'method':'pdm',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'phasebinsize':phasebinsize,
                          'mindetperbin':mindetperbin,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}

    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'pdm',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'phasebinsize':phasebinsize,
                          'mindetperbin':mindetperbin,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}
