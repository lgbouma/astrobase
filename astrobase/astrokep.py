#!/usr/bin/env python

'''astrokep.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 05/2016

Contains various useful tools for analyzing Kepler light curves.

'''


import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime

import os.path
try:
    import cPickle as pickle
except:
    import pickle

import gzip

from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, all as npall, \
    correlate as npcorrelate, zeros as npzeros, ones as npones, \
    column_stack as npcolumn_stack, concatenate as npconcatenate

from scipy.optimize import leastsq
from scipy.signal import medfilt

from sklearn.ensemble import RandomForestRegressor

import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

try:
    import pyfits
except:
    from astropy.io import fits as pyfits

###################
## LOCAL IMPORTS ##
###################



#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.kepler' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


###########################################
## UTILITY FUNCTIONS FOR FLUXES AND MAGS ##
###########################################


def keplerflux_to_keplermag(keplerflux, f12=1.74e5):
    '''
    This converts the kepler flux in electrons/sec to kepler magnitude.

    kepler mag/flux relation:
    - fkep = (10.0**(-0.4*(kepmag - 12.0)))*f12
    - f12 = 1.74e5 # electrons/sec

    '''

    kepmag = 12.0 - 2.5*nplog10(keplerflux/f12)
    return kepmag


def keplermag_to_keplerflux(keplermag, f12=1.74e5):
    '''
    This converts the kepler mag back to kepler flux.

    '''

    kepflux = (10.0**(-0.4*(keplermag - 12.0)))*f12
    return kepflux


def keplermag_to_sdssr(keplermag, kic_sdssg, kic_sdssr):
    '''

    convert from kepmag to SDSS r mag, we must know the sdssg of the target
    (from UCAC4 or other transforms). this appears to be a very rough
    transformation.

    Get kic_sdssg and kic_sdssr from extension 0 of a Kepler llc.fits file.

    '''
    kic_sdssgr = kic_sdssg - kic_sdssr

    if kic_sdssgr < 0.8:
        kepsdssr = (keplermag - 0.2*kic_sdssg)/0.8
    else:
        kepsdssr = (keplermag - 0.1*kic_sdssg)/0.9
    return kepsdssr


def flux_ppm_to_magnitudes(ppm):
    '''
    This converts Kepler's flux parts-per-million to magnitudes.

    '''
    return -2.5*nplog10(1.0 - ppm/1.0e6)



######################################################
## FUNCTIONS FOR READING KEPLER AND K2 LIGHT CURVES ##
######################################################

# this is the list of keys to pull out of the light curve FITS table
LCDATAKEYS = ['TIME','TIMECORR','CADENCENO',
              'SAP_QUALITY',
              'PSF_CENTR1','PSF_CENTR1_ERR','PSF_CENTR2','PSF_CENTR2_ERR',
              'MOM_CENTR1','MOM_CENTR1_ERR','MOM_CENTR2','MOM_CENTR2_ERR']

LCSAPKEYS = ['SAP_FLUX','SAP_FLUX_ERR','SAP_BKG','SAP_BKG_ERR']
LCPDCKEYS = ['PDCSAP_FLUX','PDCSAP_FLUX_ERR']

# this is the list of keys to pull out of the light curve header
LCHEADERKEYS = ['TIMESYS','BJDREFI','BJDREFF',
                'OBJECT','KEPLERID',
                'RA_OBJ','DEC_OBJ','EQUINOX',
                'EXPOSURE',
                'CDPP3_0','CDPP6_0','CDPP12_0',
                'PDCVAR','PDCMETHD','CROWDSAP','FLFRCSAP']

# this is the list of keys to pull out of the top header of the FITS
LCTOPKEYS = ['CHANNEL','SKYGROUP','MODULE','OUTPUT',
             'QUARTER','SEASON','DATA_REL','OBSMODE',
             'PMRA','PMDEC','PMTOTAL','PARALLAX',
             'GLON','GLAT',
             'GMAG','RMAG','IMAG','ZMAG','D51MAG',
             'JMAG','HMAG','KMAG','KEPMAG',
             'GRCOLOR','JKCOLOR','GKCOLOR',
             'TEFF','LOGG','FEH',
             'EBMINUSV','AV','RADIUS','TMINDEX']

# this is the list of keys to pull out of the aperture part of the light curve
# we also pull out the whole pixel mask, which looks something like:
# array([[0, 1, 1, 1, 1, 1, 1, 0],
#        [1, 1, 1, 3, 3, 1, 1, 1],
#        [1, 1, 3, 3, 3, 3, 1, 1],
#        [1, 1, 3, 3, 3, 3, 3, 1],
#        [1, 1, 3, 3, 3, 3, 3, 1],
#        [1, 1, 1, 1, 3, 3, 1, 1],
#        [0, 1, 1, 1, 1, 1, 1, 0]], dtype=int32)
# where the value 3 means the actual pixels used to sum the flux for this
# particular object (the optimal aperture). 1 means the pixel was collected by
# the telescope, so its flux is available
# we use CDELT1 and CDELT2 below to get the pixel scale in arcsec/px
# it should be about 3.96 arcsec/pixel in most cases
LCAPERTUREKEYS = ['NPIXSAP','NPIXMISS','CDELT1','CDELT2']


def read_kepler_fitslc(lcfits,
                       headerkeys=LCHEADERKEYS,
                       datakeys=LCDATAKEYS,
                       sapkeys=LCSAPKEYS,
                       pdckeys=LCPDCKEYS,
                       topkeys=LCTOPKEYS,
                       apkeys=LCAPERTUREKEYS,
                       appendto=None):
    '''This extracts the light curve from a single Kepler prime LC FITS file.

    Returns an lcdict.

    If appendto is an lcdict, will append measurements to that dict. This is
    used for consolidating light curves across different files. Will sort
    measurements in date order.

    '''

    # read the fits file
    hdulist = pyfits.open(lcfits)
    lchdr, lcdata = hdulist[1].header, hdulist[1].data
    lctophdr, lcaperturehdr, lcaperturedata = (hdulist[0].header,
                                               hdulist[2].header,
                                               hdulist[2].data)
    hdulist.close()

    hdrinfo = {}

    # now get the values we want from the header
    for key in headerkeys:
        if key in lchdr and lchdr[key] is not None:
            hdrinfo[key.lower()] = lchdr[key]
        else:
            hdrinfo[key.lower()] = None

    # get the number of detections
    ndet = lchdr['NAXIS2']

    # get the info from the topheader
    for key in topkeys:
        if key in lctophdr and lctophdr[key] is not None:
            hdrinfo[key.lower()] = lctophdr[key]
        else:
            hdrinfo[key.lower()] = None

    # get the info from the lcaperturehdr
    for key in lcaperturehdr:
        if key in lcaperturehdr and lcaperturehdr[key] is not None:
            hdrinfo[key.lower()] = lcaperturehdr[key]
        else:
            hdrinfo[key.lower()] = None


    # if we're appending to another lcdict
    if appendto and isinstance(appendto, dict):

        lcdict = appendto

        lcdict['quarter'].append(hdrinfo['quarter'])
        lcdict['season'].append(hdrinfo['season'])
        lcdict['datarelease'].append(hdrinfo['data_rel'])
        lcdict['obsmode'].append(hdrinfo['obsmode'])
        # we don't update the objectid

        # update lcinfo
        lcdict['lcinfo']['timesys'].append(hdrinfo['timesys'])
        lcdict['lcinfo']['bjdoffset'].append(
            hdrinfo['bjdrefi'] + hdrinfo['bjdreff']
        )
        lcdict['lcinfo']['exptime'].append(hdrinfo['exposure'])
        lcdict['lcinfo']['lcaperture'].append(lcaperturedata)
        lcdict['lcinfo']['aperpixused'].append(hdrinfo['npixsap'])
        lcdict['lcinfo']['aperpixunused'].append(hdrinfo['npixmiss'])
        lcdict['lcinfo']['pixarcsec'].append(
            (npabs(hdrinfo['cdelt1']) +
             npabs(hdrinfo['cdelt2']))*3600.0/2.0
        )
        lcdict['lcinfo']['channel'].append(hdrinfo['channel'])
        lcdict['lcinfo']['skygroup'].append(hdrinfo['skygroup'])
        lcdict['lcinfo']['module'].append(hdrinfo['module'])
        lcdict['lcinfo']['output'].append(hdrinfo['output'])
        lcdict['lcinfo']['ndet'].append(ndet)

        # the objectinfo is not updated for the same object when appending to a
        # light curve. FIXME: maybe it should be?

        # update the varinfo for this light curve
        lcdict['varinfo']['cdpp3_0'].append(hdrinfo['cdpp3_0'])
        lcdict['varinfo']['cdpp6_0'].append(hdrinfo['cdpp6_0'])
        lcdict['varinfo']['cdpp12_0'].append(hdrinfo['cdpp12_0'])
        lcdict['varinfo']['pdcvar'].append(hdrinfo['pdcvar'])
        lcdict['varinfo']['pdcmethod'].append(hdrinfo['pdcmethd'])
        lcdict['varinfo']['aper_target_total_ratio'].append(hdrinfo['crowdsap'])
        lcdict['varinfo']['aper_target_frac'].append(hdrinfo['flfrcsap'])

        # update the light curve columns now
        for key in datakeys:
            if key.lower() in lcdict:
                lcdict[key.lower()] = (
                    npconcatenate((lcdict[key.lower()], lcdata[key]))
                )

        for key in sapkeys:
            if key.lower() in lcdict['sap']:
                lcdict['sap'][key.lower()] = (
                    npconcatenate((lcdict['sap'][key.lower()], lcdata[key]))
                )

        for key in pdckeys:
            if key.lower() in lcdict['pdc']:
                lcdict['pdc'][key.lower()] = (
                    npconcatenate((lcdict['pdc'][key.lower()], lcdata[key]))
                )


        # append some of the light curve information into existing numpy arrays
        # so we can sort on them later
        lcdict['lc_channel'] = npconcatenate(
            (lcdict['lc_channel'],
             npfull_like(lcdata['TIME'],
                         hdrinfo['channel']))
        )
        lcdict['lc_skygroup'] = npconcatenate(
            (lcdict['lc_skygroup'],
             npfull_like(lcdata['TIME'],
                         hdrinfo['skygroup']))
        )
        lcdict['lc_module'] = npconcatenate(
            (lcdict['lc_module'],
             npfull_like(lcdata['TIME'],
                         hdrinfo['module']))
        )
        lcdict['lc_output'] = npconcatenate(
            (lcdict['lc_output'],
             npfull_like(lcdata['TIME'],
                         hdrinfo['output']))
        )
        lcdict['lc_quarter'] = npconcatenate(
            (lcdict['lc_quarter'],
             npfull_like(lcdata['TIME'],
                         hdrinfo['quarter']))
        )
        lcdict['lc_season'] = npconcatenate(
            (lcdict['lc_season'],
             npfull_like(lcdata['TIME'],
                         hdrinfo['season']))
        )


    # otherwise, this is a new lcdict
    else:

        # form the lcdict
        # the metadata is one-elem arrays because we might add on to them later
        lcdict = {
            'quarter':[hdrinfo['quarter']],
            'season':[hdrinfo['season']],
            'datarelease':[hdrinfo['data_rel']],
            'obsmode':[hdrinfo['obsmode']],
            'objectid':hdrinfo['object'],
            'lcinfo':{
                'timesys':[hdrinfo['timesys']],
                'bjdoffset':[hdrinfo['bjdrefi'] + hdrinfo['bjdreff']],
                'exptime':[hdrinfo['exposure']],
                'lcaperture':[lcaperturedata],
                'aperpixused':[hdrinfo['npixsap']],
                'aperpixunused':[hdrinfo['npixmiss']],
                'pixarcsec':[(npabs(hdrinfo['cdelt1']) +
                             npabs(hdrinfo['cdelt2']))*3600.0/2.0],
                'channel':[hdrinfo['channel']],
                'skygroup':[hdrinfo['skygroup']],
                'module':[hdrinfo['module']],
                'output':[hdrinfo['output']],
                'ndet':[ndet],
            },
            'objectinfo':{
                'objectid':hdrinfo['object'], # repeated here for checkplot use
                'keplerid':hdrinfo['keplerid'],
                'ra':hdrinfo['ra_obj'],
                'decl':hdrinfo['dec_obj'],
                'pmra':hdrinfo['pmra'],
                'pmdecl':hdrinfo['pmdec'],
                'pmtotal':hdrinfo['pmtotal'],
                'sdssg':hdrinfo['gmag'],
                'sdssr':hdrinfo['rmag'],
                'sdssi':hdrinfo['imag'],
                'sdssz':hdrinfo['zmag'],
                'kepmag':hdrinfo['kepmag'],
                'teff':hdrinfo['teff'],
                'logg':hdrinfo['logg'],
                'feh':hdrinfo['feh'],
                'ebminusv':hdrinfo['ebminusv'],
                'extinction':hdrinfo['av'],
                'starradius':hdrinfo['radius'],
                'twomassuid':hdrinfo['tmindex'],
            },
            'varinfo':{
                'cdpp3_0':[hdrinfo['cdpp3_0']],
                'cdpp6_0':[hdrinfo['cdpp6_0']],
                'cdpp12_0':[hdrinfo['cdpp12_0']],
                'pdcvar':[hdrinfo['pdcvar']],
                'pdcmethod':[hdrinfo['pdcmethd']],
                'aper_target_total_ratio':[hdrinfo['crowdsap']],
                'aper_target_frac':[hdrinfo['flfrcsap']],
            },
            'sap':{},
            'pdc':{},
        }

        # get the LC columns
        for key in datakeys:
            lcdict[key.lower()] = lcdata[key]
        for key in sapkeys:
            lcdict['sap'][key.lower()] = lcdata[key]
        for key in pdckeys:
            lcdict['pdc'][key.lower()] = lcdata[key]

        # turn some of the light curve information into numpy arrays so we can
        # sort on them later
        lcdict['lc_channel'] = npfull_like(lcdict['time'],
                                           lcdict['lcinfo']['channel'][0])
        lcdict['lc_skygroup'] = npfull_like(lcdict['time'],
                                            lcdict['lcinfo']['skygroup'][0])
        lcdict['lc_module'] = npfull_like(lcdict['time'],
                                          lcdict['lcinfo']['module'][0])
        lcdict['lc_output'] = npfull_like(lcdict['time'],
                                          lcdict['lcinfo']['output'][0])
        lcdict['lc_quarter'] = npfull_like(lcdict['time'],
                                           lcdict['quarter'][0])
        lcdict['lc_season'] = npfull_like(lcdict['time'],
                                          lcdict['season'][0])

    ## END OF LIGHT CURVE CONSTRUCTION ##


    # update the lcdict columns with the actual columns
    lcdict['columns'] = (
        [x.lower() for x in datakeys] +
        ['sap.%s' % x.lower() for x in sapkeys] +
        ['pdc.%s' % x.lower() for x in pdckeys] +
        ['lc_channel','lc_skygroup','lc_module',
         'lc_output','lc_quarter','lc_season']
    )

    # return the lcdict at the end
    return lcdict



def consolidate_kepler_fitslc(keplerid, lcfitsdir,
                              headerkeys=LCHEADERKEYS,
                              datakeys=LCDATAKEYS,
                              sapkeys=LCSAPKEYS,
                              pdckeys=LCPDCKEYS,
                              topkeys=LCTOPKEYS,
                              apkeys=LCAPERTUREKEYS):
    '''This gets all light curves for the given keplerid in lcfitsdir.

    Sorts the light curves by time. Returns an lcdict. This is meant to be used
    for light curves across quarters.

    Searches recursively in lcfitsdir for all of the files belonging to the
    specified keplerid.

    '''

    # use the os.walk function to start looking for files in lcfitsdir
    # FIXME: maybe we should shell out to `find` instead?
    walker = os.walk(lcfitsdir)
    matching = []
    LOGINFO('looking for Kepler light curve FITS in %s for %s...' % (lcfitsdir,
                                                                     keplerid))
    for root, dirs, files in walker:
        for sdir in dirs:
            searchpath = os.path.join(root,
                                      sdir,
                                      'kplr%09i-*_llc.fits' % keplerid)
            foundfiles = glob.glob(searchpath)

            if foundfiles:
                matching.extend(foundfiles)
                LOGINFO('found %s in dir: %s' % (repr(foundfiles),
                                                 os.path.join(root,sdir)))

    # now that we've found everything, read them all in
    if len(matching) > 0:

        LOGINFO('consolidating...')

        # the first file
        consolidated = read_kepler_fitslc(matching[0],
                                          headerkeys=headerkeys,
                                          datakeys=datakeys,
                                          sapkeys=sapkeys,
                                          pdckeys=pdckeys,
                                          topkeys=topkeys,
                                          apkeys=apkeys)
        # get the rest of the files
        for lcf in matching:
            consolidated = read_kepler_fitslc(lcf,
                                              appendto=consolidated,
                                              headerkeys=headerkeys,
                                              datakeys=datakeys,
                                              sapkeys=sapkeys,
                                              pdckeys=pdckeys,
                                              topkeys=topkeys,
                                              apkeys=apkeys)

        # get the sort indices
        # we use time for the columns and quarters for the headers
        LOGINFO('sorting by time...')

        column_sort_ind = npargsort(consolidated['time'])

        # sort the columns by time
        for col in consolidated['columns']:
            if '.' in col:
                key, subkey = col.split('.')
                consolidated[key][subkey] = (
                    consolidated[key][subkey][column_sort_ind]
                )
            else:
                consolidated[col] = consolidated[col][column_sort_ind]

        # now sort the headers by quarters
        header_sort_ind = npargsort(consolidated['quarter']).tolist()

        # this is a bit convoluted, but whatever: list -> array -> list

        for key in ('quarter', 'season', 'datarelease', 'obsmode'):
            consolidated[key] = (
                np.array(consolidated[key])[header_sort_ind].tolist()
            )

        for key in ('timesys','bjdoffset','exptime','lcaperture',
                    'aperpixused','aperpixunused','pixarcsec',
                    'channel','skygroup','module','output','ndet'):
            consolidated['lcinfo'][key] = (
                np.array(consolidated['lcinfo'][key])[header_sort_ind].tolist()
            )

        for key in ('cdpp3_0','cdpp6_0','cdpp12_0','pdcvar','pdcmethod',
                    'aper_target_total_ratio','aper_target_frac'):
            consolidated['varinfo'][key] = (
                np.array(consolidated['varinfo'][key])[header_sort_ind].tolist()
            )

        # finally, return the consolidated lcdict
        return consolidated

    # if we didn't find anything, complain
    else:

        LOGERROR('could not find any light curves '
                 'for %s in %s or its subdirectories' % (keplerid,
                                                         lcfitsdir))
        return None


########################
## READING K2 SFF LCs ##
########################

SFFTOPKEYS = LCTOPKEYS + ['CAMPAIGN']
SFFHEADERKEYS = LCHEADERKEYS + ['MASKTYPE','MASKINDE','NPIXSAP']
SFFDATAKEYS = ['T','FRAW','FCOR','ARCLENGTH','MOVING','CADENCENO']


def read_k2sff_lightcurve(lcfits):
    '''
    This reads a K2 SFF (Vandenberg+ 2014) light curve into an lcdict.

    '''

    # read the fits file
    hdulist = pyfits.open(lcfits)
    lchdr, lcdata = hdulist[1].header, hdulist[1].data
    lctophdr = hdulist[0].header

    hdulist.close()

    hdrinfo = {}

    # get the number of detections
    ndet = lchdr['NAXIS2']

    # get the info from the topheader
    for key in SFFTOPKEYS:
        if key in lctophdr and lctophdr[key] is not None:
            hdrinfo[key.lower()] = lctophdr[key]
        else:
            hdrinfo[key.lower()] = None

    # now get the values we want from the header
    for key in SFFHEADERKEYS:
        if key in lchdr and lchdr[key] is not None:
            hdrinfo[key.lower()] = lchdr[key]
        else:
            hdrinfo[key.lower()] = None

    # form the lcdict
    # the metadata is one-elem arrays because we might add on to them later
    lcdict = {
        'quarter':[hdrinfo['quarter']],
        'season':[hdrinfo['season']],
        'datarelease':[hdrinfo['data_rel']],
        'obsmode':[hdrinfo['obsmode']],
        'objectid':hdrinfo['object'],
        'campaign':[hdrinfo['campaign']],
        'lcinfo':{
            'timesys':[hdrinfo['timesys']],
            'bjdoffset':[hdrinfo['bjdrefi'] + hdrinfo['bjdreff']],
            'exptime':[hdrinfo['exposure']],
            'lcapermaskidx':[hdrinfo['maskinde']],
            'lcapermasktype':[hdrinfo['masktype']],
            'aperpixused':[hdrinfo['npixsap']],
            'aperpixunused':[None],
            'pixarcsec':[None],
            'channel':[hdrinfo['channel']],
            'skygroup':[hdrinfo['skygroup']],
            'module':[hdrinfo['module']],
            'output':[hdrinfo['output']],
            'ndet':[ndet],
        },
        'objectinfo':{
            'keplerid':hdrinfo['keplerid'],
            'ra':hdrinfo['ra_obj'],
            'decl':hdrinfo['dec_obj'],
            'pmra':hdrinfo['pmra'],
            'pmdecl':hdrinfo['pmdec'],
            'pmtotal':hdrinfo['pmtotal'],
            'sdssg':hdrinfo['gmag'],
            'sdssr':hdrinfo['rmag'],
            'sdssi':hdrinfo['imag'],
            'sdssz':hdrinfo['zmag'],
            'kepmag':hdrinfo['kepmag'],
            'teff':hdrinfo['teff'],
            'logg':hdrinfo['logg'],
            'feh':hdrinfo['feh'],
            'ebminusv':hdrinfo['ebminusv'],
            'extinction':hdrinfo['av'],
            'starradius':hdrinfo['radius'],
            'twomassuid':hdrinfo['tmindex'],
        },
        'varinfo':{
            'cdpp3_0':[hdrinfo['cdpp3_0']],
            'cdpp6_0':[hdrinfo['cdpp6_0']],
            'cdpp12_0':[hdrinfo['cdpp12_0']],
            'pdcvar':[hdrinfo['pdcvar']],
            'pdcmethod':[hdrinfo['pdcmethd']],
            'aptgttotrat':[hdrinfo['crowdsap']],
            'aptgtfrac':[hdrinfo['flfrcsap']],
        },
    }

    # get the LC columns
    for key in SFFDATAKEYS:
        lcdict[key.lower()] = lcdata[key]

    # add some of the light curve information to the data arrays so we can sort
    # on them later
    lcdict['channel'] = npfull_like(lcdict['t'],
                                     lcdict['lcinfo']['channel'][0])
    lcdict['skygroup'] = npfull_like(lcdict['t'],
                                     lcdict['lcinfo']['skygroup'][0])
    lcdict['module'] = npfull_like(lcdict['t'],
                                     lcdict['lcinfo']['module'][0])
    lcdict['output'] = npfull_like(lcdict['t'],
                                     lcdict['lcinfo']['output'][0])
    lcdict['quarter'] = npfull_like(lcdict['t'],
                                     lcdict['quarter'][0])
    lcdict['season'] = npfull_like(lcdict['t'],
                                     lcdict['season'][0])
    lcdict['campaign'] = npfull_like(lcdict['t'],
                                     lcdict['campaign'][0])

    # update the lcdict columns with the actual columns
    lcdict['columns'] = (
        [x.lower() for x in SFFDATAKEYS] +
        ['channel','skygroup','module','output','quarter','season','campaign']
    )

    # return the lcdict at the end
    return lcdict



##################
## INPUT/OUTPUT ##
##################

def kepler_lcdict_to_pkl(lcdict, outfile=None):
    '''
    This simply writes the lcdict to a gzipped pickle.

    '''

    if not outfile:
        outfile = 'EPIC%s-keplc.pkl.gz' % lcdict['objectinfo']['keplerid']

    with gzip.open(outfile,'wb') as outfd:
        pickle.dump(lcdict, outfd, protocol=2)

    return os.path.abspath(outfile)



def read_kepler_pklc(picklefile):
    '''
    This turns the gzipped pickled lightcurve back into an lcdict.

    '''

    with gzip.open(picklefile,'rb') as infd:
        lcdict = pickle.load(infd)

    return lcdict



##########################
## KEPLER LC PROCESSING ##
##########################


def find_lightcurve_gaps(times, mags, errs, mindmagdt):
    '''This finds gaps in the light curves.

    Gets the first-derivative of the light curve per point, and compares that to
    mindmagdt. If it's larger than that, then designates that as a new segment
    of the LC.

    '''


def stitch_lightcurve_gaps(times, mags, errs, mindmagdt):
    '''
    This stitches light curve gaps together.

    '''



def filter_kepler_lcdict(lcdict,
                         filterflags=True,
                         nanfilter='sap,pdc',
                         timestoignore=None):
    '''This filters the Kepler light curve dict.

    By default, this function removes points in the Kepler LC that have ANY
    quality flags set. Also removes nans.

    timestoignore is a list of tuples containing start and end times to mask:

    [(time1_start, time1_end), (time2_start, time2_end), ...]

    This function filters the dict IN PLACE!

    '''

    cols = lcdict['columns']

    # filter all bad LC points as noted by quality flags
    if filterflags:

        nbefore = lcdict['time'].size
        filterind = lcdict['sap_quality'] == 0

        for col in cols:
            if '.' in col:
                key, subkey = col.split('.')
                lcdict[key][subkey] = lcdict[key][subkey][filterind]
            else:
                lcdict[col] = lcdict[col][filterind]

        nafter = lcdict['time'].size
        LOGINFO('applied quality flag filter, ndet before = %s, ndet after = %s'
                % (nbefore, nafter))


    if nanfilter and nanfilter == 'sap,pdc':
        notnanind = (
            npisfinite(lcdict['sap']['sap_flux']) &
            npisfinite(lcdict['pdc']['pdcsap_flux'])
        )
    elif nanfilter and nanfilter == 'sap':
        notnanind = npisfinite(lcdict['sap']['sap_flux'])
    elif nanfilter and nanfilter == 'pdc':
        notnanind = npisfinite(lcdict['pdc']['pdcsap_flux'])


    # remove nans from all columns
    if nanfilter:

        nbefore = lcdict['time'].size
        for col in cols:
            if '.' in col:
                key, subkey = col.split('.')
                lcdict[key][subkey] = lcdict[key][subkey][notnanind]
            else:
                lcdict[col] = lcdict[col][notnanind]

        nafter = lcdict['time'].size

        LOGINFO('removed nans, ndet before = %s, ndet after = %s'
                % (nbefore, nafter))


    # exclude all times in timestoignore
    if (timestoignore and
        isinstance(timestoignore, list) and
        len(timestoignore) > 0):

        exclind = npfull_like(lcdict['time'],True)
        nbefore = exclind.size

        # get all the masks
        for ignoretime in timestoignore:
            time0, time1 = ignoretime[0], ignoretime[1]
            thismask = (lcdict['time'] > time0) & (lcdict['time'] < time1)
            exclind = exclind & thismask

        # apply the masks
        for col in cols:
            lcdict[col] = lcdict[col][exclind]

        nafter = lcdict['time'].size
        LOGINFO('removed timestoignore, ndet before = %s, ndet after = %s'
                % (nbefore, nafter))



###################
## KEPLER LC EPD ##
###################

def _epd_function(coeffs, fluxes, xcc, ycc, bgv, bge):
    '''
    This is the EPD function to fit.

    '''

    epdf = (
        coeffs[0] +
        coeffs[1]*npsin(2*MPI*xcc) + coeffs[2]*npcos(2*MPI*xcc) +
        coeffs[3]*npsin(2*MPI*ycc) + coeffs[4]*npcos(2*MPI*ycc) +
        coeffs[5]*npsin(4*MPI*xcc) + coeffs[6]*npcos(4*MPI*xcc) +
        coeffs[7]*npsin(4*MPI*ycc) + coeffs[8]*npcos(4*MPI*ycc) +
        coeffs[9]*bgv +
        coeffs[10]*bge
    )

    return epdf



def _epd_residual(coeffs, fluxes, xcc, ycc, bgv, bge):
    '''
    This is the residual function to minimize using scipy.optimize.leastsq.

    '''

    f = _epd_function(coeffs, fluxes, xcc, ycc, bgv, bge)
    residual = fluxes - f
    return residual



def epd_kepler_lightcurve(lcdict,
                          xccol='mom_centr1',
                          yccol='mom_centr2',
                          timestoignore=None,
                          filterflags=True,
                          writetodict=True,
                          epdsmooth=5):
    '''This runs EPD on the Kepler light curve.

    Following Huang et al. 2015, we fit and subtract the following EPD function:

    f = c0 +
        c1*sin(2*pi*x) + c2*cos(2*pi*x) + c3*sin(2*pi*y) + c4*cos(2*pi*y) +
        c5*sin(4*pi*x) + c6*cos(4*pi*x) + c7*sin(4*pi*y) + c8*cos(4*pi*y) +
        c9*bgv + c10*bge

    timestoignore is a list of tuples containing start and end times to mask
    when fitting the EPD function:

    [(time1_start, time1_end), (time2_start, time2_end), ...]

    NOTES:

    - this function returns times and mags by default
    - by default, this function removes points in the Kepler LC that have ANY
      quality flags set

    if writetodict is set, adds the following columns to the lcdict:

    epd_time = time array
    epd_sapflux = uncorrected flux before EPD
    epd_epdsapflux = corrected flux after EPD
    epd_epdsapcorr = EPD flux corrections
    epd_bkg = background array
    epd_bkg_err = background errors array
    epd_xcc = xcoord array
    epd_ycc = ycoord array
    epd_quality = quality flag array

    and updates the 'columns' list in the lcdict as well.

    '''

    times, fluxes, background, background_err = (lcdict['time'],
                                                 lcdict['sap']['sap_flux'],
                                                 lcdict['sap']['sap_bkg'],
                                                 lcdict['sap']['sap_bkg_err'])
    xcc = lcdict[xccol]
    ycc = lcdict[yccol]
    flags = lcdict['sap_quality']

    # filter all bad LC points as noted by quality flags
    if filterflags:

        nbefore = times.size

        filterind = flags == 0

        times = times[filterind]
        fluxes = fluxes[filterind]
        background = background[filterind]
        background_err = background_err[filterind]
        xcc = xcc[filterind]
        ycc = ycc[filterind]
        flags = flags[filterind]

        nafter = times.size
        LOGINFO('applied quality flag filter, ndet before = %s, ndet after = %s'
                % (nbefore, nafter))


    # remove nans
    find = (npisfinite(xcc) & npisfinite(ycc) &
            npisfinite(times) & npisfinite(fluxes) &
            npisfinite(background) & npisfinite(background_err))

    nbefore = times.size

    times = times[find]
    fluxes = fluxes[find]
    background = background[find]
    background_err = background_err[find]
    xcc = xcc[find]
    ycc = ycc[find]
    flags = flags[find]

    nafter = times.size
    LOGINFO('removed nans, ndet before = %s, ndet after = %s'
            % (nbefore, nafter))


    # exclude all times in timestoignore
    if (timestoignore and
        isinstance(timestoignore, list) and
        len(timestoignore) > 0):

        exclind = npfull_like(times,True)

        nefore = times.size

        # apply all the masks
        for ignoretime in timestoignore:
            time0, time1 = ignoretime[0], ignoretime[1]
            thismask = (times > time0) & (times < time1)
            exclind = exclind & thismask

        # quantities after masks have been applied
        times = times[exclind]
        fluxes = fluxes[exclind]
        background = background[exclind]
        background_err = background_err[exclind]
        xcc = xcc[exclind]
        ycc = ycc[exclind]
        flags = flags[exclind]

        nafter = times.size
        LOGINFO('removed timestoignore, ndet before = %s, ndet after = %s'
                % (nbefore, nafter))


    # now that we're all done, we can do EPD
    # first, smooth the light curve
    smoothedfluxes = medfilt(fluxes, epdsmooth)

    # initial fit coeffs
    initcoeffs = npones(11)

    # fit the the smoothed mags and find better coeffs
    leastsqfit = leastsq(_epd_residual,
                         initcoeffs,
                         args=(fluxes, xcc, ycc, background, background_err))

    # if the fit succeeds, then get the EPD fluxes
    if leastsqfit[-1] in (1,2,3,4):

        fitcoeffs = leastsqfit[0]
        epdfit = _epd_function(fitcoeffs,
                               fluxes,
                               xcc,
                               ycc,
                               background,
                               background_err)
        epdfluxes = npmedian(fluxes) + fluxes - epdfit

        # write these to the dictionary if requested
        if writetodict:

            lcdict['epd'] = {}

            lcdict['epd']['time'] = times
            lcdict['epd']['sapflux'] = fluxes
            lcdict['epd']['epdsapflux'] = epdfluxes
            lcdict['epd']['epdsapcorr'] = epdfit
            lcdict['epd']['bkg'] = background
            lcdict['epd']['bkg_err'] = background_err
            lcdict['epd']['xcc'] = xcc
            lcdict['epd']['ycc'] = ycc
            lcdict['epd']['quality'] = flags

            for newcol in ['epd.time','epd.sapflux',
                           'epd.epdsapflux','epd.epdsapcorr',
                           'epd.bkg','epd.bkg.err',
                           'epd.xcc','epd.ycc',
                           'epd.quality']:

                if newcol not in lcdict['columns']:
                    lcdict['columns'].append(newcol)

        return times, epdfluxes, fitcoeffs, epdfit

    else:

        LOGERROR('could not fit EPD function to light curve')
        return None, None, None, None



def rfepd_kepler_lightcurve(lcdict,
                            xccol='mom_centr1',
                            yccol='mom_centr2',
                            timestoignore=None,
                            filterflags=True,
                            writetodict=True,
                            epdsmooth=23,
                            decorr='xcc,ycc',
                            nrftrees=200):
    '''
    This uses a RandomForestRegressor to fit and correct K2 light curves.

    Fits the X and Y positions, and the background and background error.

    timestoignore is a list of tuples containing start and end times to mask
    when fitting the EPD function:

    [(time1_start, time1_end), (time2_start, time2_end), ...]

    By default, this function removes points in the Kepler LC that have ANY
    quality flags set.

    if writetodict is set, adds the following columns to the lcdict:

    rfepd_time = time array
    rfepd_sapflux = uncorrected flux before EPD
    rfepd_epdsapflux = corrected flux after EPD
    rfepd_epdsapcorr = EPD flux corrections
    rfepd_bkg = background array
    rfepd_bkg_err = background errors array
    rfepd_xcc = xcoord array
    rfepd_ycc = ycoord array
    rfepd_quality = quality flag array

    and updates the 'columns' list in the lcdict as well.

    '''
    times, fluxes, background, background_err = (lcdict['time'],
                                                 lcdict['sap']['sap_flux'],
                                                 lcdict['sap']['sap_bkg'],
                                                 lcdict['sap']['sap_bkg_err'])
    xcc = lcdict[xccol]
    ycc = lcdict[yccol]
    flags = lcdict['sap_quality']

    # filter all bad LC points as noted by quality flags
    if filterflags:

        nbefore = times.size

        filterind = flags == 0

        times = times[filterind]
        fluxes = fluxes[filterind]
        background = background[filterind]
        background_err = background_err[filterind]
        xcc = xcc[filterind]
        ycc = ycc[filterind]
        flags = flags[filterind]

        nafter = times.size
        LOGINFO('applied quality flag filter, ndet before = %s, ndet after = %s'
                % (nbefore, nafter))


    # remove nans
    find = (npisfinite(xcc) & npisfinite(ycc) &
            npisfinite(times) & npisfinite(fluxes) &
            npisfinite(background) & npisfinite(background_err))

    nbefore = times.size

    times = times[find]
    fluxes = fluxes[find]
    background = background[find]
    background_err = background_err[find]
    xcc = xcc[find]
    ycc = ycc[find]
    flags = flags[find]

    nafter = times.size
    LOGINFO('removed nans, ndet before = %s, ndet after = %s'
            % (nbefore, nafter))


    # exclude all times in timestoignore
    if (timestoignore and
        isinstance(timestoignore, list) and
        len(timestoignore) > 0):

        exclind = npfull_like(times,True)

        nefore = times.size

        # apply all the masks
        for ignoretime in timestoignore:
            time0, time1 = ignoretime[0], ignoretime[1]
            thismask = (times > time0) & (times < time1)
            exclind = exclind & thismask

        # quantities after masks have been applied
        times = times[exclind]
        fluxes = fluxes[exclind]
        background = background[exclind]
        background_err = background_err[exclind]
        xcc = xcc[exclind]
        ycc = ycc[exclind]
        flags = flags[exclind]

        nafter = times.size
        LOGINFO('removed timestoignore, ndet before = %s, ndet after = %s'
                % (nbefore, nafter))


    # now that we're all done, we can do EPD

    # set up the regressor
    RFR = RandomForestRegressor(n_estimators=nrftrees)

    if decorr == 'xcc,ycc,bgv,bge':
        # collect the features and target variable
        features = npcolumn_stack((xcc,ycc,background,background_err))
    elif decorr == 'xcc,ycc':
        # collect the features and target variable
        features = npcolumn_stack((xcc,ycc))
    elif decorr == 'bgv,bge':
        # collect the features and target variable
        features = npcolumn_stack((background,background_err))
    else:
        LOGERROR("couldn't understand decorr, not decorrelating...")
        return None

    # smooth the light curve
    if epdsmooth:
        smoothedfluxes = medfilt(fluxes, epdsmooth)
    else:
        smoothedfluxes = fluxes

    # fit, then generate the predicted values, then get corrected values
    RFR.fit(features, smoothedfluxes)
    flux_corrections = RFR.predict(features)
    corrected_fluxes = npmedian(fluxes) + fluxes - flux_corrections

    # remove the random forest to save RAM
    del RFR

    # write these to the dictionary if requested
    if writetodict:

        lcdict['rfepd'] = {}
        lcdict['rfepd']['time'] = times
        lcdict['rfepd']['sapflux'] = fluxes
        lcdict['rfepd']['epdsapflux'] = corrected_fluxes
        lcdict['rfepd']['epdsapcorr'] = flux_corrections
        lcdict['rfepd']['bkg'] = background
        lcdict['rfepd']['bkg_err'] = background_err
        lcdict['rfepd']['xcc'] = xcc
        lcdict['rfepd']['ycc'] = ycc
        lcdict['rfepd']['quality'] = flags

        for newcol in ['rfepd.time','rfepd.sapflux',
                       'rfepd.epdsapflux','rfepd.epdsapcorr',
                       'rfepd.bkg','rfepd.bkg.err',
                       'rfepd.xcc','rfepd.ycc',
                       'rfepd.quality']:

            if newcol not in lcdict['columns']:
                lcdict['columns'].append(newcol)


    return times, corrected_fluxes, flux_corrections
