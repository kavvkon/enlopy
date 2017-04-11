# -*- coding: utf-8 -*-
"""
Methods that generate or adjusted energy related timeseries based on given assumptions/input
"""
from __future__ import absolute_import, division, print_function
from future.builtins.iterators import filter, map, zip

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.linalg
import scipy.stats

from .utils import make_timeseries, clean_convert
from .analysis import countweekend_days_per_month

__all__ = ['disag_upsample', 'gen_load_from_daily_monthly', 'gen_load_sinus', 'gen_load_from_LDC',
           'gen_load_from_PSD', 'detect_outliers', 'gen_demand_response', 'add_noise',
           'gen_corr_arrays', 'gen_analytical_LDC']

_EPS = np.finfo(np.float64).eps

def disag_upsample(Load1, disag_profile, to_offset='h'):
    """ Upsample given timeseries, disaggregating based on given load profiles.
    e.g. From daily to hourly. Each day is split. The sum of each day remains the same.
   """
    # def custom_resampler(array_like):
    #        return np.sum(array_like)+5
    # series.resample('15m').apply(custom_resampler)
    # Different per day of the week ?


    #First have to reindex to the full
    orig_freq = Load1.index.freqstr
    start = Load1.index[0]
    end = Load1.index[-1] + 1 #An extra period is needed at the end to match the sum FIXME
    df1 = Load1.reindex(pd.date_range(start, end, freq=to_offset))

    def mult_profile(x, profile):
        #Normalizing to keep the sum the same..
        profile = profile / np.sum(profile)
        return x.mean() * profile  #using mean() assuming that there is one value and the rest is nan

    #then transform per sampled period correspnding to the len(disag_profile)
    return df1.resample(orig_freq).transform(mult_profile, disag_profile).dropna()


def gen_load_from_daily_monthly(ML, DWL, DNWL, weight=0.5, year=2015):
    """Generate annual timeseries using monthly demand and daily profiles.
    Working days are followed by non-working days which produces loads with
    unrealistic temporal sequence, which means that they cannot be treated as
    timeseries.

    Arguments:
        ML: monthly load (size = 12)
        DWL: daily load (working day) (size = 24). Have to be normalized (sum=1)
        DNWL: daily load (non working day) (size = 24) Have to be normalized (sum=1)
        weight: weighting factor between working and non working day (0 - 1)
    Returns:
        pd.Series: Generated timeseries
    """
    #TODO: refactor. Can i use disag_upsample() ?
    if not(np.isclose(DWL.sum(),1) and np.isclose(DNWL.sum(),1)):
        print ('Daily profiles should be normalized')
        #TODO: Normalize here?
        return
    out = make_timeseries(year=year, length=8760, freq='H')  # Create empty pandas with datetime index
    import calendar
    febdays = 29 if calendar.isleap(year) else 28
    Days = np.array([31, febdays, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # Assumptions for non working days per month. Only weekends
    # TODO: Custom Calendars with holidays BDays
    DaysNW = countweekend_days_per_month(out.resample('d').mean())
    DaysW = Days - DaysNW
    for month in range(12):
        # Estimate total load for working and non working day
        TempW = (ML[month] * weight * DaysW[month] /
                (weight * DaysW[month] + (1 - weight) * DaysNW[month]) / DaysW[month] )
        TempNW = (ML[month] * (1 - weight) * DaysNW[month] /
                 (weight * DaysW[month] + (1 - weight) * DaysNW[month]) / DaysNW[month])
        for hour in range(24):
            out.loc[(out.index.month == month + 1) &  #months dont start from 0
                   (out.index.weekday < 5) &
                   (out.index.hour == hour)] = (TempW * DWL[hour])
            out.loc[(out.index.month == month + 1) &
                   (out.index.weekday >= 5) &
                   (out.index.hour == hour)] = (TempNW * DNWL[hour])
    return out


def gen_load_sinus(daily_1, daily_2, monthly_1, monthly_2, annually_1, annually_2):
    """Generate sinusoidal load with daily, weekly and yearly seasonality. Each term is estimated based on 
     the following expression:
            .. math:: f(x;A1,A2,w) =  A1 * np.cos(2 * \\pi/w * x) + A2 * np.sin(2 * \\pi/w * x)
    Arguments:
        daily_1 (float): cosine coefficient for daily component
        daily_2 (float): sinus coefficient for daily component
        monthly_1 (float): cosine coefficient for monthly component
        monthly_2 (float): sinus coefficient for monthly component
        annually_1 (float): cosine coefficient for annual component
        annually_2 (float): sinus coefficient for annual component
    Returns:
        pd.Series: Generated timeseries

    Arguements:
    :type daily_1: object
    """
    def sinusFunc(x, w, A1, A2): # fourrier coefficient
        return A1 * np.cos(2 * np.pi/w * x) + A2 * np.sin(2 * np.pi/w * x)

    x = np.arange(0, 8760)
    # daily, weekly, annual periodicity     #TODO: custom periodicity
    coeffs ={24: (daily_1, daily_2),
             168: (monthly_1, monthly_2),
             8760: (annually_1, annually_2)
             }

    out = 0
    for period, values in coeffs.items():
        out += sinusFunc(x, period, *values)

    return make_timeseries(out)


def gen_corr_arrays(Na, hours, M):
    """ Generating correlated normal variates
    Assume one wants to create a vector of random variates Z which is
    distributed according to Z~N(μ,Σ) where μ is the vector of means,
    and Σ is the variance-covariance matrix.
    http://comisef.wikidot.com/tutorial:correlateduniformvariates
    Arguments:
        Na: number of vectors e.g (3)
        hours: vector size (e.g 8760 hours)
        M: correlation matrix
    Returns:
         list: Realization of randomly generated correlated variables. Size : (Na, hours) e.g. (3, 8760)
    """
    if Na != np.size(M, 0):  # rows of pars have to be the same size as rows and cols of M
        Y = -1
        print('Parameters and corr matrix dimensions do not agree.')
        pass

    newM = M.copy()  # changing an array element without copying it changes it globally!
    u = np.random.randn(hours, Na)
    if min(np.linalg.eig(M)[0]) < 0:  # is M positive definite?
        print ('Error: Eigenvector is not positive')
        # Make M positive definite:
        la, v = np.linalg.eig(newM)
        la[la < 0] = np.spacing(1)  # Make all negative eigenvalues zero
        ladiag = np.diag(la)  # Diagonal of eigenvalues
        newM = np.dot(np.dot(v, ladiag), v.T)  # Estimate new M = v * L * v'

    # Transformation is needed to change normal to uniform (Spearman - Pearson)
    for i in np.arange(0, Na):  # 1:Na
        for j in np.arange(max(Na-1, i), Na):  # max(Na-1,i):Na
            if i != j:
                newM[i, j] = 2 * np.sin(np.pi * newM[i, j] / 6)
                newM[j, i] = 2 * np.sin(np.pi * newM[j, i] / 6)

    if min(np.linalg.eig(newM)[0]) <= 0:  # check again brute force #while
        print ('Error: Eigenvector is not positive')
        pass
        # M[1:(Na+1):(Na*Na)] = 1
        # M = hyper_decomp(M)

    cF = scipy.linalg.cholesky(newM)
    Y = np.dot(u, cF).T
    Y = scipy.stats.norm.cdf(Y)  # remove if you produce random.rand?
    return Y


def gen_load_from_LDC(LDC, Y=None, N=8760):
    """ Generate loads based on a Inverse CDF, such as a Load Duration Curve
    Inverse transform sampling: Compute the value x such that F(x) = u.
    Take x to be the random number drawn from the distribution described by F.
        LDC: 2 x N vector of the x, y coordinates of an LDC function.
            x coordinates have to be normalized (max = 1 ==> 8760hrs )
        Y (optional): a vector of random numbers. To be used for correlated loads.
            If None is supplied a random vector (8760) will be created.
        N (optional): Length of produced timeseries (if Y is not provided)

    Returns a vector with the same size as Y that follows the LDC distribution
    """
    # func_inv = scipy.interpolate.interp1d(LDC[0], a[1])
    # simulated_loads = func_inv(Y)
    # ------- Faster way:   # np.interp is faster but have to sort LDC
    if Y is None:  # if there is no Y, generate a random vector
        Y = np.random.rand(N)
    # if np.all(np.diff(LDC[0]) > 0) == False: #if sorted
    idx = np.argsort(LDC[0])
    LDC_sorted = LDC[:, idx].copy()
    simulated_loads = np.interp(Y, LDC_sorted[0], LDC_sorted[1])
    # no need to insert timed index since there is no spectral information
    return simulated_loads


def gen_load_from_PSD(Sxx, x, dt=1):
    """
    Algorithm for generating samples of a random process conforming to spectral
    density Sxx(w) and probability density function p(x).
    This is done by an iterative process which 'shuffles' the timeseries till
    convergence of both power spectrum and marginal distribution is reached.
    Also known as "Iterated Amplitude Adjusted Fourier Transform (IAAFT)"
    Adopted from:
    J.M. Nichols, C.C. Olson, J.V. Michalowicz, F. Bucholtz, (2010)
    "A simple algorithm for generating spectrally colored, non-Gaussian signals
    Probabilistic Engineering Mechanics", Vol 25, 315-322
    Schreiber, T. and Schmitz, A. (1996) "Improved Surrogate Data for
    Nonlinearity Tests", Physical Review Letters, Vol 77, 635-638.

    Arguments:
        Sxx: Spectral density (two sided)
        x: Sequence of observations created by the desirable PDF
        dt: Desired temporal sampling interval. [Dt = 2pi / (N * Dw)]

    """
    N = len(x)
    Sxx[N/2+1] = 0  # zero out the DC component (remove mean)
    Xf = np.sqrt(2 * np.pi * N * Sxx / dt)  # Convert PSD to Fourier amplitudes
    Xf = np.fft.ifftshift(Xf)  # Put in Matlab FT format
    # The following lines were commented out because they outscale the data
    # modifying thus its PDF. However, according to Nichols et al. they
    # guarantee that the new data match the signal variance
    #vs = (2 * np.pi / N / dt) * sum(Sxx) * (N / (N-1))  # Get signal variance (as determined by PSD)
    #out = x * np.sqrt(vs / np.var(x))
    out = x
    mx = np.mean(out)
    out = out - mx  # subtract the mean
    indx = np.argsort(out)
    xo = out[indx].copy()  # store sorted signal xo with correct p(x)

    k = 1
    indxp = np.zeros(N)  # initialize counter
    while(k):
        Rk = np.fft.fft(x)  # Compute FT
        Rp = np.angle(Rk)  # ==> np.arctan2(np.imag(Rk), np.real(Rk))  # Get phases
        out = np.real(np.fft.ifft(np.exp(1j * Rp) * np.abs(Xf)))  # Give signal correct PSD
        indx = np.argsort(out)  # Get rank of signal with correct PSD
        out[indx] = xo  # rank reorder (simulate nonlinear transform)
        k = k + 1  # increment counter
        if np.array_equal(indx, indxp):
            print('Converged after %i iterations') % k
            k = 0  # if we converged, stop
        indxp = indx  # re-set ordering for next iter
    out = out + mx  # Put back in the mean
    return make_timeseries(out)


def add_noise(Load, mode, st, r=0.9, Lmin=0):
    """ Add noise based on specific distribution
    LOAD_ADDNOISE Add noise to a 1x3 array [El Th Co].
       Mode 1 = Normal Dist
       Mode 2 = Uniform Dist
       Mode 3 = Gauss Markov
       st = Noise parameter. Scaling of random values
    """
    def GaussMarkov(mu, st, r):
        """A.M. Breipohl, F.N. Lee, D. Zhai, R. Adapa
        A Gauss-Markov load model for the application in risk evaluation
        and production simulation
        Transactions on Power Systems, 7 (4) (1992), pp. 1493-1499
        Arguments:
            mu: array of means. Can be either 1d or 2d
            st: array of standard deviations. Can be either 1d or 2d
            r: autocorrelation (one lag) coefficient. Has to be between [-1,1]
        Returns:
            a realization of the timeseries

        """
        mu = np.atleast_2d(mu)
        loadlength = mu.shape
        rndN = np.random.randn(*loadlength)
        noisevector = st * np.ones(loadlength)
        y = np.zeros(loadlength)
        noisevector[noisevector == 0] = _EPS
        y[0] = mu[0] + noisevector[0] * rndN[0]

        for t in mu.T:
            for i in range(len(t)):
                y[i] = (mu[i] +
                        r * noisevector[i] /
                        noisevector[i - 1] * (y[i - 1] - mu[i - 1]) +
                        noisevector[i] * np.sqrt(1 - r ** 2) * rndN[i])
            y[y < Lmin] = Lmin  # remove negative elements
            return y

    L = np.atleast_2d(Load)

    if st == 0:
        print('No noise to add')
        return Load

    loadlength = L.shape  # 8760
    if mode == 1:  # Normal
        noisevector = st * np.random.randn(*loadlength)  # gauss.. should it have a zero sum?
        out = L * (1 + noisevector)
    elif mode == 2:  # Uniform
        noisevector = st * np.random.rand(*loadlength)
        out = L * ((1 - st) + st * noisevector)
    elif mode == 3:  # Gauss-Markov
        out = GaussMarkov(L, st, r)
    else:
        raise ValueError('Not available mode')
    return clean_convert(np.squeeze(out))


def gen_analytical_LDC(U, duration=8760, bins=1000):
    """Generates the Load Duration Curve based on empirical parameters.
        .. math:: f(x;P,CF,BF) = \\frac{P-x}{P-BF \\cdot P}^{\\frac{CF-1}{BF-CF}}
    
    Arguments:
        U(dict): parameter vector [Peak load, capacity factor%, base load%, hours]
    Returns:
        (np.array) a 2D array [x, y] ready for plotting (e.g. plt(*gen_analytical_LDC(U)))
    """
    if isinstance(U,dict):
        P = U['peak']  # peak load
        CF = U['LF']  # load factor
        BF = U['base']  # base load
        h = U['hourson']  # hours
    else: #unpack
        P, CF, BF, h = U

    x = np.linspace(0, P, bins)

    ff = h * ((P - x)/(P - BF * P))**((CF - 1)/(BF - CF))
    ff[x < (BF*P)] = h
    ff[x > P] = 0
    return ff/duration, x


def gen_demand_response(Load, percent_peak_hrs_month=0.03, percent_shifted=0.05, shave=False):
    """Simulate a demand response mechanism that makes the load profile less peaky.
    The load profile is analyzed per selected period (currently month month) and the peak hours have their load shifted
     to low load hours or shaved. When not shaved the total load is the same as that one from the initial timeseries,
     otherwise it is smaller due to the shaved peaks. The ppeak load is reduced by a predefined percentage.
    Arguments:
        Load (pd.Series): Load 
        percent_peak_hrs_month (float): fraction of hours to be shifted
        percent_shifted (float): fraction of energy to be shifted if the day is tagged for shifting/shaving
        shave (bool): If False peak load will be transfered to low load hours, otherwise it will be shaved.
    Return:
        pd.Series: New load profile with reduced peaks. The peak can be shifted to low load hours or shaved
    """
    if not Load.index.is_all_dates:
        print ('Need date Time indexed series. Trying to force one.')
        Load = clean_convert(Load, force_timed_index=True)
    demand = Load

    def hours_per_month(demand):
        """Assign to each row hours per month"""
        dic_hours_per_month = demand.groupby(demand.index.month).count().to_dict()
        return demand.resample('m').transform(lambda x: list(map(dic_hours_per_month.get, x.index.month)))

    total_demand = demand.sum()

    # Monthly demand rank
    # TODO: parametrize: we can check peaks on a weekly or daily basis
    demand_m_rank = demand.resample('m').transform(lambda x: x.rank(method='min', ascending=False))

    # find which hours are going to be shifted
    bool_shift_from = demand_m_rank <= np.round(hours_per_month(demand) * percent_peak_hrs_month)

    DR_shift_from = percent_shifted * demand  # demand_fpeak * total_demand * percent_shifted
    DR_shift_from[~bool_shift_from] = 0

    # find hours that are going to have
    if shave:
        #If (peak) shaving we do not shift the loads anywhere
        DR_shift_to = 0
    else:
        # Estimate amount of load to be shifted per month
        sum_shifted = DR_shift_from.groupby(DR_shift_from.index.month).sum()
        count_shifted = DR_shift_from[DR_shift_from > 0].groupby(DR_shift_from[DR_shift_from > 0].index.month).count()
        shift_to_month = sum_shifted / count_shifted
        #Find which hours are going to be filled with the shifted load
        bool_shift_to = demand_m_rank > np.round(hours_per_month(demand) * (1 - percent_peak_hrs_month))

        df_month = pd.Series(demand.index.month, index=demand.index)
        DR_shift_to = df_month.map(shift_to_month)
        DR_shift_to[~bool_shift_to] = 0

    # Adjusted hourly demand
    dem_adj = demand.copy()
    dem_adj[bool_shift_from] = dem_adj[bool_shift_from] * (1 - percent_shifted)
    dem_adj[~bool_shift_from] = dem_adj[~bool_shift_from] + DR_shift_to
    return dem_adj


def detect_outliers(Load, threshold=None, window=5, plot_diagnostics=False):
    """ . Inspired by https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    Arguments:
        Load: input timeseries
        threshold: if None then 3 sigma is selected as threshold
        window: how many values to check
        plot_diagnostics: Plot diagnostics to check whether the outliers were removed accurately
    Return:
        timeseries cleaned from outliers
   """
    # TODO : Clean zero values (interpolate)

    a = Load.rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill')
    difference = np.abs(Load - a)

    if threshold is None:
        threshold = 3 * np.std(Load)
    outlier_idx = difference > threshold

    filtered_series = Load.copy()
    filtered_series[outlier_idx] = np.nan
    filtered_series = filtered_series.interpolate(method='time')
    if plot_diagnostics:
        kw = dict(marker='o', linestyle='none', color='r', alpha=0.5)
        Load[outlier_idx].plot(**kw)
    return filtered_series
