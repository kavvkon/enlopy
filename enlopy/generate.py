# -*- coding: utf-8 -*-
"""
Methods that generate or adjusted energy related timeseries based on given assumptions/input
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.linalg
import scipy.stats

from .utils import make_timeseries, clean_convert
from .analysis import countweekend_days_per_month

__all__ = ['disag_upsample', 'gen_daily_stoch_el', 'gen_load_from_daily_monthly', 'gen_load_sinus', 'gen_load_from_LDC',
           'gen_load_from_PSD', 'gen_gauss_markov', 'remove_outliers', 'gen_demand_response', 'add_noise',
           'gen_corr_arrays', 'gen_analytical_LDC']

_EPS = np.finfo(np.float64).eps

def disag_upsample(Load, disag_profile, to_offset='h'):
    """ Upsample given timeseries, disaggregating based on given load profiles.
    e.g. From daily to hourly. The load of each day is distributed according to the disaggregation profile. The sum of each day remains the same.

    Arguments:
        Load (pd.Series): Load profile to disaggregate
        disag_profile (pd.Series, np.ndarray): disaggregation profile to be used on each timestep of the load. Has to be compatible with selected offset.
        to_offset (str): Resolution of upsampling. has to be a valid pandas offset alias. (check `here <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`__ for all available offsets)
    Returns:
        pd.Series: the upsampled timeseries
   """
    #First reindexing to the new resolution.
    orig_freq = Load.index.freqstr
    start = Load.index[0]
    end = Load.index[-1] + 1 #An extra period is needed at the end to match the sum FIXME
    df1 = Load.reindex(pd.date_range(start, end, freq=to_offset, closed='left'))

    def mult_profile(x, profile):
        #Normalizing to keep the sum the same..
        profile = profile / np.sum(profile)
        return x.mean() * profile  #using mean() assuming that there is one value and the rest is nan

    #then transform per sampled period correspnding to the len(disag_profile)
    return df1.resample(orig_freq).transform(mult_profile, disag_profile).dropna()


def gen_daily_stoch_el(total_energy=1.0):
    """Generate stochastic dummy daily load based on hardcoded values. These values are the result of
    statistical analysis of electric loads profiles of more than 100 households. Mean and standard deviations per timestep were extracted
    from the normalized series. These are fed to :meth:`gen_gauss_markov` method.
    Arguments:
        total_energy: Sum of produced timeseries (daily load)
    Returns:
        nd.array: random realization of timeseries
    """

    means = np.array([0.02603978, 0.02266633, 0.02121337, 0.02060187, 0.02198724,
             0.02731497, 0.03540281, 0.0379463, 0.03646055, 0.03667756,
             0.03822946, 0.03983243, 0.04150124, 0.0435474, 0.0463219,
             0.05051979, 0.05745442, 0.06379564, 0.06646279, 0.06721004,
             0.06510399, 0.05581182, 0.04449689, 0.03340142])
    stds = np.array([0.00311355, 0.00320474, 0.00338432, 0.00345542, 0.00380437,
           0.00477251, 0.00512785, 0.00527501, 0.00417598, 0.00375874,
           0.00378784, 0.00452212, 0.00558736, 0.0067245, 0.00779101,
           0.00803175, 0.00749863, 0.00365208, 0.00406937, 0.00482636,
           0.00526445, 0.00480919, 0.00397309, 0.00387489])
    a = gen_gauss_markov(means,stds, .8)
    return a / a.sum() * total_energy


def gen_load_from_daily_monthly(ML, DWL, DNWL, weight=0.5, year=2015):
    """Generate annual timeseries using monthly demand and daily profiles.
    Working days and weekends are built from different profiles having different weighting factors.

    Arguments:
        ML: monthly load (size = 12)
        DWL: daily load (working day) (size = 24). Have to be normalized (sum=1)
        DNWL: daily load (non working day) (size = 24) Have to be normalized (sum=1)
        weight: weighting factor between working and non working day (0 - 1)
    Returns:
        pd.Series: Generated timeseries
    """
    #TODO: refactor. Can i use disag_upsample() ?
    if not(np.isclose(DWL.sum(), 1) and np.isclose(DNWL.sum(), 1)):
        raise ValueError('Daily profiles should be normalized')
        #TODO: Normalize here?
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
    """ Generate sinusoidal load with daily, weekly and yearly seasonality. Each term is estimated based
    on the following expression: :math:`f(x;A1,A2,w) =  A1 \\cos(2 \\pi/w \\cdot x) + A2 \\sin(2 \\pi/w \\cdot x)`

    Arguments:
        daily_1 (float): cosine coefficient for daily component (period 24)
        daily_2 (float): sinus coefficient for daily component (period 24)
        monthly_1 (float): cosine coefficient for monthly component (period 168)
        monthly_2 (float): sinus coefficient for monthly component (period 168)
        annually_1 (float): cosine coefficient for annual component (period 8760)
        annually_2 (float): sinus coefficient for annual component (period 8760)
    Returns:
        pd.Series: Generated timeseries
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


def gen_corr_arrays(Na, length, M, to_uniform=True):
    """ Generating correlated normal variates.
    Assume one wants to create a vector of random variates Z which is
    distributed according to Z~N(μ,Σ) where μ is the vector of means,
    and Σ is the variance-covariance matrix.
    http://comisef.wikidot.com/tutorial:correlateduniformvariates

    Arguments:
        Na (int): number of vectors e.g (3)
        length (int): generated vector size (e.g 8760)
        M (np.ndarray): correlation matrix. Should be of size Na x Na
        to_uniform (bool): True if the correlation matrix needs to be adjusted for uniforms
    Returns:
         np.ndarray: Realization of randomly generated correlated variables. Size : (Na, length) e.g. (3, 8760)
    """
    if Na != np.size(M, 0):  # rows of pars have to be the same size as rows and cols of M
        print('Parameters and corr matrix dimensions do not agree.')
        return False

    newM = M.copy()  # changing an array element without copying it changes it globally!
    u = np.random.randn(length, Na)
    if min(np.linalg.eig(M)[0]) < 0:  # is M positive definite?
        print ('Error: Eigenvector is not positive. Trying to make positive, but it may differ from the initial.')
        # Make M positive definite:
        la, v = np.linalg.eig(newM)
        la[la < 0] = np.spacing(1)  # Make all negative eigenvalues zero
        ladiag = np.diag(la)  # Diagonal of eigenvalues
        newM = np.dot(np.dot(v, ladiag), v.T)  # Estimate new M = v * L * v'

    # Transformation is needed to change normal to uniform (Spearman - Pearson)
    if to_uniform:
        for i in np.arange(0, Na):  # 1:Na
            for j in np.arange(max(Na-1, i), Na):  # max(Na-1,i):Na
                if i != j:
                    newM[i, j] = 2 * np.sin(np.pi * newM[i, j] / 6)
                    newM[j, i] = 2 * np.sin(np.pi * newM[j, i] / 6)

    if min(np.linalg.eig(newM)[0]) <= 0:
        print ('Error: Eigenvector is still not positive. Aborting')
        return False

    cF = scipy.linalg.cholesky(newM)
    Y = np.dot(u, cF).T
    Y = scipy.stats.norm.cdf(Y)  # remove if you produce random.rand?
    return Y


def gen_load_from_LDC(LDC, Y=None, N=8760):
    """ Generate loads based on a Inverse CDF, such as a Load Duration Curve (LDC)
    Inverse transform sampling: Compute the value x such that F(x) = u.
    Take x to be the random number drawn from the distribution described by F.

    .. note::
        Due to the sampling process this function produces load profiles with unrealistic temporal sequence, which means that they cannot be treated as
        timeseries. It is recommended that :meth:`gen_load_from_PSD` is used afterwards.

    Arguments:
        LDC (np.ndarray): Load duration curve (2 x N) vector of the x, y coordinates of an LDC function (results of (get_LDC).
            x coordinates have to be normalized (max: 1 => 8760hrs )
        Y (nd.array): a vector of random numbers. To be used for correlated loads.
            If None is supplied a random vector (8760) will be created.
        N (int): Length of produced timeseries (if Y is not provided)

    Returns:
         np.ndarray: vector with the same size as Y that respects the statistical distribution of the LDC
    """
    if Y is None:  # if there is no Y, generate a random vector
        Y = np.random.rand(N)
    func_inv = scipy.interpolate.interp1d(LDC[0], LDC[1], bounds_error=False, fill_value=0)
    simulated_loads = func_inv(Y)
    # ------- Faster way:   # np.interp is faster but have to sort LDC
    # if np.all(np.diff(LDC[0]) > 0) == False: #if sorted
    #idx = np.argsort(LDC[0])
    #LDC_sorted = LDC[:, idx].copy()
    #simulated_loads = np.interp(Y, LDC_sorted[0], LDC_sorted[1])
    # no need to insert timed index since there is no spectral information
    return simulated_loads


def gen_load_from_PSD(Sxx, x, dt=1):
    """
    Algorithm for generating samples of a random process conforming to spectral
    density Sxx(w) and probability density function p(x).

    .. note::
        This is done by an iterative process which 'shuffles' the timeseries till convergence of both
        power spectrum and marginal distribution is reached.
        Also known as "Iterated Amplitude Adjusted Fourier Transform (IAAFT). Adopted from `J.M. Nichols, C.C. Olson, J.V. Michalowicz, F. Bucholtz, (2010), "A simple algorithm for generating spectrally colored, non-Gaussian signals" Probabilistic Engineering Mechanics, Vol 25, 315-322`
        and `Schreiber, T. and Schmitz, A. (1996) "Improved Surrogate Data for Nonlinearity Tests", Physical Review Letters, Vol 77, 635-638.`

    Arguments:
        Sxx: Spectral density (two sided)
        x: Sequence of observations created by the desirable PDF. You can use :meth:`gen_load_from_LDC` for that.
        dt: Desired temporal sampling interval. [Dt = 2pi / (N * Dw)]
    Returns:
        pd.Series: The spectrally corrected timeseries

    """
    N = len(x)
    Sxx[int(N/2)+1] = 0  # zero out the DC component (remove mean)
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
            print('Converged after {} iterations'.format(k))
            k = 0  # if we converged, stop
        indxp = indx  # re-set ordering for next iter
    out = out + mx  # Put back in the mean
    return out


def gen_gauss_markov(mu, st, r):
    """ Generate timeseries based on means, stadnard deviation and autocorrelation per timestep

    Based on
    A.M. Breipohl, F.N. Lee, D. Zhai, R. Adapa, A Gauss-Markov load model for the application in risk evaluation
    and production simulation, Transactions on Power Systems, 7 (4) (1992), pp. 1493-1499

    Arguments:
        mu: array of means. Can be either 1d or 2d
        st: array of standard deviations. Can be either 1d or 2d. Can be either scalar (same for entire timeseries or array with the same length as the timeseries
        r:  Autoregressive coefficient AR(1). Has to be between  [-1,1]. Can be either scalar (same for entire timeseries or array with the same length as the timeseries
    Returns:
        pd.Series, pd.DataFrame: a realization of the timeseries
    """
    mu = np.atleast_2d(mu)
    loadlength = mu.shape
    rndN = np.random.randn(*loadlength)
    if np.atleast_2d(st).shape[1] == 1:
        noisevector = st * np.ones(loadlength)
    elif len(st) == loadlength[1]:
        noisevector =  np.atleast_2d(st)
    else:
        raise ValueError('Length of standard deviations must be the same as the length of means. You can also use one value for the entire series')

    if np.atleast_2d(r).shape[1] == 1:
        rvector = r * np.ones(loadlength)
    elif len(r) == loadlength[1]:
        rvector = np.atleast_2d(r)
    else:
        raise ValueError('Length of autocorrelations must be the same as the length of means. You can also use one value for the entire series')

    y = np.zeros(loadlength)
    noisevector[noisevector == 0] = _EPS
    y[:,0] = mu[:,0] + noisevector[:, 0] * rndN[:, 0]

 #   for t in mu.T:
    for i in range(mu.shape[1]):
        y[:,i] = (mu[:,i] +
                r * noisevector[:, i] /
                noisevector[:, i - 1] * (y[:, i - 1] - mu[:, i - 1]) +
                noisevector[:, i] * np.sqrt(1 - rvector[:, i] ** 2) * rndN[:, i])
    return y.squeeze()


def add_noise(Load, mode, st, r=0.9, Lmin=0):
    """ Add noise with given characteristics.

    Arguments:
        Load (pd.Series,pd.DataFrame): 1d or 2d timeseries
        mode (int):1 Normal Distribution, 2: Uniform Distribution, 3: Gauss Markov (autoregressive gaussian)
        st (float): Noise parameter. Scaling of random values
        r (float): Applies only for mode 3. Autoregressive coefficient AR(1). Has to be between  [-1,1]
        Lmin (float): minimum load values. This is used to trunc values below zero if they are generated with a lot of noise
    Returns:
        pd.Series: Load with noise
    """

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
    elif mode == 3:  # Gauss-Markov, same as
        out = gen_gauss_markov(L, st, r)
    else:
        raise ValueError('Not available mode')
    out[out < Lmin] = Lmin  # remove negative elements

    return clean_convert(np.squeeze(out), force_timed_index=True, freq='h') # assume hourly timeseries if no timeindex is passed


def gen_analytical_LDC(U, duration=8760, bins=1000):
    """Generates the Load Duration Curve based on empirical parameters. The following equation is used.
    :math:`f(x;P,CF,BF) = \\frac{P-x}{P-BF \\cdot P}^{\\frac{CF-1}{BF-CF}}`

    Arguments:
        U (tuple): parameter vector [Peak load, capacity factor%, base load%, hours] or dict
    Returns:
        np.ndarray: a 2D array [x, y] ready for plotting (e.g. plt(\*gen_analytical_LDC(U)))
    """
    if isinstance(U, dict):
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
    otherwise it is smaller due to the shaved peaks. The peak load is reduced by a predefined percentage.

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


def remove_outliers(Load, **kwargs):
    """ Removes outliers identified by :meth:`detect_outliers` and replaces them by interpolated value.

    Arguments:
        Load: input timeseries
        **kwargs: Exposes keyword arguments of :meth:`detect_outliers`
    Returns:
        Timeseries cleaned from outliers
    """
    from .analysis import detect_outliers
    outlier_idx = detect_outliers(Load, **kwargs)
    filtered_series = Load.copy()
    filtered_series[outlier_idx] = np.nan
    return filtered_series.interpolate(method='time')
