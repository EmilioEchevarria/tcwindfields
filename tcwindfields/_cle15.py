# /usr/bin/env python
import numpy as np
import sys
import copy
from shapely.geometry import LineString
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, PchipInterpolator


def E04_outerwind_r0input_nondim_MM0(r0, fcor, Cdvary, C_d, w_cool, Nr=None):
    '''
    Inputs:
        r0 = outer radius where V=0 [m]
        fcor = Coriolis parameter [s-1]
        Cdvary = 0: C_d constant; 1 : C_d=C_d(V) following Donelan et al (2004)
        C_d = drag coefficient; ignored if Cdvary = 1
        w_cool = radiative subsidence rate (positive = downwards) [ms-1]
        Nr = number of radial nodes inwards of r0. default will calculate entire profile
    '''

    fcor = np.abs(fcor)
    M0 = .5*fcor*(r0**2)

    drfracr0 = .001
    if ((r0 > 2500*1000) | (r0 < 200*1000)):
        drfracr0 = drfracr0/10

    if Nr is None:
        Nr = int(1/drfracr0)

    if Nr > int(1/drfracr0):
        Nr = int(1/drfracr0)

    rfracr0_max = 1.0
    rfracr0_min = rfracr0_max - (Nr-1)*drfracr0
    rrfracr0 = np.arange(rfracr0_min,rfracr0_max+drfracr0,drfracr0)
    MMfracM0 = np.full_like(rrfracr0, np.nan)
    MMfracM0[-1] = 1.0

    rfracr0_temp = rrfracr0[-2]
    MfracM0_temp = MMfracM0[-1]
    MMfracM0[-2] = MfracM0_temp

    C_d_lowV = 6.2e-4
    V_thresh1 = 6
    V_thresh2 = 35.4
    C_d_highV = 2.35e-3
    linear_slope = (C_d_highV-C_d_lowV)/(V_thresh2-V_thresh1)

    for ii in range(int(Nr)-2):
        if Cdvary==1:
            V_temp = (M0/r0)*((MfracM0_temp/rfracr0_temp)-rfracr0_temp)
            if V_temp<=V_thresh1:
                C_d = C_d_lowV
            elif V_temp>V_thresh2:
                C_d = C_d_highV
            else:
                C_d = C_d_lowV + linear_slope*(V_temp-V_thresh1)

        gam = C_d*fcor*r0/w_cool
        dMfracM0_drfracr0_temp = gam*((MfracM0_temp-rfracr0_temp**2)**2)/(1-rfracr0_temp**2)
        MfracM0_temp = MfracM0_temp - dMfracM0_drfracr0_temp*drfracr0
        rfracr0_temp = rfracr0_temp - drfracr0
        MMfracM0[-ii-3] = MfracM0_temp

    return rrfracr0,MMfracM0


def ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11):
    fcor = np.abs(fcor)
    if rmax_or_r0 == 'rmax':
        rmax = r_in
    else:
        print('rmax_or_r0 must be set to"rmax"')

    with np.errstate(divide='ignore', invalid='ignore'):
        V_ER11 = (1./rr_ER11)*(Vmax*rmax + .5*fcor*rmax**2)*((2*((rr_ER11/rmax)**2))/(2-CkCd+CkCd*((rr_ER11/rmax)**2)))**(1/(2-CkCd)) - .5*fcor*rr_ER11
    V_ER11[np.isnan(V_ER11)] = 0
    V_ER11[rr_ER11==0] = 0

    if rmax_or_r0 == 'rmax':
        i_rmax = np.argmax(V_ER11)
        f = PchipInterpolator(V_ER11[i_rmax+1:][::-1], rr_ER11[i_rmax+1:][::-1])
        r_out = float(f(0))
    else:
        print('rmax_or_r0 must be set to"rmax"')
    return V_ER11,r_out


def ER11_radprof(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11):
    dr = rr_ER11[1]-rr_ER11[0]
    V_ER11,r_out = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11)

    if rmax_or_r0 == 'rmax':
        drin_temp = r_in-rr_ER11[np.argwhere(V_ER11==np.max(V_ER11))[0,0]]
    elif rmax_or_r0 =='r0':
        i_peak = np.argmax(V_ER11)
        f = PchipInterpolator(V_ER11[i_peak:][::-1], rr_ER11[i_peak:][::-1])
        drin_temp = r_in - float(f(0))

    dVmax_temp = Vmax - np.max(V_ER11)

    r_in_save = copy.copy(r_in)
    Vmax_save = copy.copy(Vmax)

    n_iter = 0
    while((np.abs(drin_temp)>dr/2) or (np.abs(dVmax_temp/Vmax_save)>=10**-2)):
        n_iter = n_iter + 1
        if n_iter>20:
            V_ER11 = float('NaN')*np.zeros(rr_ER11.size)
            r_out = float('NaN')
            break

        r_in = r_in + drin_temp

        while(np.abs(dVmax_temp/Vmax)>=10**-2):
            Vmax = Vmax + dVmax_temp
            [V_ER11,r_out] = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11)
            Vmax_prof = np.max(V_ER11)
            dVmax_temp = Vmax_save-Vmax_prof

        [V_ER11,r_out] = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11)
        Vmax_prof = np.max(V_ER11)
        dVmax_temp = Vmax_save-Vmax_prof
        if rmax_or_r0=='rmax':
            drin_temp = r_in_save-rr_ER11[np.argwhere(V_ER11==Vmax_prof)[0,0]]
        elif rmax_or_r0=='r0':
            i_peak = np.argmax(V_ER11)
            f = PchipInterpolator(V_ER11[i_peak:][::-1], rr_ER11[i_peak:][::-1])
            drin_temp = r_in_save - float(f(0))

    return V_ER11,r_out


def curveintersect(x1, y1, x2, y2):
    x_intersections = []
    y_intersections = []

    ind_x1 = np.sign(np.diff(x1))
    ind_x2 = np.sign(np.diff(x2))

    i1 = 0
    while i1 < len(x1) - 1:
        ind_max1 = i1 + np.where(ind_x1[i1:] != ind_x1[i1])[0][0] if np.any(ind_x1[i1:] != ind_x1[i1]) else len(x1) - 1
        i1_indices = np.arange(i1, ind_max1 + 1)

        i2 = 0
        while i2 < len(x2) - 1:
            ind_max2 = i2 + np.where(ind_x2[i2:] != ind_x2[i2])[0][0] if np.any(ind_x2[i2:] != ind_x2[i2]) else len(x2) - 1
            i2_indices = np.arange(i2, ind_max2 + 1)

            if ind_x1[i1] == 0 and ind_x2[i2] != 0:
                x_loc = x1[i1]
                y_loc = interp1d(x2[i2_indices], y2[i2_indices], bounds_error=False, fill_value=np.nan)(x_loc)
                if not (np.nanmin(y1[i1_indices]) <= y_loc <= np.nanmax(y1[i1_indices])):
                    x_loc, y_loc = [], []
            elif ind_x2[i2] == 0 and ind_x1[i1] != 0:
                x_loc = x2[i2]
                y_loc = interp1d(x1[i1_indices], y1[i1_indices], bounds_error=False, fill_value=np.nan)(x_loc)
                if not (np.nanmin(y2[i2_indices]) <= y_loc <= np.nanmax(y2[i2_indices])):
                    x_loc, y_loc = [], []
            elif ind_x1[i1] != 0 and ind_x2[i2] != 0:
                x_loc, y_loc = curveintersect_local(x1[i1_indices], y1[i1_indices], x2[i2_indices], y2[i2_indices])
            else:
                x_loc, y_loc = [], []

            if x_loc is not None and y_loc is not None:
                x_intersections.extend(np.atleast_1d(x_loc))
                y_intersections.extend(np.atleast_1d(y_loc))

            i2 = i2_indices[-1] + 1
        i1 = i1_indices[-1] + 1

    return np.array(x_intersections), np.array(y_intersections)


def curveintersect_local(x1, y1, x2, y2):
    if not np.array_equal(x1, x2):
        xx = np.unique(np.concatenate([x1, x2]))
        xx = xx[(xx >= max(min(x1), min(x2))) & (xx <= min(max(x1), max(x2)))]
        if len(xx) < 2:
            return [], []
        yy = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value=np.nan)(xx) - \
             interp1d(x2, y2, kind='linear', bounds_error=False, fill_value=np.nan)(xx)
    else:
        xx = x1
        yy = y1 - y2

    x_crossings = mminvinterp(xx, yy, 0)
    if x_crossings.size > 0:
        y_crossings = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value=np.nan)(x_crossings)
        return x_crossings, y_crossings
    return [], []


def mminvinterp(x, y, y_target):
    x = np.asarray(x)
    y = np.asarray(y)

    if y_target < y.min() or y_target > y.max():
        return np.array([])

    crossings = []
    for i in range(len(y) - 1):
        if (y[i] - y_target) * (y[i + 1] - y_target) < 0:
            alpha = (y_target - y[i]) / (y[i + 1] - y[i])
            crossings.append(x[i] + alpha * (x[i + 1] - x[i]))
    return np.array(crossings)


def ER11E04_nondim_rmaxinput(Vmax, rmax, fcor, Cdvary, C_d, w_cool,
                              CkCdvary=None, CkCd=None, eye_adj=None, alpha_eye=None):
    '''
    Inputs:
        Vmax = maximum wind speed [ms-1]
        rmax = radius of maximum winds [m]
        fcor = Coriolis parameter [s-1] 
        Cdvary = 1=C_d varies following Donelan et al 2004; 0=input value
        C_d = drag coefficient in outer region; ignored if Cdvary = 1
        w_cool = radiative-subsidence rate [sm-1]
        CkCdvary = 1=C_k/C_d varies following quadratic fit to Vmax from Chavas et al. 2015; 0=input value
        CkCd = ratio of surface exchange coefficients
        eye_adj = 0 = use ER11 profile in eye; 1 = empirical adjustment
        alpha_eye = V/Vm in eye is reduced by factor (r/rm)^alpha_eye; ignored if eye_adj=0

    Outputs:
        rr = vector of radii [m]
        VV = vector of wind speeds at rr [ms-1]
        r0 = outer radius where V=0 [m]
        rmerge = radius of merge point between inner and outer solutions [m]
        Vmerge = wind speed at rmerge [ms-1]
    '''
    if CkCdvary is not None and CkCd is None:
        raise ValueError('CkCdvary has been input, but not CkCd')
    if eye_adj is not None and alpha_eye is None:
        raise ValueError('eye_adj has been input, but not alpha_eye')
    if CkCdvary is None and CkCd is None and eye_adj is None and alpha_eye is None:
        CkCdvary = 0
        CkCd = 1
        eye_adj = 0
        alpha_eye = .15
    if eye_adj is None and alpha_eye is None:
        eye_adj = 0
        alpha_eye = .15

    fcor = np.abs(fcor)
    if CkCdvary==1:
        CkCd = 5.5041e-04*Vmax**2 - 0.0259*Vmax + 0.7627
    CkCd = np.min((1.9, CkCd))

    drfracrm = .01
    if rmax > 100.*1000:
        drfracrm = drfracrm/10.
    rfracrm_min = 0.
    rfracrm_max = 60.
    rrfracrm_ER11 = np.arange(rfracrm_min, rfracrm_max+drfracrm, drfracrm)

    rr_ER11 = rrfracrm_ER11*rmax
    rmax_or_r0 = 'rmax'

    soln_converged = 0
    count = 0
    while soln_converged==0:
        count += 1
        VV_ER11, dummy = ER11_radprof(Vmax, rmax, rmax_or_r0, fcor, CkCd, rr_ER11)
        if not np.isnan(np.max(VV_ER11)):
            soln_converged = 1
        else:
            CkCd = CkCd + .1
        if count >= 100:
            break

    if soln_converged == 1:
        Mm = rmax*Vmax + .5*fcor*rmax**2
        MMfracMm_ER11 = (rr_ER11*VV_ER11 + .5*fcor*rr_ER11**2)/Mm

        rmaxr0_min = .001
        rmaxr0_max = .75
        rmaxr0_new = (rmaxr0_max+rmaxr0_min)/2
        rmaxr0 = rmaxr0_new
        drmaxr0 = rmaxr0_max - rmaxr0
        drmaxr0_thresh = .000001

        while np.abs(drmaxr0) >= drmaxr0_thresh:
            r0 = rmax/rmaxr0_new
            Nr = 100000
            rrfracr0_E04, MMfracM0_E04 = E04_outerwind_r0input_nondim_MM0(r0, fcor, Cdvary, C_d, w_cool, Nr)

            rrfracr0_ER11 = rrfracrm_ER11*(rmaxr0_new)
            M0_E04 = .5*fcor*r0**2
            MMfracM0_ER11 = MMfracMm_ER11*(Mm/M0_E04)

            X0, Y0 = curveintersect(rrfracr0_E04, MMfracM0_E04, rrfracr0_ER11, MMfracM0_ER11)
            if X0.size == 0:
                drmaxr0 = abs(drmaxr0)/2
            else:
                drmaxr0 = -abs(drmaxr0)/2
                rmerger0 = np.mean(X0)
                MmergeM0 = np.mean(Y0)

            rmaxr0 = rmaxr0_new
            rmaxr0_new = rmaxr0_new + drmaxr0

        M0 = .5*fcor*r0**2
        Mm = .5*fcor*rmax**2 + rmax*Vmax

        ii_ER11 = np.argwhere((rrfracr0_ER11 < rmerger0) & (MMfracM0_ER11 < MmergeM0))[:,0]
        ii_E04 = np.argwhere((rrfracr0_E04 >= rmerger0) & (MMfracM0_E04 >= MmergeM0))[:,0]
        MMfracM0_temp = np.hstack((MMfracM0_ER11[ii_ER11], MMfracM0_E04[ii_E04]))
        rrfracr0_temp = np.hstack((rrfracr0_ER11[ii_ER11], rrfracr0_E04[ii_E04]))

        drfracrm = .01
        rfracrm_min = 0
        rfracrm_max = r0/rmax
        rrfracrm = np.arange(rfracrm_min, rfracrm_max+drfracrm+drfracrm, drfracrm)
        f = PchipInterpolator(rrfracr0_temp*(r0/rmax), MMfracM0_temp*(M0/Mm))
        MMfracMm = f(rrfracrm)

        rrfracr0 = rrfracrm*rmax/r0
        MMfracM0 = MMfracMm*Mm/M0

        with np.errstate(invalid='ignore'):
            VV = (Mm/rmax)*(MMfracMm/rrfracrm) - .5*fcor*rmax*rrfracrm
        rr = rrfracrm*rmax

        VV[rr==0] = 0

        rmerge = rmerger0*r0
        Vmerge = (M0/r0)*((MmergeM0/rmerger0) - rmerger0)

    return rr, VV, r0, rmerge, Vmerge
