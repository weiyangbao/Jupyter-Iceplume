import matplotlib.pyplot as plt
import numpy as np
import os, sys
import tef_fun_lorenz as tfl

def process_section(U, S, DA, ot, xi, Sc, testing=False):
    
    """
    Process TEF extractions, giving transport vs. salinity for:
    volume, salt, and salinity-squared.

    Input:
    - U: Along-channel velocity in T, Z, Y, X
    - S: Salinity in T, Z, Y, X
    - DA: Cell area in T, Z, Y, X
    - ot: Ocean time in seconds
    - xi: Location index of the section to be processed
    - Sc: The salinity class to be plot in testing

    Output:
    - tef_q: transport in salinity bins, hourly, (m3/s)
    - tef_vel: velocity in salinity bins, hourly, (m/s)
    - tef_da: area in salinity bins, hourly, (m2)
    - tef_qs: salt transport in salinity bins, hourly, (g/kg m3/s)
    - tef_qs2: salinity-squared transport in salinity bins, hourly, (g2/kg2 m3/s)
    - sbins: the salinity bin centers

    """
    
    s = S[:,:,:,xi]
    u = U[:,:,:,xi]
    da = DA[:,:,:,xi]
    sma = np.ma.masked_where(s==0, s)
    topo = np.ma.getmask(sma) # Masked Topography
    uma = np.ma.MaskedArray(u, mask=topo)
    q = uma * da
    qs = q * sma
    qs2 = q * sma * sma

    NT, NZ, NX = q.shape
    # initialize intermediate results arrays for TEF quantities
    sedges = np.linspace(0, 35, 1001) 
    sbins = sedges[:-1] + np.diff(sedges)/2
    NS = len(sbins) # number of salinity bins

    # TEF variables
    tef_q = np.zeros((NT, NS))
    tef_vel = np.zeros((NT, NS))
    tef_da = np.zeros((NT, NS))
    tef_qs = np.zeros((NT, NS))
    tef_qs2 = np.zeros((NT, NS))

    # other variables
    qnet = np.zeros(NT)
    fnet = np.zeros(NT)
    ssh = np.zeros(NT)
    g = 9.81
    rho = 1025

    for tt in range(NT):
        if np.mod(tt,1000) == 0:
            print('  time %d out of %d' % (tt,NT))
            sys.stdout.flush()
            
        qi = q[tt,:,:].squeeze()
        if isinstance(qi, np.ma.MaskedArray):
            qf = qi[qi.mask==False].data.flatten()
        else:
            qf = qi.flatten()
            
        si = sma[tt,:,:].squeeze()
        if isinstance(si, np.ma.MaskedArray):
            sf = si[qi.mask==False].data.flatten()
        else:
            sf = si.flatten()
            
        dai = da[tt,:,:].squeeze()
        if isinstance(dai, np.ma.MaskedArray):
            daf = dai[qi.mask==False].data.flatten()
        else:
            daf = dai.flatten()
            
        qsi = qs[tt,:,:].squeeze()
        if isinstance(qsi, np.ma.MaskedArray):
            qsf = qsi[qi.mask==False].data.flatten()
        else:
            qsf = qsi.flatten()
            
        qs2i = qs2[tt,:,:].squeeze()
        if isinstance(qs2i, np.ma.MaskedArray):
            qs2f = qs2i[qi.mask==False].data.flatten()
        else:
            qs2f = qs2i.flatten()
            
        # sort into salinity bins
        inds = np.digitize(sf, sedges, right=True)
        indsf = inds.copy().flatten()
        counter = 0
        for ii in indsf:
            tef_q[tt, ii-1] += qf[counter]
            tef_da[tt, ii-1] += daf[counter] # new
            tef_qs[tt, ii-1] += qsf[counter]
            tef_qs2[tt, ii-1] += qs2f[counter]
            counter += 1
        
        # also keep track of volume transport
        qnet[tt] = qf.sum()
        


    # Calculating the velocity from transport/area
    # NOTE: we require tef_q = tef_vel * tef_da
    zmask = tef_da > 1 # Is this about the right number (1 m2)?
    # Just avoiding divide-by-zero errors.
    tef_vel[zmask] = tef_q[zmask] / tef_da[zmask]


    if testing:

        plt.close('all')
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        iSc = (np.abs(sbins-Sc)).argmin()
        ax.plot((ot-ot[0])/3600, tef_q[:,iSc], '-r', lw=2,label=r'tef_q')
        ax.plot((ot-ot[0])/3600, tef_vel[:,iSc]*tef_da[:,iSc], '-b', lw=1,label=r'ref_vel*tef_da')
        ax.legend(loc='best',fontsize=15)
        ax.set_xlabel(r'Time (h)',size=15)
        ax.set_ylabel(r'Transport ($m^3/s$)',size=15)
        ax.set_title('Q(s=' + str(Sc) + ',x=' + str(xi) + ')',fontsize=15)
        plt.show()
        
    return tef_q, tef_vel, tef_da, tef_qs, tef_qs2, sbins





def bulk_calc(tef_q, tef_vel, tef_da, tef_qs, tef_qs2, sbins, ot):
    """
    Code to calculate a TEF time series using Marvin Lorenz' new multi-layer code.
    
    # Form time series of 2-layer TEF quantities, from the multi-layer bulk values.
    
    """
    pad = 0
    # subsample
    tef_q_lp = tef_q[pad:-(pad+1):1, :]
    tef_vel_lp = tef_vel[pad:-(pad+1):1, :]
    tef_da_lp = tef_da[pad:-(pad+1):1, :]
    tef_qs_lp = tef_qs[pad:-(pad+1):1, :]
    tef_qs2_lp = tef_qs2[pad:-(pad+1):1, :]
    #ot = ot[pad:-(pad+1):1]
    #qnet_lp = qnet[pad:-(pad+1):1]

    # get sizes and make sedges (the edges of sbins)
    DS = sbins[1]-sbins[0]
    sedges = np.concatenate((sbins,np.array([sbins[-1]] + DS))) - DS/2
    NT = len(ot)
    NS = len(sedges)


    # calculate Q(s) and Q_s(s), and etc.
    Qv=np.zeros((NT, NS))
    Vv=np.zeros((NT, NS))
    Av=np.zeros((NT, NS))
    Qs=np.zeros((NT, NS))
    Qs2=np.zeros((NT, NS))
    # Note that these are organized low s to high s, but still follow
    # the TEF formal definitions from MacCready (2011)
    Qv[:,:-1] = np.fliplr(np.cumsum(np.fliplr(tef_q), axis=1))
    Vv[:,:-1] = tef_vel
    Av[:,:-1] = tef_da
    Qs[:,:-1] = np.fliplr(np.cumsum(np.fliplr(tef_qs), axis=1))
    Qs2[:,:-1] = np.fliplr(np.cumsum(np.fliplr(tef_qs2), axis=1))
    
    # prepare arrays to hold multi-layer output
    nlay = 30
    QQ = np.nan * np.ones((NT, nlay))
    VV = np.nan * np.ones((NT, nlay))
    AA = np.nan * np.ones((NT, nlay))
    SS = np.nan * np.ones((NT, nlay))
    SS2 = np.nan * np.ones((NT, nlay))
    
    
    for dd in range(NT):
            
        qv = Qv[dd,:]
        vv = Vv[dd,:]
        av = Av[dd,:]
        qs = Qs[dd,:]
        qs2 = Qs2[dd,:]
                
        Q_in_m, Q_out_m, V_in_m, V_out_m, A_in_m, A_out_m, s_in_m, s_out_m, s2_in_m, s2_out_m, div_sal, ind, minmax =             tfl.calc_bulk_values(sedges, qv, vv, av, qs, qs2, print_info=False)
                        
        
   
    #print(' ind = %s' % (str(ind)))
    #print(' minmax = %s' % (str(minmax)))
    #print(' div_sal = %s' % (str(div_sal)))
    #print(' Q_in_m = %s' % (str(Q_in_m)))
    #print(' s_in_m = %s' % (str(s_in_m)))
    #print(' Q_out_m = %s' % (str(Q_out_m)))
    #print(' s_out_m = %s' % (str(s_out_m)))
        
    #fig = plt.figure(figsize=(12,8))
        
    #ax = fig.add_subplot(121)
    #ax.plot(Qv[dd,:], sedges,'.k')
    #min_mask = minmax=='min'
    #max_mask = minmax=='max'
    #print(min_mask)
    #print(max_mask)
    #ax.plot(Qv[dd,ind[min_mask]], sedges[ind[min_mask]],'*b')
    #ax.plot(Qv[dd,ind[max_mask]], sedges[ind[max_mask]],'*r')
    #ax.grid(True)
    #ax.set_title('Q(s) Time index = %d' % (dd))
    #ax.set_ylim(-.1,35.1)
    #ax.set_ylabel('Salinity')
        
    #ax = fig.add_subplot(122)
    #ax.plot(tef_q_lp[dd,:], sbins)
    #ax.grid(True)
    #ax.set_title('-dQ/ds')
        
        # save multi-layer output
        qq = np.concatenate((Q_in_m, Q_out_m))
        vv = np.concatenate((V_in_m, V_out_m))
        aa = np.concatenate((A_in_m, A_out_m))
        ss = np.concatenate((s_in_m, s_out_m))
        ss2 = np.concatenate((s2_in_m, s2_out_m))
        ii = np.argsort(ss)
        if len(ii)>0:
            ss = ss[ii]
            ss2 = ss2[ii]
            qq = qq[ii]
            vv = vv[ii]
            aa = aa[ii]
            NL = len(qq)
            QQ[dd, :NL] = qq
            VV[dd, :NL] = vv
            AA[dd, :NL] = aa
            SS[dd, :NL] = ss
            SS2[dd, :NL] = ss2

        dd+=1
    
    
    
    # separate positive and negative transports
    QQp = QQ.copy()
    QQp[QQ<=0] = np.nan
    QQm = QQ.copy()
    QQm[QQ>=0] = np.nan
        
    # full transports
    QQm = QQm
    QQp = QQp
    
    # form two-layer versions
    Qp = np.nansum(QQp, axis=1)
    QSp = np.nansum(QQp*SS, axis=1)
    QS2p = np.nansum(QQp*SS2, axis=1)
    Qm = np.nansum(QQm, axis=1)
    QSm = np.nansum(QQm*SS, axis=1)
    QS2m = np.nansum(QQm*SS2, axis=1)
    
    # TEF salinities
    Sp = QSp/Qp
    Sm = QSm/Qm
    
    # TEF variance
    S2p = QS2p/Qp
    S2m = QS2m/Qm
    
    Qin = Qp
    Qout = Qm
    Sin = Sp
    Sout = Sm
    
    
    # adjust sign convention so that positive flow is salty
    #SP = np.nanmean(Sp)
    #SM = np.nanmean(Sm)
    #if SP > SM:
        
        # initial positive was inflow
    #    Sin = Sp
    #    Sout = Sm
    #    S2in = S2p
    #    S2out = S2m
    #    Qin = Qp
    #    Qout = Qm
    #    in_sign = 1
        
    #    QSin = QSp
    #    QSout = QSm
        
    #    QS2in = QS2p
    #    QS2out = QS2m
        
    #elif SM > SP:
        # initial postive was outflow
    #    Sin = Sm
    #    Sout = Sp
    #    S2in = S2m
    #    S2out = S2p
    #    Qin = -Qm
    #    Qout = -Qp
    #    in_sign = -1
        
    #    QSin = -QSm
    #    QSout = -QSp
        
    #    QS2in = -QS2m
    #    QS2out = -QS2p
    #else:
    #    print('ambiguous sign!!')   
        
    return Qin, Qout, Sin, Sout
