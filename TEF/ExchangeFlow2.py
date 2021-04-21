# Code to calculate Two-Layer Exchange Flow

import xarray as xr
import numpy as np
import tef_fun_lorenz as tfl

def ef_Eu(U, S, DA, xi):
    ## Eulerian decomposition
    # - u: Along-channel velocity m/s (t,z, x)
    # - s: Salinity psu (t, z, x)
    # - da: Cell face area (t,z, x)
    
    #s = S[:,:,:,xi]
    #u = U[:,:,:,xi]
    #da = DA[:,:,:,xi]
    
    NT, NZ, NY, NX = S.shape
    
    s = S.mean(2)
    u = U.mean(2)
    da = DA.mean(2)
    sma = np.ma.masked_where(s==0, s)
    topo = np.ma.getmask(sma) # Masked Topography
    uma = np.ma.MaskedArray(u, mask=topo)
    
    # Output variables 
    Q1 = np.zeros(NT)
    Q2 = np.zeros(NT)
    S1 = np.zeros(NT)
    S2 = np.zeros(NT)
    
    for tt in range(NT):
                    
        uz = uma[tt,:,xi]
        sz = sma[tt,:,xi]
        
        #uy = u[tt,:,:]
        #sy = s[tt,:,:]
        #area = da[tt,:,:]
        
        #q = uy * area
        #qs = q * sy
        
        #Q1[tt] = q[q>0].sum()
        #Q2[tt] = q[q<0].sum()
        #S1[tt] = qs[qs>0].sum() / q[q>0].sum()
        #S2[tt] = qs[qs<0].sum() / q[q<0].sum()
        if any(uz<0)==False:
            pass
            
        else:
            l = np.where(uz < 0)[-1][0] 
            Q1[tt] = np.sum(uz[:l]*da[tt,:l,xi]) * NY
            Q2[tt] = np.sum(uz[l:]*da[tt,l:,xi]) * NY
            S1[tt] = np.sum(sz[:l]*da[tt,:l,xi]*uz[:l]) / np.sum(da[tt,:l,xi]*uz[:l])
            S2[tt] = np.sum(sz[l:]*da[tt,l:,xi]*uz[l:]) / np.sum(da[tt,l:,xi]*uz[l:])
            
    return Q1, Q2, S1, S2




def ef_TEF(u, s, da, xi):
    # Isohaline decomposition (TEF, MacCready 2011)
    
    NT, NZ, NX = s.shape
    
    # Output variables 
    Qin = np.zeros(NT)
    Qout = np.zeros(NT)
    Sin = np.zeros(NT)
    Sout = np.zeros(NT)
    
    # initialize intermediate results arrays for TEF quantities
    sedges = np.linspace(0, 35, 1001)
    sbins = sedges[:-1] + np.diff(sedges)/2
    NS = len(sbins) # number of salinity bins
    
    sma = np.ma.masked_where(s==0, s)
    topo = np.ma.getmask(sma) # Masked Topography
    uma = np.ma.MaskedArray(u, mask=topo)
    q = uma * da
    qs = q * sma
    tef_q = np.zeros((NT, NS))
    tef_qs = np.zeros((NT, NS))
    
        
    for tt in range(NT):
        
        qi = q[tt,:,xi].squeeze()
        if isinstance(qi, np.ma.MaskedArray):
            qf = qi[qi.mask==False].data.flatten()
        else:
            qf = qi.flatten()
            
        qsi = qs[tt,:,xi].squeeze()
        if isinstance(qsi, np.ma.MaskedArray):
            qsf = qsi[qi.mask==False].data.flatten()
        else:
            qsf = qsi.flatten()
        
        
        si = sma[tt,:,xi].squeeze()
        if isinstance(si, np.ma.MaskedArray):
            sf = si[qi.mask==False].data.flatten()
        else:
            sf = si.flatten()
        
        # sort into salinity bins
        inds = np.digitize(sf, sedges, right=True)
        indsf = inds.copy().flatten()
        counter = 0
        for ii in indsf:
            tef_q[tt, ii-1] += qf[counter]
            tef_qs[tt, ii-1] += qsf[counter]
            counter += 1


      
    tef_q_lp = tef_q[275:,:].mean(0)
    tef_qs_lp = tef_qs[275:,:].mean(0)
    Qv = np.zeros((NT, NS))
    Qs = np.zeros((NT, NS))
    # Organized from low s to high s
    Qv = np.fliplr(np.cumsum(np.fliplr(tef_q), axis=1))
    Qs = np.fliplr(np.cumsum(np.fliplr(tef_qs), axis=1))
    
 
    
    for t in range(NT):
        
        Q_in_m = []
        Q_out_m = []
        s_in_m = []
        s_out_m = []
        
        # use the find_extrema algorithm
        ind, minmax = tfl.find_extrema(Qv[t,:], print_info=False)
        
        index=[]   
        i=0
        while i < len(ind)-1:
        # compute the transports and sort to in and out
            Q_i = -(Qv[t,ind[i+1]]-Qv[t,ind[i]])
            F_i = -(Qs[t,ind[i+1]]-Qs[t,ind[i]])
            s_i = np.abs(F_i) / np.abs(Q_i)
            
            if Q_i<0 and np.abs(Q_i)>1:
                Q_out_m.append(Q_i)
                s_out_m.append(s_i)

            elif Q_i > 0 and np.abs(Q_i)>1:
                Q_in_m.append(Q_i)
                s_in_m.append(s_i)
        
            else:
                index.append(i)
            i+=1
        
        qq = np.concatenate((Q_in_m, Q_out_m))
        NL = len(qq)
        QQ=np.zeros(NL)
        QQ[:NL] = qq
        
        ss = np.concatenate((s_in_m, s_out_m))
        SS=np.zeros(NL)
        SS[:NL] = ss
        
        # separate positive and negative transports
        QQp = QQ.copy()
        QQp[QQ<=0] = np.nan
        QQm = QQ.copy()
        QQm[QQ>=0] = np.nan
        
        # full transports
        QQm = QQm
        QQp = QQp
        
        QSp = np.nansum(QQp*SS, axis=0)
        QSm = np.nansum(QQm*SS, axis=0)
        
        Qp = np.nansum(QQp, axis=0)
        Qm = np.nansum(QQm, axis=0)
        
        # TEF salinities
        Sp = QSp/Qp
        Sm = QSm/Qm
        
        Qin[t] = Qp
        Qout[t] = Qm
        Sin[t] = Sp
        Sout[t] = Sm
        
        # adjust sign convention so that positive flow is salty
        #SP = np.nanmean(Sp)
        #SM = np.nanmean(Sm)
        #if SP > SM:       
            # initial positive was inflow
        #    Sin[t] = Sp
        #    Sout[t] = Sm
        #    Qin[t] = Qp
        #    Qout[t] = Qm
            
        
        #elif SM > SP:
            # initial postive was outflow
        #    Sin[t] = Sm
        #    Sout[t] = Sp
        #    Qin[t] = -Qm
        #    Qout[t] = -Qp
        #else:
        #    print('ambiguous sign!!')
        
    return Qin, Qout, Sin, Sout
            
        