# Code to calculate Two-Layer Exchange Flow

import xarray as xr
import numpy as np

def ef_Eu(u, s, da, xi):
    ## Eulerian decomposition
    # - u: Along-channel velocity m/s (t,z, x)
    # - s: Salinity psu (t, z, x)
    # - da: Cell face area (t,z, x)
    #mask = (s == 0) # Exclude data covered by the sill
    #s[mask] = np.nan
    #u[mask] = np.nan
    NT, NZ, NX = s.shape
    
    # Output variables 
    Q1 = np.zeros(NT)
    Q2 = np.zeros(NT)
    S1 = np.zeros(NT)
    S2 = np.zeros(NT)
    
    for tt in range(NT):
                    
        uz = u[tt,:,xi]
        sz = s[tt,:,xi]
            
        if any(uz<0)==False:
            pass
            
        else:
            l = np.where(uz < 0)[-1][0] 
            Q1[tt] = np.sum(uz[:l]*da[tt,:l,xi])
            Q2[tt] = np.sum(uz[l:]*da[tt,l:,xi])
            S1[tt] = np.sum(sz[:l]*da[tt,:l,xi]) / da[tt,:l,xi].sum()
            S2[tt] = np.sum(sz[l:]*da[tt,l:,xi]) / da[tt,l:,xi].sum()
            
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
    
    q = u * da
    qs = q * s
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
        
        si = s[tt,:,xi].squeeze()
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
        
        index=[]   
        i=0
        while i < Qv.shape[1]-1:
        # compute the transports and sort to in and out
            Q_i = -(Qv[t,i+1]-Qv[t,i])
            F_i = -(Qs[t,i+1]-Qs[t,i])
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
        
    return Qin, Qout, Sin, Sout
            
        