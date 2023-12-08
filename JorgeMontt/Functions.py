"""
Functions for Fjord Modeling.
"""
import numpy as np
import xarray as xr
import gsw
import os, sys
sys.path.append(os.path.abspath('../TEF'))
import TEF_Variables as tef


def vol_temp(datapath, case_id, xi):
    # Calculate the volume temperature (Used to check steady state)
    # Control volume between cross-fjord sections at xi[0] and xi[-1]
    State0 = xr.open_dataset(datapath+'state_' + str(format(case_id,'03d')) + '.nc')
    Grid = xr.open_dataset(datapath+'grid.nc')
    State = State0.isel(T=~State0.get_index("T").duplicated())    
    
    # Confine to the range of fjord
    state = State.isel(X=slice(xi[0],xi[1]), Xp1=slice(xi[0],xi[1]+1), Y=slice(35, 45))
    grid = Grid.isel(X=slice(xi[0],xi[1]), Xp1=slice(xi[0],xi[1]+1), Y=slice(35, 45))

    #U = (state.U.data[:,:,:,1:] + state.U.data[:,:,:,:-1]) / 2 # Along-channel velocity
    Time = state.T.data
    
    drF = np.broadcast_to(grid.drF.data[np.newaxis, :, np.newaxis, np.newaxis], state.Temp.data.shape)
    dyF = np.broadcast_to(grid.dyF.data[np.newaxis, np.newaxis, :, :], state.Temp.data.shape)
    HFacC = np.broadcast_to(grid.HFacC.data[np.newaxis, :, :, :], state.Temp.data.shape)
    DA = drF * dyF * HFacC
    
    da = DA.mean(2)
    s = state.S.data.mean(2)
    temp = state.Temp.data.mean(2)
    sma = np.ma.masked_where(s==0, s)
    topo = np.ma.getmask(sma) # Masked Topography
    tma = np.ma.MaskedArray(temp, mask=topo)
    
    temp_tx = np.sum(tma * da, axis=1) / np.sum(da, axis=1)
    vTemp = np.mean(temp_tx.data, axis=1)
        
    return vTemp, Time



def volflux_ts(datapath, case_id, xid):
    # Calculate along-fjord volume fluxes at cross-section xid
    State0 = xr.open_dataset(datapath+'state_' + str(format(case_id,'03d')) + '.nc')
    State = State0.isel(T=~State0.get_index("T").duplicated())
    Grid = xr.open_dataset(datapath+'grid.nc')

    state = State.isel(X=slice(200), Xp1=slice(201), Y=slice(35,45))
    grid = Grid.isel(X=slice(200), Xp1=slice(201), Y=slice(35,45))  
    
    time = state.T.data / 86400
    
    # Area = np.empty([90, 10]) # Cross Y direction
    # Area[:20, :] = 400
    # Area[20:50, :] = 800
    # Area[50:, :] = 1200
    # HFacC = grid.HFacC.data[:,:,xid] # vertical fraction of open cell at Center

    area = grid.drF * grid.dyF * grid.HFacC
    U = (state.U.data[:,:,:,1:] + state.U.data[:,:,:,:-1]) / 2
    Q1 = np.empty(len(time))
    Q2 = np.empty(len(time))
    for t in range(len(time)):
        #Q = U[t,:,:,xid] * Area * HFacC
        Q = U[t,:,:,xid] * area.data[:,:,xid]
        Q1[t] = Q[Q > 0].sum()
        Q2[t] = Q[Q < 0].sum()
    
    return time, Q1, Q2



def godin_shape():
    """
    Based on matlab code of 4/8/2013  Parker MacCready
    Returns a 71 element numpy array that is the weights
    for the Godin 24-24-25 tildal averaging filter. This is the shape given in
    Emery and Thomson (1997) Eqn. (5.10.37)
    ** use ONLY with hourly data! **
    """
    k = np.arange(12)
    filt = np.NaN * np.ones(71)
    filt[35:47] = (0.5/(24*24*25))*(1200-(12-k)*(13-k)-(12+k)*(13+k))
    k = np.arange(12,36)
    filt[47:71] = (0.5/(24*24*25))*(36-k)*(37-k)
    filt[:35] = filt[:35:-1]
    return filt

def filt_godin(data):
    """
    Input: 1D numpy array of HOURLY values
    Output: Array of the same size, filtered with 24-24-25 Godin filter,
        padded with nan's
    """
    filt = godin_shape()
    npad = np.floor(len(filt)/2).astype(int)
    smooth = np.convolve(data, filt, mode = 'same')
    smooth[:npad] = np.nan
    smooth[-npad:] = np.nan
    # smooth[:npad] = data[:npad]
    # smooth[-npad:] = data[-npad:]
    return smooth



def filt_godin_mat(data):
    """
    Input: ND numpy array of HOURLY, with time on axis 0.
    Output: Array of the same size, filtered with 24-24-25 Godin filter,
        padded with nan's
    """
    filt = godin_shape()
    n = np.floor(len(filt)/2).astype(int)
    sh = data.shape
    df = data.flatten('F')
    dfs = np.convolve(df, filt, mode = 'same')
    smooth = dfs.reshape(sh, order='F')
    smooth[:n,:] = np.nan
    smooth[-n:,:] = np.nan
    return smooth



def tef_transport(datapath, case_id, xid):    
    # Calculate TEF transports at cross-fjord section xid
    State0 = xr.open_dataset(datapath+'state_' + str(format(case_id,'03d')) + '.nc')
    Grid = xr.open_dataset(datapath+'grid.nc')
    State = State0.isel(T=~State0.get_index("T").duplicated())
    
    # Confine to the range of fjord
    state = State.isel(X=slice(200), Xp1=slice(201), Y=slice(35,45))
    grid = Grid.isel(X=slice(200), Xp1=slice(201), Y=slice(35,45))
    

    ot = state.T.data # Time in seconds
    S = state.S.data # Salinity
    U = (state.U.data[:,:,:,1:] + state.U.data[:,:,:,:-1]) / 2 # Along-fjord velocity
    # Grid area
    drF = np.broadcast_to(grid.drF.data[np.newaxis, :, np.newaxis, np.newaxis], U.shape)
    dyF = np.broadcast_to(grid.dyF.data[np.newaxis, np.newaxis, :, :], U.shape)
    HFacC = np.broadcast_to(grid.HFacC.data[np.newaxis, :, :, :], U.shape)
    DA = drF * dyF * HFacC       
               
    tef_q1, tef_vel1, tef_da1, tef_qs1, tef_qs21, sbins1 = tef.process_section(U,S,DA,ot,xid,23,testing=False)

    qin, qout, sin, sout = tef.bulk_calc(tef_q1, tef_vel1, tef_da1, tef_qs1, tef_qs21, sbins1, ot)
    
        
    return qin, qout, sin, sout



def along_fjord_state(datapath, case_id):
    # Get along-fjord properties in Time, Depth, Distance dimensions
    
    file0 = xr.open_dataset(datapath+'state_' + str(format(case_id,'03d')) + '.nc')
    # De-duplicating data
    file = file0.isel(T=~file0.get_index("T").duplicated())
    state = file.isel(X=slice(200), Xp1=slice(201), Y=slice(35, 45), Yp1=slice(35, 45))
    
    # Extracting data and converting to NumPy arrays
    time = state.T.data
    X = state.X.data
    x_dist = X / 1000
    depth = state.Z.data
    pres = gsw.p_from_z(depth, -48.25)
    pt = state.Temp.mean('Y').data
    s = state.S.mean('Y').data
    p = np.broadcast_to(pres[np.newaxis, :, np.newaxis], pt.shape)
    rho = gsw.rho(s, pt, p)

    # Along-fjord velocity
    u0 = state.U.mean('Y').data
    u = (u0[:, :, 1:] + u0[:, :, :-1]) / 2

    # Vertical velocity
    W0 = state.W.data
    bt = np.zeros((W0.shape[0], 1, W0.shape[2], W0.shape[3]))  # Bottom vertical velocity (0 m/s)
    W = np.concatenate((W0, bt), axis=1)
    wzy = (W[:, 1:, :, :] + W[:, :-1, :, :]) / 2
    w = wzy.mean(axis=2)

    return x_dist, depth, time, pt, s, rho, u, w
    