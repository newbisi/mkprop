import scipy.sparse
import scipy.linalg
import numpy as np
from .Krylovprop import *

def midpoint(u,tnow,tend,dtfixed,prob,ktol=1e-8,ktype=1,reo=1,nrm=None, inr=None):
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    dt = min(dtfixed,tend-tnow)
    y0=u
    tlist, dtlist = [], []
    while (tnow<tend):
        mv, dmv = prob.setupHamiltonian(tnow+0.5*dt)
        y1, errest, tlist, mlist = expimv_pKry(mv,y0,t=dt,tol=ktol,
                                       ktype=ktype,reo=reo,nrm=nrm,inr=inr)
        y0=y1
        tnow += dt
        dt = min(dtfixed,tend-tnow)
    return y1, tlist, dtlist


def adaptivemidpoint(u,tnow,tend,dtinit,prob,tol=1e-8,m=30,ktype=1,reo=1,nrm=None, inr=None):
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    y0=u
    p=2
    dt = min(dtinit,tend-tnow)
    sig = 1j
    tlist=[]
    dtlist=[]
    errmlist=[]
    spstep=0.9
    spstepfail=0.7
    errpartm=0.99
    tolkry=(1-errpartm)*tol
    mcolist1=[]
    mcolist2=[]
    mcolist3=[]
    mco1=0
    mco2=0
    mco3=0
    while (tnow<tend):
        anow = [1.0]
        cnow = [0.5]
        chat = 0
        mv, dmv = prob.setupHamiltonianCFM(anow,cnow,chat,tnow,dt)
        y1, errestkry, tkrylist, mlist = expimv_pKry(mv,y0,tol=tolkry,t=dt,
                                               m=m,ktype=ktype,reo=reo,nrm=nrm,inr=inr)
        
        # computer error estimate
        dHpsi = dmv(y1)
        HdHpsi = mv(dHpsi)
        Hpsi = mv(y1)
        dHHpsi = dmv(Hpsi)
        defp1 = Hpsi + dt*dHpsi + sig*dt**2/2*(HdHpsi-dHHpsi)
        
        mv2, _ = prob.setupHamiltonian(tnow+dt)
        def1 = defp1-mv2(y1)
        errestmag = dt/(p+1)*nrm(def1)

        mco1+=sum(mlist)
        mco3+=2
        # test error estimate
        if errestkry+errestmag<dt*tol:
            y0 = y1
            tnow += dt
            dtlist.append(dt)
            tlist.append(tnow)
            errmlist.append(errestmag)
            mcolist1.append(mco1)
            mcolist2.append(mco2)
            mcolist3.append(mco3)
            mco1 = 0
            mco2 = 0
            mco3 = 0
            dtnew = spstep * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
        else:
            mco2 += mco1+mco3
            mco1 = 0
            mco3 = 0
            dtnew = spstepfail * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
    return y1,tlist,dtlist,errmlist,mcolist1,mcolist2,mcolist3

def adaptivemidpoint_symdef(u,tnow,tend,dtinit,prob,tol=1e-8,m=30,ktype=1,reo=1,nrm=None, inr=None):
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    y0=u
    p=2
    dt = min(dtinit,tend-tnow)
    sig = 1j
    tlist=[]
    dtlist=[]
    errmlist=[]
    spstep=0.9
    spstepfail=0.7
    errpartm=0.99
    tolkry=(1-errpartm)*tol
    mv0, _ = prob.setupHamiltonian(tnow)
    y0e = mv0(y0)
    while (tnow<tend):
        mv, dmv = prob.setupHamiltonian(tnow+0.5*dt)
        y1, errestkry, tkrylist, mlist = expimv_pKry(mv,y0,tol=tolkry,t=dt,m=m,ktype=ktype,
                                                                reo=reo,nrm=nrm,inr=inr)
        y0esym=mv(y0)-y0e
        defp2, errestkry, tkrylist, mlist = expimv_pKry(mv,y0esym,tol=tolkry,t=dt,m=m,
                                                                   ktype=ktype,reo=reo,nrm=nrm,inr=inr)
    
        # computer error estimate
        gam2 = mv(y1)
        mv2, _ = prob.setupHamiltonian(tnow+dt)
        y1enext = mv2(y1)
        def1 = 0.5*(gam2+defp2-y1enext)
        errestmag = dt/(p+1)*nrm(def1)
        
        # test error estimate
        if errestkry+errestmag<dt*tol:
            y0 = y1
            y0e = y1enext
            tnow += dt
            dtlist.append(dt)
            tlist.append(tnow)
            errmlist.append(errestmag)
            dtnew = spstep * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
        else:
            dtnew = spstepfail * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
    return y1,tlist,dtlist,errmlist


def CFM4(u,tnow,tend,dtfixed,prob,ktol=1e-8,m=30,ktype=1,reo=1,nrm=None, inr=None):
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    y0=u

    c = 3**0.5/6
    cmat = [0.5-c, 0.5+c]
    amat=[[0.25+c,0.25-c],[0.25-c,0.25+c]]
    chat = -0.5
    jexps=len(cmat)
    
    dt = min(dtfixed,tend-tnow)
    nexps=1
    sig = 1j
    tlist=[]
    dtlist=[]
    errmlist=[]
    while (tnow<tend):
        
        know=0
        anow = amat[know]
        mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tnow,dt)
        y1sub, errestkry, tkrylist, mlist = expimv_pKry(mv,y0,tol=ktol,t=dt,m=m,
                                                                   ktype=ktype,reo=reo,nrm=nrm,inr=inr)
    
        know=1
        anow = amat[know]
        mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tnow,dt)
        y1, errestkry, tkrylist, mlist = expimv_pKry(mv,y1sub,tol=ktol,t=dt,m=m,
                                                                ktype=ktype,reo=reo,nrm=nrm,inr=inr)
    
        y0 = y1
        tnow += dt
        dtlist.append(dt)
        tlist.append(tnow)
    return y1, dtlist, tlist

def adaptiveCFM4_old(u,tnow,tend,dtinit,prob,tol=1e-8,m=30,ktype=1,reo=1,nrm=None, inr=None):
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    c = 3**0.5/6
    cmat = [0.5-c, 0.5+c]
    amat=[[0.25+c,0.25-c],[0.25-c,0.25+c]]
    chat = -0.5
    parth1 = [1,0]
    parth2 = [0,1]
    p=4
    dt = min(dtinit,tend-tnow)

    sig = 1j
    tlist=[]
    dtlist=[]
    errmlist=[]
    spstep=0.9
    spstepfail=0.7
    errpartm=0.99
    tolkry=(1-errpartm)*tol

    y0=u
    mv0, _ = prob.setupHamiltonian(tnow)
    ye0 = mv0(y0)
    ye0 = ye0+0j
    while (tnow<tend):
        y0sub=y0
        e0sub=-0.5*ye0
    
        know=0
        anow = amat[know]
        mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tnow,dt)
    
        dHpsi = dmv(y0sub)
        Hpsi = mv(y0sub)
        HdHpsi = mv(dHpsi)
        dHHpsi = dmv(Hpsi)
        Gam11 = dt/2*dHpsi - sig*dt**2*1/12*(HdHpsi-dHHpsi)
        if parth1[0]>0:
            Gam11 += parth1[0]*Hpsi 
        e0sub += Gam11
    
        y1sub, errestkry, tkrylist, mlist = expimv_pKry(mv,y0sub,tol=tolkry,t=dt,m=m,ktype=2,reo=0)
        e1sub, errestkry, tkrylist, mlist = expimv_pKry(mv,e0sub,tol=tolkry,t=dt,m=m,ktype=2,reo=0)
    
        dHpsi = dmv(y1sub)
        Hpsi = mv(y1sub)
        HdHpsi = mv(dHpsi)
        dHHpsi = dmv(Hpsi)
        Gam12 = dt/2*dHpsi + sig*dt**2*1/12*(HdHpsi-dHHpsi)
        if parth2[0]>0:
            Gam12 += parth2[0]*Hpsi 
        e1sub += Gam12
    
        know=1
        anow = amat[know]
        mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tnow,dt)
    
        dHpsi = dmv(y1sub)
        Hpsi = mv(y1sub)
        HdHpsi = mv(dHpsi)
        dHHpsi = dmv(Hpsi)
        Gam21 = dt/2*dHpsi - sig*dt**2*1/12*(HdHpsi-dHHpsi)
        if parth1[1]>0:
            Gam21 += parth1[1]*Hpsi 
        e1sub += Gam21
    
        y2sub, errestkry, tkrylist, mlist = expimv_pKry(mv,y1sub,tol=tolkry,t=dt,m=m,ktype=2,reo=0)
        e2sub, errestkry, tkrylist, mlist = expimv_pKry(mv,e1sub,tol=tolkry,t=dt,m=m,ktype=2,reo=0)
    
        dHpsi = dmv(y2sub)
        Hpsi = mv(y2sub)
        HdHpsi = mv(dHpsi)
        dHHpsi = dmv(Hpsi)
        Gam22 = dt/2*dHpsi + sig*dt**2*1/12*(HdHpsi-dHHpsi)
        if parth2[1]>0:
            Gam22 += parth2[1]*Hpsi 
        e2sub += Gam22
    
        # computer error estimate
    
        y1 = y2sub
        mv2, _ = prob.setupHamiltonian(tnow+dt)
        ye1 = mv2(y1)
        def1 = e2sub-0.5*ye1
        errestmag = dt/(p+1)*nrm(def1)
    
        # test error estimate
        if errestkry+errestmag<dt*tol:
            y0 = y1
            ye0 = ye1
            tnow += dt
            dtlist.append(dt)
            tlist.append(tnow)
            errmlist.append(errestmag)
            dtnew = spstep * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
        else:
            dtnew = spstepfail * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
    return y1, tlist, dtlist, errmlist


def adaptiveCFMp4j2(u,tnow,tend,dtinit,prob,tol=1e-8,m=30,ktype=1,reo=1,nrm=None, inr=None):
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    c = 3**0.5/6
    cmat = [0.5-c, 0.5+c]
    amat=[[0.25+c,0.25-c],[0.25-c,0.25+c]]
    parth1 = [1,0]
    parth2 = [0,1]
    return adaptiveCFM4_symdefHermite(u,tnow,tend,dtinit,prob,cmat,amat,parth1,parth2,tol,m,ktype,reo,nrm,inr)

def adaptiveCFMp4j3(u,tnow,tend,dtinit,prob,tol=1e-8,m=30,ktype=1,reo=1,nrm=None, inr=None):
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    c = 15**0.5
    cmat = [0.5-c/10, 0.5, 0.5+c/10]
    amat = [[0.302146842308616954258187683416,
             -0.030742768872036394116279742324,
             0.004851603407498684079562131338],
            [-0.029220667938337860559972036973,
             0.505929982188517232677003929089,
             -0.029220667938337860559972036973],
            [0.004851603407498684079562131337,
             -0.030742768872036394116279742324,
             0.302146842308616954258187683417]]
    sa=0.5/sum(amat[0])
    parth1 = [sa, 0.5, 1-sa]
    parth2 = [1-sa, 0.5, sa]
    return adaptiveCFM4_symdefHermite(u,tnow,tend,dtinit,prob,cmat,amat,parth1,parth2,tol,m,ktype,reo,nrm,inr)
    
def adaptiveCFM4_symdefHermite(u,tnow,tend,dtinit,prob,cmat,amat,parth1,parth2,tol,m,ktype,reo,nrm,inr):

    chat = -0.5
    jexps=len(amat)
    p=4
    dt = min(dtinit,tend-tnow)

    sig = 1j
    tlist=[]
    dtlist=[]
    errmlist=[]
    mclist1=[]
    mclist2=[]
    mclist3=[]
    spstep=0.8
    spstepfail=0.7
    errpartm=0.99
    tolkry=(1-errpartm)*tol

    y0=u
    mv0, _ = prob.setupHamiltonian(tnow)
    ye0 = mv0(y0)
    ye0 = ye0+0j
    mc1=0
    mc2=0
    mc3=1
    while (tnow<tend):
        y0sub=y0
        e0sub=-0.5*ye0
        for know in range(jexps):
            anow = amat[know]
            mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tnow,dt)
    
            dHpsi = dmv(y0sub)
            Hpsi = mv(y0sub)
            HdHpsi = mv(dHpsi)
            dHHpsi = dmv(Hpsi)
            Gam11 = dt/2*dHpsi - sig*dt**2*1/12*(HdHpsi-dHHpsi)
            if parth1[know]!=0:
                Gam11 += parth1[know]*Hpsi 
            e0sub += Gam11
    
            y1sub, errestkry, tkrylist, mlist1 = expimv_pKry(mv,y0sub,tol=tolkry,
                                                                        t=dt,m=m,ktype=ktype,reo=reo)
            e1sub, errestkry, tkrylist, mlist2 = expimv_pKry(mv,e0sub,tol=tolkry,
                                                                        t=dt,m=m,ktype=ktype,reo=reo)
            
            dHpsi = dmv(y1sub)
            Hpsi = mv(y1sub)
            HdHpsi = mv(dHpsi)
            dHHpsi = dmv(Hpsi)
            Gam12 = dt/2*dHpsi + sig*dt**2*1/12*(HdHpsi-dHHpsi)
            if parth2[know]!=0:
                Gam12 += parth2[know]*Hpsi 
            e1sub += Gam12
            e0sub = e1sub
            y0sub = y1sub
            mc1 += sum(mlist1)
            mc3 += sum(mlist2)+2

        # computer error estimate
        y1 = y1sub
        mv2, _ = prob.setupHamiltonian(tnow+dt)
        ye1 = mv2(y1)
        mc3 += 1
        def1 = e1sub-0.5*ye1
        errestmag = dt/(p+1)*nrm(def1)
        
    
        # test error estimate
        if errestkry+errestmag<dt*tol:
            y0 = y1
            ye0 = ye1
            dtlist.append(dt)
            tlist.append(tnow)
            errmlist.append(errestmag)
            mclist1.append(mc1)
            mclist2.append(mc2)
            mclist3.append(mc3)
            mc1=0
            mc2=0
            mc3=0
            tnow += dt
            dtnew = spstep * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
        else:
            mc2 = mc1+mc3
            mc1=0
            mc3=0
            dtnew = spstepfail * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
    return y1, tlist, dtlist, errmlist, mclist1, mclist2,mclist3
        
