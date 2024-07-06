import scipy.sparse
import scipy.linalg
import numpy as np

def midpoint(u,tnow,tend,dtfixed,prob,ktol=1e-8):
    dt = min(dtfixed,tend-tnow)
    y0=u
    tlist, dtlist = [], []
    while (tnow<tend):
        mv, dmv = prob.setupHamiltonian(tnow+0.5*dt)
        y1, errest, mlist = prob.expimv(mv,dt,y0,ktol)
        y0=y1
        tnow += dt
        dt = min(dtfixed,tend-tnow)
    info = [tlist, dtlist]
    return y1, info


def adaptivemidpoint(u,tnow,tend,dtinit,prob,tol=1e-8):
    #,m=30,ktype=1,reo=1
    inr, nrm = prob.getnrm()
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
    elist = []
    while (tnow<tend):
        anow = [1.0]
        cnow = [0.5]
        chat = 0
        mv, dmv = prob.setupHamiltonianCFM(anow,cnow,chat,tnow,dt)
        y1, errestsub, _ = prob.expimv(mv,dt,y0,tolkry)
        
        # computer error estimate
        dHpsi = dmv(y1)
        HdHpsi = mv(dHpsi)
        Hpsi = mv(y1)
        dHHpsi = dmv(Hpsi)
        defp1 = Hpsi + dt*dHpsi + sig*dt**2/2*(HdHpsi-dHHpsi)
        
        mv2, _ = prob.setupHamiltonian(tnow+dt)
        Hpsi = mv2(y1)
        def1 = defp1-Hpsi
        errestmag = dt/(p+1)*nrm(def1)

        # test error estimate
        if errestsub+errestmag<dt*tol:
            y0 = y1
            tnow += dt
            dtlist.append(dt)
            tlist.append(tnow)
            errmlist.append(errestmag)
            epsi = inr(y1,Hpsi)
            elist.append(epsi)
            if errestmag!=0:
                dtnew = spstep * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            else:
                dt = 2*dt
            dt = min(dtnew,tend-tnow)
        else:
            dtnew = spstepfail * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
    info = [tlist,dtlist,errmlist]
    return y1,info

def adaptivemidpoint_symdef(u,tnow,tend,dtinit,prob,tol=1e-8):
    # m=30,ktype=1,reo=1
    inr, nrm = prob.getnrm()
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
        y1, errestsub, _ = prob.expimv(mv,dt,y0,tolkry)
        y0esym=mv(y0)-y0e
        defp2, _, _ = prob.expimv(mv,dt,y0esym,tolkry)
    
        # computer error estimate
        gam2 = mv(y1)
        mv2, _ = prob.setupHamiltonian(tnow+dt)
        y1enext = mv2(y1)
        def1 = 0.5*(gam2+defp2-y1enext)
        errestmag = dt/(p+1)*nrm(def1)
        
        # test error estimate
        if errestsub+errestmag<dt*tol:
            y0 = y1
            y0e = y1enext
            tnow += dt
            dtlist.append(dt)
            tlist.append(tnow)
            errmlist.append(errestmag)
            if errestmag!=0:
                dtnew = spstep * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            else:
                dt = 2*dt
            dt = min(dtnew,tend-tnow)
        else:
            dtnew = spstepfail * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
    info = [tlist,dtlist,errmlist]
    return y1,info


def CFM4(u,tnow,tend,dtfixed,prob,ktol=1e-8):
    inr, nrm = prob.getnrm()
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
        y1sub, _, _ = prob.expimv(mv,dt,y0,ktol)
    
        know=1
        anow = amat[know]
        mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tnow,dt)
        y1, _, _ = prob.expimv(mv,dt,y1sub,ktol)
    
        y0 = y1
        tnow += dt
        dtlist.append(dt)
        tlist.append(tnow)
    info = [dtlist, tlist]
    return y1, info

def adaptivempnew(u,tnow,tend,dtinit,prob,tol=1e-8, testdt=None):
    cmat = [0.5]
    amat=[[1.0]]
    parth1 = [0.0]
    parth2 = [1.0]
    p=2
    Gamid='stdmidpoint'
    return adaptiveCFM(u,tnow,tend,dtinit,prob,
                       p, Gamid,cmat,amat,parth1,parth2,tol,
                       chat=0,testdt=testdt)
def adaptivempsymnew(u,tnow,tend,dtinit,prob,tol=1e-8, testdt=None):
    cmat = [0.5]
    amat=[[1.0]]
    parth1 = [0.5]
    parth2 = [0.5]
    p=2
    Gamid='symmidpoint'
    return adaptiveCFM(u,tnow,tend,dtinit,prob,
                       p, Gamid,cmat,amat,parth1,parth2,tol,
                       chat=-0.5,testdt=testdt)
    
def adaptiveCFMp4j2(u,tnow,tend,dtinit,prob,tol=1e-8, testdt=None):
    c = 3**0.5/6
    cmat = [0.5-c, 0.5+c]
    amat=[[0.25+c,0.25-c],[0.25-c,0.25+c]]
    parth1 = [1,0]
    parth2 = [0,1]
    p=4
    Gamid='Hermitep4'
    return adaptiveCFM(u,tnow,tend,dtinit,prob,
                       p, Gamid,cmat,amat,parth1,parth2,tol,
                       testdt=testdt)
def adaptiveCFMp4j2_stddef(u,tnow,tend,dtinit,prob,tol=1e-8, testdt=None):
    c = 3**0.5/6
    cmat = [0.5-c, 0.5+c]
    amat=[[0.25+c,0.25-c],[0.25-c,0.25+c]]
    parth1 = [0,0]
    parth2 = [1,1]
    p=4
    Gamid='Hermitep4'
    return adaptiveCFM(u,tnow,tend,dtinit,prob,
                       p, Gamid,cmat,amat,parth1,parth2,tol,
                       chat=0, testdt=testdt)


def adaptiveCFMp4j3(u,tnow,tend,dtinit,prob,tol=1e-8, testdt=None):
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
    amat = [[37.0/240 + 10*c/3/87, -1.0/30, 37.0/240 - 10*c/3/87],
            [-11.0/360, 23.0/45, -11.0/360],
            [37.0/240 - 10*c/3/87, -1.0/30, 37.0/240 + 10*c/3/87]]
    sa=0.5/sum(amat[0])
    parth1 = [sa, 0.5, 1-sa]
    parth2 = [1-sa, 0.5, sa]
    p=4
    Gamid='Hermitep4'
    return adaptiveCFM(u,tnow,tend,dtinit,prob,
                       p, Gamid,cmat,amat,parth1,parth2,tol,
                       testdt=testdt)

def adaptiveCFMBBK4(u,tnow,tend,dtinit,prob,tol=1e-8, testdt=None):
    c = 15**0.5
    cmat = [0.5-c/10, 0.5, 0.5+c/10]
    a1 = [(10+c)/180, -1/9, (10-c)/180]
    a2 = [(15+8*c)/180,1/3,(15-8*c)/180]
    amat = [a1,a2,a2[::-1],a1[::-1]]
    parth1 = [1,1,0,0]
    parth2 = [0,0,1,1]
    p=4
    Gamid='Hermitep4'
    return adaptiveCFM(u,tnow,tend,dtinit,prob,
                       p, Gamid,cmat,amat,parth1,parth2,tol,
                       testdt=testdt)
    
def adaptiveCFM(u,tnow,tend,dtinit,prob,
                p, Gamid, cmat,amat,parth1,parth2,
                tol, chat = -0.5, testdt=None):
    inr, nrm = prob.getnrm()
    jexps=len(amat)

    dt = min(dtinit,tend-tnow)


    sig = 1j
    tlist=[]
    dtlist=[]
    errmlist=[]
    mclist1=[]
    mclist2=[]
    mclist3=[]
    elist = []
    spstep=0.8
    spstepfail=0.7
    errpartm=0.99
    tolkry=(1-errpartm)*tol

    y0=y1=u
    
    if testdt is not None:
        nruns = len(testdt)
        n = len(u)
        Yout = np.zeros([n,nruns],dtype=np.cdouble)
        resetu = True
        dt = testdt[0]
        
    else:
        resetu = False
        
    if (chat != 0):
        mv0, _ = prob.setupHamiltonian(tnow)
        ye0 = mv0(y0)
        ye0 = ye0+0j

    jrun = 0
    mc1=0
    mc2=0
    mc3=1
    while (tnow<tend):
        errestsub = 0.0
        
        y0sub=y0
        if (chat != 0):
            e0sub=chat*ye0
            hase0 = True
        else:
            hase0 = False
            
        for know in range(jexps):
            anow = amat[know]
            mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tnow,dt)

            # defect Gamma part 1
            Gam11 = Gammapart(Gamid, 1, mv, dmv, y0sub, sig, dt, parth1[know])
            if Gam11 is not None:
                if (hase0):
                    e0sub += Gam11
                else:
                    e0sub = Gam11
                    hase0 = True

            if ((hasattr(prob, 'applyexpV')) and (sum(anow)<1e-12)):
                y1sub = prob.applyexpV(sig,dt,y0sub)
                if (hase0):
                    e1sub = prob.applyexpV(sig,dt,e0sub)
                mlist1, mlist2 = [], []
            else:
                y1sub, errestsubk, mlist1 = prob.expimv(mv,dt,y0sub,tolkry)
                if (hase0):
                    e1sub, _, mlist2 = prob.expimv(mv,dt,e0sub,tolkry)
                else:
                    mlist2 = []
                errestsub += errestsubk

            # defect Gamma part 2
            Gam12 = Gammapart(Gamid, 2, mv, dmv, y1sub, sig, dt, parth2[know])
            if Gam12 is not None:
                if (hase0):
                    e1sub += Gam12
                else:
                    e1sub = Gam12
            e0sub = e1sub
                
            y0sub = y1sub
            mc1 += sum(mlist1)
            mc3 += sum(mlist2)+2

        y1 = y1sub
        
        # compute error estimate
        mv2, _ = prob.setupHamiltonian(tnow+dt)
        Hpsi = mv2(y1)
        mc3 += 1
        def1 = e1sub-(1+chat)*Hpsi
        errestmag = dt/(p+1)*nrm(def1)

        # test error estimate
        if ((errestsub+errestmag<=dt*tol) or (resetu)):
            if not (resetu):
                y0 = y1
                ye0 = Hpsi
                dtlist.append(dt)
                tnow += dt
                tlist.append(tnow)
            
            errmlist.append(errestmag)
            mclist1.append(mc1)
            mclist2.append(mc2)
            mclist3.append(mc3)
            mc1=0
            mc2=0
            mc3=0
            energynow = inr(y1,Hpsi)
            elist.append(energynow)
            #normlist.append(nrm(y1))

            if (resetu):
                Yout[:,jrun] = y1
                jrun += 1
                if (jrun==nruns):
                    tnow = tend
                else:
                    dt = testdt[jrun]
                tolkry=max(1e-14,1e-2*(1-errpartm)*errestmag)
                    
            else:
                if errestmag!=0:
                    dtnew = spstep * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
                else:
                    dt = 2*dt
                dt = min(dtnew,tend-tnow)
        else:
            mc2 = mc1+mc3
            mc1=0
            mc3=0
            dtnew = spstepfail * (errpartm * dt * tol/errestmag)**(1.0/p) * dt
            dt = min(dtnew,tend-tnow)
    info = [tlist, dtlist, errmlist, mclist1, mclist2,mclist3, elist]
    if (resetu):
        y1 = Yout
    return y1, info

def Gammapart(Gamid, part, mv, dmv, y0, sig, dt, b):
    if (Gamid=='Hermitep4'):
        if (part==1):
            a=-1
        else:
            a=1
        dHpsi = dmv(y0)
        Hpsi = mv(y0)
        HdHpsi = mv(dHpsi)
        dHHpsi = dmv(Hpsi)
        Gamsub = dt/2*dHpsi + a*sig*dt**2*1/12*(HdHpsi-dHHpsi)
        if b!=0:
            Gamsub += b*Hpsi 
        return Gamsub
    elif (Gamid=='symmidpoint'):
        return 0.5*mv(y0)
    elif (Gamid=='stdmidpoint'):
        if (part==1):
            return None
        else:
            dHpsi = dmv(y0)
            Hpsi = mv(y0)
            HdHpsi = mv(dHpsi)
            dHHpsi = dmv(Hpsi)
            return Hpsi + dt*dHpsi + sig*dt**2/2*(HdHpsi-dHHpsi)

def referenceCFM(u,tnow, dts, prob,cmat,amat, refsteps, tolkry):
    inr, nrm = prob.getnrm()
    jexps=len(amat)
    sig = 1j
    nruns = len(dts)
    n = len(u)
    Yout = np.zeros([n,nruns],dtype=np.cdouble)
    chat = 0.0
    for jrun in range(nruns):
        y0sub = u
        dt = dts[jrun]/refsteps
        tz = tnow
        for jsubstep in range(refsteps):
            for know in range(jexps):
                anow = amat[know]
                mv, dmv = prob.setupHamiltonianCFM(anow,cmat,chat,tz,dt)
                #if ((hasattr(prob, 'applyexpV')) and (sum(anow)<1e-12)):
                #    y1sub = prob.applyexpV(sig,dt,y0sub)
                #else:
                y1sub, errestsubk, mlist1 = prob.expimv(mv,dt,y0sub,tolkry)
                y0sub = y1sub
            tz += dt
        Yout[:,jrun] = y1sub
    return Yout
    