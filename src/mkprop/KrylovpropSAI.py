import scipy.sparse
import scipy.linalg
import scipy
import numpy as np
from .KS import *
from .smallexpimv import *

def expimv_SaIKry(mv,mvinv,shi,u,t=1.0,m=40, sig=1j, mvCayley=None, inr=None, nrm=None, tol=1e-8, ktype=1, reo=1,
                V=None, optinfo=0, costratio=10, shortr = False,
                fixedts = None, testallm = False):
    # 
    # approximate the action of the matrix exponential
    # y \approx exp(itA)*u
    # A,           the matrix A or a procedure for the matrix-vector product A*u 
    #              if callable, then the code uses A(u) for the matrix-vector product A*u, otherwise, A.dot(u)
    # nrm and inr, procedures for the underlying vector norm ||u|| and inner product x^H*y
    # m,           maximal dimension of Krylov subspace
    # tol,         relative tolerance for ||error|| < tol*t
    # ktype,       ktype=1 for Arnoldi and ktype=0 for Lanczos
    # reo,         reo > 0 for reorthogonalization
    # optinfo,     return additional information, optout = 0.. result only as output
    #                                             optout = 1.. info = [success, errest, tlist, mlist]
    #                                             optout = 2.. full info
    # costratio,   approx. cost of mv / cost of inner product to provide
    #              estimate for most cost efficient Krylov dimension
    # mv = lambda x : A.dot(x)
    # mvinv = lambda x : (inv(A-shi*I)).dot(x)
    # optinal:
    # mvCayley = lambda x : (A-shi.conjugate()*I).dot((inv(A-shi*I)).dot(x))
    
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
        if nrm is None:
            nrm = lambda x : np.linalg.norm(x,2)
    else:
        if nrm is None:
            nrm = lambda x : np.sqrt(inr(x,x).real)
            
    if ((mvCayley is None) and (shi.imag!=0) and (shortr)):
        mvCayley = lambda x : mvinv(mv(x)-shi.conjugate()*x)
    
    n = len(u)

    # set tnow, tend, and ts for output
    tnow = 0
    if hasattr(t, '__len__'):
        nruns=len(t)
        ts = t
        tend = t[-1]
    else:
        nruns = 1
        ts = [t]
        tend = t
    toutnext = ts[0]
    
    n = len(u)
    if V is None:
        V = np.zeros([n,m],dtype=np.cdouble)
        
    if (testallm):
        nruns = 1
        Yout = np.zeros([n,m],dtype=np.cdouble)
        fixedts = [tend]    
    else:
        Yout = np.zeros([n,nruns],dtype=np.cdouble)

    if (fixedts is None):
        adaptivet = True
        nsteps = 0
    else:
        adaptivet = False
        nsteps = len(fixedts)

    Amtype=1
        
    # variables for step size control
    logtol = np.log(tol)
    errest=0
    needstep = True
    laststep = False
    tlist=[]
    mlist=[]
    deflist=[]
    ocheck=[]
    errest1list=[]
    errest2list=[]

    # success is always True in the current version
    success = True

    y0 = u

    jout = 0
    jstep = 0
    while (tnow<tend):
        if (adaptivet):
            # adaptively choose dt later
            dt = tend - tnow
        else:
            # fixed dt
            dt = fixedts[jstep]-tnow
            jstep += 1
            
        ####  construct SaI Krylov subspace
        if ((shi.imag!=0) and (shortr)):
            # use isometric arnoldi
            V, Zm, beta, hnext, vnext = arnoldiisometric( mvCayley, u, m, inr=inr, nrm=nrm, reo=reo, Vm=V)
            Avnext = mv(vnext)
            kapp = (inr(vnext,Avnext)).real
            Am, eH = computeAmfromZm(shi,m,Zm,hnext,kapp)
        else:
            if (shortr):
                ktype = 2
            else:
                ktype = 1
            V, Hm, beta, hnext, vnext = polKry( mvinv, u, m, inr=inr, nrm=nrm, reo=reo, ktype=ktype, V=V)
            Avnext = mv(vnext)
            kapp = (inr(vnext,Avnext)).real
            Am, eH = computeAmfromHm(shi,m,Hm,hnext,kapp)
        muse = m
        # construct_krylov_subspace done
        if beta<2*tol:
            pass
        else:
            if (adaptivet):
                if (needstep):
                    muse = m
                    dtsuggest = np.exp((logtol-gaml)/(muse-1))
                    if (tnow+dtsuggest >= tend):
                        dt = tend-tnow
                        laststep = True
                    else:
                        dt = dtsuggest
                        laststep = False
            else:
                # dt is already set
                muse = m
            if (adaptivet):
                if (laststep):
                    tnext = tend
                tnext = tnow + dt
            else:
                tnext = fixedts[jstep-1]

            if (testallm):
                for mj in range(muse):
                    if mj+1<muse:
                        vnext1 = V[:,mj+1]
                        Avnext1 = mv(vnext1)
                        kapp1 = (inr(vnext1,Avnext1)).real
                        if ((shi.imag!=0) and (shortr)):
                            hnext1 = Zm[mj+1,mj].real
                        else:
                            hnext1 = Hm[mj+1,mj].real
                    else:
                        hnext1 = hnext
                        Avnext1 = Avnext
                        kapp1 = kapp
                        hnext1 = hnext
                    if ((shi.imag!=0) and (shortr)):
                        Am, eH = computeAmfromZm(shi,mj+1,Zm,hnext1,kapp1)
                    else:
                        Am, eH = computeAmfromHm(shi,mj+1,Hm,hnext1,kapp1)
                    x1 = smallexpimv(sig,dt,Am,mj+1,Amtype)
                    Yout[:,mj] = beta*V[:,:mj+1].dot(x1)
                    if optinfo>0:
                        errest = errestSaIres(dt,beta,hnext1,eH,x1,Avnext1,shi,
                                              V,mj+1,vnext1,kapp1,Amtype,nrm)
                        errest1list.append(errest)
                        errest = errestSaIrdefin(dt,beta,hnext1,eH,sig,Am,Avnext1,shi,
                                                 V,mj+1,vnext1,kapp1,mj+1,Amtype,nrm)
                        errest2list.append(errest)
            else:
                while (toutnext <= tnext):
                    dt1 = toutnext-tnow
                    x1 = smallexpimv(sig,dt1,Am,muse,Amtype)
                    Yout[:,jout] = beta*V[:,:muse].dot(x1)
                    if optinfo>0:
                        errest = errestSaIres(dt1,beta,hnext,eH,x1,Avnext,shi,
                                              V,muse,vnext,kapp,Amtype,nrm)
                        errest1list.append(errest)
                        errest = errestSaIrdefin(dt1,beta,hnext,eH,sig,Am,Avnext,shi,
                                                 V,muse,vnext,kapp,muse,Amtype,nrm)
                        errest2list.append(errest)
                        #errest2list.append(beta*hnext*npart*defint)
                    jout += 1
                    if jout < nruns:
                        toutnext = ts[jout]
                    else:
                        toutnext = np.inf

                if (tnext < tend):
                    x1 = smallexpimv(sig,dt,Am,muse,Amtype)
                    y0 = beta*V[:,:muse].dot(x1)
                    errest += 0
                    
            if optinfo>1:
                if (testallm):
                    for mj in range(muse):
                        ocheck.append(nrm(V[:,:mj+1].conj().T.dot(V[:,:mj+1])-np.eye(mj+1)))
                else:
                    ocheck.append(np.max(abs(V[:,:muse].conj().T.dot(V[:,:muse])-np.eye(muse))))
           
            tnow = tnext
            tlist.append(dt)

    if optinfo>0:
        info = [success, errest1list, errest2list, tlist, deflist, ocheck]
        return Yout, info
    else:
        return Yout

def errestSaIres(dt,beta,hnext,eH,x1,Avnext,shi,V,muse,vnext,kapp,Amtype,nrm):
    vp2 = Avnext-shi*vnext
    vp3 = hnext*abs(kapp-shi.conjugate())*np.linalg.norm(eH)
    if (nrm(vp2)**2-vp3**2)>0:
        npart = (nrm(vp2)**2-vp3**2)**0.5
    else:
        # avoid error message, this case should not occur anyway
        vp1 = hnext*(kapp-shi.conjugate())*V[:,:muse].dot(eH.conjugate())
        npart = nrm(vp1+vp2)
    spart = dt*beta*hnext*abs(eH.dot(x1))
    return spart*npart
    
def errestSaIrdefin(dt,beta,hnext,eH,sig,Am,Avnext,shi,V,muse,vnext,kapp,m,Amtype,nrm):
    vp2 = Avnext-shi*vnext
    vp3 = hnext*abs(kapp-shi.conjugate())*np.linalg.norm(eH)
    if (nrm(vp2)**2-vp3**2)>0:
        npart = (nrm(vp2)**2-vp3**2)**0.5
    else:
        # avoid error message, this case should not occur anyway
        vp1 = hnext*(kapp-shi.conjugate())*V[:,:muse].dot(eH.conjugate())
        npart = nrm(vp1+vp2)
    evaladef = lambda s : abs(eH.dot(smallexpimv(sig,s,Am,m,Amtype)))
    #defint, definterr = scipy.integrate.quad(evaladef, 0, dt)
    defint, definterr = scipy.integrate.quad(evaladef, 0, dt,limit=30)
    return beta*hnext*npart*defint

