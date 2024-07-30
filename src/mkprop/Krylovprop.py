import scipy.sparse
import scipy.linalg
import scipy
import numpy as np
from .KS import *
from .smallexpimv import *

def expimv_pKry(A,u,t=1.0,m=40, sig=1j ,inr=None, nrm=None, tol=1e-8, ktype=1, reo=1,
                V=None, optinfo=0, costratio=10,
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
    
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
        if nrm is None:
            nrm = lambda x : np.linalg.norm(x,2)
    else:
        if nrm is None:
            nrm = lambda x : np.sqrt(inr(x,x).real)
    if callable(A):
        mv = A
    else:
        if hasattr(A, 'dot'):
            mv = lambda x : A.dot(x)
        else:
            raise TypeError("callable function or matrix with .dot() is expected for mv input.")

    # set tnow, tend, and ts for output
    tnow = 0
    if hasattr(t, '__len__'):
        returnmultipley = True
        nruns=len(t)
        ts = t
        tend = t[-1]
    else:
        returnmultipley = False
        nruns = 1
        ts = [t]
        tend = t
    toutnext = ts[0]
    
    n = len(u)
    if V is None:
        V = np.zeros([n,m],dtype=np.cdouble)
    if ktype > 1:
        # Lanczos
        H = np.zeros([m,m],dtype=np.double)
        GStype = 2
    else:
        # Arnoldi
        H = np.zeros([m,m],dtype=np.cdouble)
        GStype = 1

    if (testallm):
        nruns = 1
        Yout = np.zeros([n,m],dtype=np.cdouble)
        returnmultipley = True
        fixedts = [tend]    
    else:
        Yout = np.zeros([n,nruns],dtype=np.cdouble)

    if (fixedts is None):
        adaptivet = True
        nsteps = 0
    else:
        adaptivet = False
        nsteps = len(fixedts)
        
    # variables for step size control
    logtol = np.log(tol)
    errest=0
    needstep = True
    laststep = False
    tlist=[]
    mlist=[]
    hnextlist=[]
    ocheck=[]
    bestmlist=[]
    cstlistsub=[]
    cstlist=[]
    errestaclist=[]
    errestlist = []
    errestreslist =[]
    errestres2list =[]
    nrmlist=[]
    elist=[]

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
        
        beta = nrm(y0)
        if beta<2*tol:
            tnow = tend
            # return y0
            break
        V[:,0]=y0/beta
        
        gaml = np.log(beta)
        logdt = np.log(dt)
        countnrm = 0
        tr1 = 0
        tr2 = 0
        for k in range(m):
            w = mv(V[:,k])
            cinrgs = orthogonalize(H, V, k, w, inr, nrm, reo, GStype)
            countnrm += cinrgs
            htemp = nrm(w)
            countnrm += 1
            if ((adaptivet) and (beta*htemp<tol)): # Ja22 condition for lucky breakdown
                muse = k+1
                needstep = False
                errest += dt*beta*htemp
                break
            gaml += np.log(htemp/(k+1))
            if ((adaptivet) and (gaml + k*logdt < logtol)):
                muse = k+1
                needstep = False
                errest += np.exp(gaml + muse*logdt)
                break
    
            if (k<m-1):
                V[:,k+1]=w/htemp
                H[k+1,k]=htemp
            if (optinfo>1):
                tr1+=H[k,k]
                tr2+=H[k,k]**2
                errestacc=0
                if k>0:
                    dtest = np.exp((logtol-gaml)/k)
                    costest = costratio*(k+1)+countnrm
                    cstlistsub.append(costest/dtest)
                    tr2+=2*H[k-1,k]*H[k,k-1]
                    rho1 = (sig*tr1).real/(k+1)
                    rho2 = (sig**2*(tr1**2+tr2)).real/((k+1)*(k+2))+ \
                            (((sig*tr1).imag)**2-((sig*tr1).real)**2)/((k+1)**2)
                    errestac = rho1*(k+1)/(k+2)*dtest+(rho1**2+rho2)*(k+1)/(k+3)*dtest**2

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
                gamltemp = np.log(beta)
                for mj in range(muse):
                    x1 = smallexpimv(sig,dt,H,mj+1,ktype)
                    Yout[:,mj] = beta*V[:,:mj+1].dot(x1)
                    if optinfo>0:
                        if mj+1<muse:
                            hnext1 = H[mj+1,mj].real
                            gamltemp += np.log(H[mj+1,mj].real/(mj+1))
                        else:
                            hnext1 = htemp
                            gamltemp=gaml
                        errestaysm = np.exp(gamltemp + (mj+1)*np.log(dt))
                        errestlist.append(errest + errestaysm)
                    if optinfo>1:
                        errestreslist.append(beta*hnext1*dt*abs(x1[-1]))
                        errestres2list.append(beta*hnext1*dt/(mj+1)*abs(x1[-1]))

            else:
                while (toutnext <= tnext):
                    dt1 = toutnext-tnow
                    x1 = smallexpimv(sig,dt1,H,muse,ktype)
                    Yout[:,jout] = beta*V[:,:muse].dot(x1)
                    if optinfo>0:
                        errestaysm = np.exp(gaml + muse*np.log(dt1))
                        errestlist.append(errest + errestaysm)
                    if optinfo>1:
                        errestreslist.append(beta*htemp*dt1*abs(x1[-1]))
                        errestres2list.append(beta*htemp*dt1/(muse)*abs(x1[-1]))
                    jout += 1
                    if jout < nruns:
                        toutnext = ts[jout]
                    else:
                        toutnext = np.inf

                if (tnext < tend):
                    x1 = smallexpimv(sig,dt,H,muse,ktype)
                    y0 = beta*V[:,:muse].dot(x1)
                    errest += np.exp(gaml + muse*np.log(dt))
                    
            if optinfo>1:
                errestaclist.append(errestac)
                hnextlist.append(htemp)
                if (testallm):
                    for mj in range(muse):
                        ocheck.append(nrm(V[:,:mj+1].conj().T.dot(V[:,:mj+1])-np.eye(mj+1)))
                else:
                    ocheck.append(np.max(abs(V[:,:muse].conj().T.dot(V[:,:muse])-np.eye(muse))))
                bestmlist.append(np.argmin(cstlistsub)+2)
                cstlist.append(cstlistsub)


            
            tnow = tnext

            tlist.append(dt)
            mlist.append(muse)

    if not returnmultipley:
        Yout = np.reshape(Yout,n)

    if optinfo>1:
        info = [success, errestlist, tlist, mlist, hnextlist, ocheck, cstlist, bestmlist,
                errestaclist, errestreslist, errestres2list]
        return Yout, info
    elif optinfo==1:
        info = [success, errestlist, tlist, mlist]
        return Yout, info
    else:
        return Yout


