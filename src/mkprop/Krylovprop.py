import scipy.sparse
import scipy.linalg
import numpy as np

def expimv_pKry(A,u,t=1.0,m=40,nrm=None,inr=None,tol=1e-8,ktype=1,reo=1,optsout=0, costratio=10):
    # approximate the action of the matrix exponential
    # y \approx exp(itA)*u
    # A,           the matrix A or a procedure for the matrix-vector product A*u 
    #              if callable, then the code uses A(u) for the matrix-vector product A*u, otherwise, A.dot(u)
    # nrm and inr, procedures for the underlying vector norm ||u|| and inner product x^H*y
    # m,           maximal dimension of Krylov subspace
    # tol,         relative tolerance for ||error|| < tol*t
    # ktype,       ktype=1 for Arnoldi and ktype=0 for Lanczos
    # reo,         reo > 0 for reorthogonalization
    # optsout,     return additional information
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

    n = len(u)
    V = np.zeros([n,m],dtype=np.cdouble)
    if ktype > 1:
        H = np.zeros([m,m],dtype=np.double)
    else:
        H = np.zeros([m,m],dtype=np.cdouble)  
    sig=1j
    y0 = u
    logtol = np.log(tol)
    errest=0
    tnow = 0
    needstep = True
    tlist=[]
    mlist=[]
    hnextlist=[]
    ocheck=[]
    bestmlist=[]
    cstlistsub=[]
    cstlist=[]
    errestaclist=[]
    while (tnow<t):
        dt = t - tnow
        beta = nrm(y0)
        if beta<2*tol:
            tnow = t
            break
        V[:,0]=y0/beta
        
        gaml = np.log(beta)
        logdt = np.log(dt)
        countnrm = 0
        tr1 = 0
        tr2 = 0
        for k in range(m):
            w = mv(V[:,k])
            if (ktype>1): # lanczos
                if k>0:
                    H[k-1,k] = H[k,k-1]
                    w = w - V[:,k-1]*H[k-1,k] 
                H[k,k] = inr(V[:,k],w).real
                w = w - V[:,k]*H[k,k]
                countnrm += 1
                if reo>0:
                    for j in range(k):
                        temp1 = inr(V[:,j],w)
                        w = w - temp1 * V[:,j]
                    temp2 = inr(V[:,k],w)
                    H[k,k] += temp2.real
                    w = w - V[:,k]*temp2
                    countnrm += k
                         
            else: # arnoldi with modified gram-schmidt
                for j in range(k+1):
                    H[j,k] = inr(V[:,j],w)
                    w = w - H[j,k] * V[:,j]
                countnrm += k
                if reo>0:
                    for j in range(k+1):
                        temp1 = inr(V[:,j],w)
                        H[j,k] += temp1
                        w = w - temp1 * V[:,j]
                    countnrm += k
            htemp = nrm(w)
            countnrm += 1
            if beta*htemp<tol: # Ja22 condition for lucky breakdown
                muse = k+1
                needstep = False
                errest += dt*beta*htemp
                break
            gaml += np.log(htemp/(k+1))
            if (gaml + k*logdt < logtol):
                muse = k+1
                needstep = False
                errest += np.exp(gaml + muse*logdt)
                break
    
            if (k<m-1):
                V[:,k+1]=w/htemp
                H[k+1,k]=htemp
            if (optsout>0):
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
        if (needstep):
            muse = m
            dt = min(np.exp((logtol-gaml)/(muse-1)), t - tnow)
            errest += dt*tol
        if ktype > 1: # lanczos
            a=np.diagonal(H)[:muse] # diagonal
            b=np.diagonal(H,offset=-1)[:muse-1] # lower secondary diagonal
            lam, Q = scipy.linalg.eigh_tridiagonal(a,b)
            QH=Q.conj().T
            explam_e1 = np.exp(sig*dt*lam)*QH[:,0]
            x1 = Q.dot(explam_e1)
        else:
            expH = scipy.linalg.expm(sig*dt*H[:muse,:muse])
            x1 = expH[:muse,0]
        y0 = beta*V[:,:muse].dot(x1)
        tnow += dt
        tlist.append(dt)
        mlist.append(muse)
        if optsout>0:
            errestaclist.append(errestac)
            hnextlist.append(htemp)
            ocheck.append(np.max(V[:,:muse].conj().T.dot(V[:,:muse])-np.eye(muse)))
            bestmlist.append(np.argmin(cstlistsub)+2)
            cstlistsub = []
            cstlist.append(cstlistsub)

    if optsout>0:
        return y0, errest, tlist, mlist, hnextlist, ocheck, cstlist, bestmlist, errestaclist
    else:
        return y0, errest, tlist, mlist

        
