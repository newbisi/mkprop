import numpy as np

def orthogonalize(H, V, k, w, inr, nrm, reo, GStype):
# Hm is atleast k x k matrix
    if (GStype==1):
    # MGS modified Gram-Schmidt
        for j in range(k+1):
            H[j,k] = inr(V[:,j],w)
            w -= H[j,k] * V[:,j]
        countinr = k
        if (reo>0):
            countinr += k
            for j in range(k+1):
                Htemp = inr(V[:,j],w)
                H[j,k] += Htemp
                w -= Htemp * V[:,j]

    elif (GStype==2):
    # three-term recursion
    # Hm is real and symmetric tridiagonal matrix
        if k>0:
            H[k-1,k] = H[k,k-1]
            w -= V[:,k-1]*H[k-1,k] 
        H[k,k] = inr(V[:,k],w).real
        w -= V[:,k]*H[k,k]
        countinr = 1
        
        if (reo>0):
            for j in range(k):
                temp1 = inr(V[:,j],w)
                w -= temp1 * V[:,j]
            temp2 = inr(V[:,k],w)
            H[k,k] += temp2.real
            w -= V[:,k]*temp2
            countinr += k
            
    elif  (GStype==3):
    # CGS classical Gram-Schmidt
        for j in range(k+1):
            H[j,k] = inr(V[:,j],w)
        w -= V[:,:k] * H[:k,k]
        countinr = k

        if (reo>0):
            Htemp=np.zeros(k)
            for j in range(k+1):
                Htemp[j] = inr(V[:,j],w)
            H[:k,k] += Htemp
            w -= V[:,:k] * Htemp
            countinr += k
    return countinr



def arnoldiisometric( mv, v1, m, inr, nrm, reo, Vm=None):
    # no re-orthogonalization procedure available for isometric Arnoldi yet
    if Vm is None:
        n=len(v1)
        Vm=np.zeros([n,m],dtype=np.cdouble)
    Hm = np.eye(m,dtype=np.cdouble)
    gams = np.zeros(m,dtype=np.cdouble)
    sigs = np.zeros(m,dtype=np.cdouble)
    
    beta = nrm(v1)
    Vm[:,0] = v1/beta
        
    vb1=Vm[:,0]
    for k in range(m):
        w = mv(Vm[:,k]);
        gams[k] = -1 * inr(vb1,w)
        vnext = w + gams[k]*vb1;
        # hnext = (1 - abs(gams[k])**2 )**0.5 # this reduces cost but introduces significant round-off errors !
        if (reo>0):
            for j in range(k+1):
                Htemp = inr(Vm[:,j],vnext)
                vnext -= Htemp * Vm[:,j]
        hnext = nrm(vnext);
        #if hnext < eps;
        # stop iteration
        
        if k<m-1:
            Vm[:,k+1] = vnext/hnext
            sigs[k] = hnext
            G = np.array([[-gams[k], sigs[k]],[sigs[k], gams[k].conjugate()]])
            Hm[:,k:k+2] = Hm[:,k:k+2].dot(G)
            vbnext = sigs[k]*vb1 + gams[k].conjugate()*Vm[:,k+1]
            vb1 = vbnext/nrm(vbnext) # Schae08, normalize here to preserve orthogonality for larger m
        else:
            Hm[:,k] *= -gams[k]
            vnext = vnext/hnext

    return Vm, Hm, beta, hnext, vnext


def polKry(mv, v, m, inr=None, nrm=None, reo=1, ktype=1, V=None):
    n = len(v)
    if inr is None:
        inr = lambda x,y : np.vdot(x,y)
    if nrm is None:
        nrm = lambda x : np.sqrt(inr(x,x).real)
    if V is None:
        V = np.zeros([n,m],dtype=np.cdouble)
    if (ktype==1):
        # Arnoldi with modified gram-schmidt
        GStype = 1
        H = np.zeros([m,m],dtype=np.cdouble)
    else:
        # Lanczos
        GStype = 2
        H = np.zeros([m,m],dtype=np.double)
    
    beta = nrm(v)
    V[:,0]=v/beta
    
    zero = 1e-30*beta
    
    for k in range(m):
        w=mv(V[:,k])
        orthogonalize(H, V, k, w, inr, nrm, reo, GStype)
        hnext = nrm(w)
        # check if zero
        if k<m-1:
            H[k+1,k] = hnext
            V[:,k+1] = w/hnext
        else:
            vnext = w/hnext

    return  V, H, beta, hnext, vnext


def computeAmfromHm(shi,m,Hm,hnext,kapp):
    Hinv = np.linalg.inv(Hm[:m,:m])
    emHi = Hinv[m-1,:m]
    Am = (Hinv+Hinv.T.conjugate())/2 + shi.real*np.eye(m) + hnext**2*(kapp-shi.real)*np.outer(emHi.conjugate(),emHi)
    return Am, emHi
def computeAmfromZm(shi,m,Zm,hnext,kapp):
    Hinv = np.linalg.inv(np.eye(m)-Zm[:m,:m]) 
    emHi = Hinv[m-1,:m]
    Hp1 = (shi.conjugate()*np.eye(m) - shi*Zm[:m,:m]).dot(Hinv)
    #Am = Hp1 + hnext**2*(kapp-shi.conjugate())*np.outer(emHi.conjugate(),emHi)
    Am = (Hp1+Hp1.T.conjugate())/2 + hnext**2*(kapp-shi.real)*np.outer(emHi.conjugate(),emHi)
    return Am, emHi

