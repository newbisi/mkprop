import scipy.linalg
import scipy
import numpy as np

def smallexpimv(sig,dt,H,muse,ktype):
    if muse > 1:
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
    elif muse==1:
        x1 = np.array([np.exp(sig*dt*H[0,0])])
    else:
        x1 = None
    return x1

# add smallexpimv routine for qudrature of em*Hinv*exp(t*Am)*e1
# which can be evaluated for a vector t, this could be done by using the eigenvalue decomposition of Am
# 
# em.dot(Hinv.dot(Q)) * np.exp(T*Lam) .dot(QH[:,0])
