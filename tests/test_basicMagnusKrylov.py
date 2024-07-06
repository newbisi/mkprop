import numpy as np
import mkprop
from exlaser import doublewellproblem as prob
import scipy.sparse

def test_basicMagnusKrylov():

    # setup problem from exlaser.py
    n=500
    Hamiltonian = prob(n)
    inr, nrm = Hamiltonian.getnrm()
    u = Hamiltonian.getinitialstate()

    # define initial and final time
    tnow = 0
    tend = 0.05
    tau = tend-tnow
    
    # some options for Krylov method, use Lanczos
    #ktype = 2
    #mmax = 60
    
    tolref=1e-6
    dtinitref = 5e-1
    yref,info = mkprop.adaptiveCFMp4j2(u,tnow,tend,dtinitref,Hamiltonian,tol=tolref)
    
    # test adaptive midpoint rule
    tol=1e-4
    dtinit = 1e-3
    y,info = mkprop.adaptivemidpoint(u,tnow,tend,dtinit,Hamiltonian,tol=tol)

    errorsmallertol = (nrm(yref-y)/tau <= tol)
    if errorsmallertol:
        assert True
    else:
        assert False

def test_basicKrylov():
    # setup problem from exlaser.py
    nrm = lambda x : np.linalg.norm(x,2)
    n = 500
    e = np.ones(n)
    e1 = np.ones(n-1)
    # M is scaled finite difference discretization of 1d Laplace operator
    M = scipy.sparse.diags([e1,-2*e,e1], [-1,0,1])
    u=np.random.rand(n)
    dt = 50
    tol = 1e-6
    yref = scipy.sparse.linalg.expm_multiply(1j*dt*M,u)
    y = mkprop.expimv_pKry(M,u,t=dt,tol=tol)

    errorsmallertol =  (nrm(yref-y)/dt <= tol)
    if errorsmallertol:
        assert True
    else:
        assert False
        
