# mkprop package
by Tobias Jawecki

This package provides some Python routines for time integration based on Magnus-Krylov and Krylov methods.

**The package is still under development. Future versions might have different API. The current version is not fully documented.**

## Getting started
run `python -m pip install .` inside the mkprop folder to install the package.

## Mathematical problem description

Let $\psi(t)\in\mathbb{C}^n$ refer to the solution of the system of ODE's
$$\psi'(t)=\mathrm{i}H\psi(t),\qquad \psi(t_0)=u\in\mathbb{C}^{n},$$
where $H\in\mathbb{C}^{n\times n}$ and $u$ denotes an initial vector for a dimension $n$. Then $\psi(t)$ is given by the action of the matrix exponential on $u$, i.e.,
$$\psi(t_0+t) = \exp(\mathrm{i}tH)u.$$
In a similar manner, we consider non-autonomous system of ODE's.
The solution to the system of ODE's
$$\psi'(t)=\mathrm{i}H(t)\psi(t),\qquad \psi(t_0)\in\mathbb{C}^{n},$$
where $H(t)\in\mathbb{C}^{n\times n}$ depends on the time $t$. The solution $\psi(t)$ can be described by a Magnus expansion
$$\psi(t_0+t) = \exp(\mathrm{i}t\Omega(t_0,t))\psi(t_0),$$
where the Magnus expansion
$$\Omega(t_0,t)=\sum_{j=1}^\infty\Omega_j(t_0,t)$$
exists for a sufficiently small time-step $t$.

## Krylov methods
The routine `expimv_pKry` provides an approximation to the action of the matrix exponential,
$$y_K(t,H,u) \approx \exp(\mathrm{i}H)u.$$
The approximation is computed to satisfy the error bound
$$\lVert y_K(t,H,u) -\exp(\mathrm{i}tH)u\rVert\leq \varepsilon t,$$
where $\varepsilon>0$ is a given tolerance.
The approximation is using Krylov method (Arnoldi or Lanczos) and requires the matrix $H$ or the action of the matrix $\psi \mapsto H\psi$.

useage: `examples/basicKrylov.ipynb`

```python
import scipy.sparse
import scipy.linalg
import numpy as np
import mkprop
nrm = lambda x : np.linalg.norm(x,2)

n = 1000
e = np.ones(n)
e1 = np.ones(n-1)+1e-5*np.random.rand(n-1)
# M is finite difference discretization of 1d Laplace operator
M = scipy.sparse.diags([e1.conj(),-2*e,e1], [-1,0,1])
u=np.random.rand(n)
u=u/nrm(u)
dt = 50
tol = 1e-6
yref = scipy.sparse.linalg.expm_multiply(1j*dt*M,u)
y,_,_,mused = mkprop.expimv_pKry(M,u,t=dt,tol=tol)
print("approximation error = %.2e, tolerance = %.2e" % (nrm(yref-y)/dt, tol))
# output: approximation error = 5.06e-08, tolerance = 1.00e-06
```
## Magnus integrators
The solution of the non-autonomous system of ODE's is given by a Magnus expansion $\psi(t) = \exp(\mathrm{i}t\Omega(t))u$ which exists for sufficiently small time steps. The matrix $\Omega(t)$ corresponds to an infinite sum of integrals over terms depending on $H$ and commutators of $H$ evaluated at different times. This series can be truncated to construct approximations to the Magnus expansion, and integrals can be approximated by quadrature rules. E.g., the midpoint rule
$$\exp(\mathrm{i}t\Omega(t,\tau))u\approx\exp(\mathrm{i}tH(\tau+t/2))u,$$
for a sufficiently small time-step $t$.
A very general approach consists of avoiding commutator terms in higher order approximations. E.g., the fourth order method
$$\exp(\mathrm{i}t\Omega(t,\tau))u\approx\exp(\mathrm{i}tB_2(t,\tau))\exp(\mathrm{i}tB_1(t,\tau))u,$$
where $B_1$ and $B_2$ correspond to linear combinitions of $H$ evaluated at different times. In general, higher order methods allow taking larger time steps $t$.

The following methods are available:
[CFM integrators with table of coefficients](https://github.com/newbisi/mkprop/blob/main/docs/tableofcoef.ipynb)

Currently, only methods of order $p=2$ and $p=4$ are implemented with adaptive step-size control.

## Defining a problem with a time-dependent Hamiltonian H(t) 
Define a simple problem, see also `examples/exlaser.py`.
```python
import numpy as np
import sympy as sym
import scipy.sparse

class doublewellproblem():
    def __init__(self,n):
        self.n = n
        L = 5
        self.x = np.linspace(-L,L,n+1)

        # problem related inner product
        dx = 2*L/(n+1)
        self.inr = lambda x,y : dx*np.vdot(x,y)
        
        # H_0
        e = np.ones(n+1)
        e1 = np.ones(n)
        D2 = dx**(-2)*scipy.sparse.diags([e1.conj(),-2*e,e1], [-1,0,1])
        self.H0 = D2

        # a double well potential V_0
        V0 = self.x**4 - 20*self.x**2 

        # add a time-dependent potential V(t)
        # prepare V(t) and dV(t)
        T = 5.0
        omega = 10
        x = sym.Symbol('x')
        t = sym.Symbol('t')
        exprV = 10*sym.sin((np.pi*t/T)**2)*sym.sin(omega*t)*x
        exprdV = sym.diff(exprV, t)

        evalVt = sym.lambdify([x,t], exprV, "numpy")
        evaldVt = sym.lambdify([x,t], exprdV, "numpy")
        self.Vt = lambda tau: (V0  + evalVt(self.x,tau))
        self.dVt = lambda tau: evaldVt(self.x,tau)
        
    def getprop(self):
        # return nodes x and inner product
        return self.x, self.inr
        
    def getinitialstate(self):
        # return initial state
        s, x0 = 0.2, -2.5
        u = (s*np.pi)**(-0.25)*np.exp(-(self.x-x0)**2/(2*s))
        return u
        
    def setupHamiltonian(self,t):
        # return a routine to apply H(t) = H0 - (V0 + V(t))
        # and dH(t) = - dV(t)
        self.V = self.Vt(t)
        self.dV = self.dVt(t)
        H = self.H0 - scipy.sparse.diags([self.V], [0])
        mv = lambda u : H.dot(u)
        dH = -self.dV
        dmv = lambda u : dH*u
        return mv, dmv
        
    def setupHamiltonianCFM(self,a,c,chat,t,dt):
        # return a routine to apply sum_j aj*H(t + cj*dt)
        # and dH(t) = sum_j (c_j+chat)*aj*dH(t + cj*dt)
        # for some scalar chat
        jexps = len(a)
        V_CFM = a[0]*self.Vt(t+c[0]*dt)
        dV_CFM = (c[0]+chat)*a[0]*self.dVt(t+c[0]*dt)
        for j in range(jexps-1):
            V_CFM += a[j+1]*self.Vt(t+c[j+1]*dt)
            dV_CFM += (c[j+1]+chat)*a[j+1]*self.dVt(t+c[j+1]*dt)
        self.V = V_CFM
        self.dV = dV_CFM
        if sum(a)!=0:
            H = sum(a)*self.H0 - scipy.sparse.diags([self.V], [0])
            mv = lambda u : H.dot(u)
        else:
            mv = lambda u : -self.V*u
        dmv = lambda u : -self.dV*u
        return mv, dmv
```

## Magnus-Krylov methods
This package provides adaptive Magnus-Krylov methods, namely, using the adaptive midpoint rule and CFM integrators with error estimates based on symmetrized defects and works of Auzinger et al.. Again, Magnus-Krylov approximations
$$y_{MK}(t,H,u)\approx  \exp(\mathrm{i}t\Omega(t))u,$$
are computed to satisfy the error bound 
$$\lVert y_{MK}(t,H,u) -\exp(\mathrm{i}t\Omega(t))u\rVert\leq \varepsilon t,$$
where $\varepsilon>0$ is a given tolerance.

useage: `examples/basicMagnusKrylov.ipynb`
```python
import numpy as np
import mkprop
from exlaser import doublewellproblem as prob
import matplotlib.pyplot as plt

# setup problem from exlaser.py
n=1200
Hamiltonian = prob(n)
x, inr = Hamiltonian.getprop()
nrm = lambda u : ((inr(u,u)).real)**0.5
u = Hamiltonian.getinitialstate()

# define initial and final time
tnow = 0
tend = 0.1
tau = tend-tnow

# some options for Krylov method, use Lanczos
ktype = 2
mmax = 60

# compute reference solution with adaptive fourth order CFM integrator
tolref=1e-6
dtinitref = 5e-2
yref,_,_,_,_,_,_ = mkprop.adaptiveCFMp4j2(u,tnow,tend,dtinitref,Hamiltonian,tol=tolref,
                                          m=mmax,ktype=ktype,inr=inr)

# test adaptive midpoint rule
tol=1e-4
dtinit = 1e-3
y,_,_,_,_,_,_ = mkprop.adaptivemidpoint(u,tnow,tend,dtinit,Hamiltonian,tol=tol,
                                        m=mmax,ktype=ktype,inr=inr)

print("approximation error = %.2e, tolerance = %.2e" % (nrm(yref-y)/tau, tol))
# output: approximation error = 6.40e-05, tolerance = 1.00e-04
```
## Examples

`examples/exlaser.py`: define Hamiltonian, etc..

`examples/comparestepscosts.ipyn`: compare step sizes and computational cost of adaptive Magnus-Krylov methods. Step sizes:

![dt over t](https://github.com/newbisi/mkprop/blob/main/examples/stepsize.png)

Cost:

![cost per dt over t](https://github.com/newbisi/mkprop/blob/main/examples/costperstepsize.png)

`examples/testerrasymorder.ipynb`:
Errors and error estimates over the step size, asymptotic order.

![errors over dt](https://github.com/newbisi/mkprop/blob/main/examples/asymptoticerror.png)
