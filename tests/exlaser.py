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
        if abs(sum(a))>1e-14: # not zero
            H = sum(a)*self.H0 - scipy.sparse.diags([self.V], [0])
            mv = lambda u : H.dot(u)
        else:
            mv = lambda u : -self.V*u
        dmv = lambda u : -self.dV*u
        return mv, dmv
        
    def applyexpV(self,sig,t,u):
        # return exp(1j*t*V)*u
        # where V corresponds to V(t) or a linear combination sum_j aj*V(t + cj*dt)
        # depending on the last call of setupHamiltonian or setupHamiltonianCFM
        # this routine is only used in BBK18 time integrators
        expitV = np.exp(-sig*t*self.V)
        return u*expitV