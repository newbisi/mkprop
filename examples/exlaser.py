import numpy as np
import sympy as sym
import scipy.sparse

class doublewellproblem():
    def __init__(self,n,L):
        self.n = n
        self.x = np.linspace(-L,L,n+1) 
        dx = 2*L/(n+1)
        self.inr = lambda x,y : dx*np.vdot(x,y)
        self.nrm = lambda x : np.sqrt(self.inr(x,x).real)
        e = np.ones(n+1)
        e1 = np.ones(n)
        D2 = dx**(-2)*scipy.sparse.diags([e1.conj(),-2*e,e1], [-1,0,1])
        self.H0 = D2
        
        V0 = self.x**4 - 20*self.x**2 

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
        #self.Vt = lambda tau: (V0  + 10*np.sin((np.pi*tau/T)**2)*np.sin(omega*tau)*self.x)
        #self.dVt = lambda tau: 10*np.cos((np.pi*tau/T)**2)*2*(np.pi*tau/T)*np.pi/T*np.sin(omega*tau)*self.x+\
        #                      10*np.sin((np.pi*tau/T)**2)*np.cos(omega*tau)*omega*self.x
    def getprop(self):
        return self.x, self.nrm, self.inr
        
    def getinitialstate(self):
        sigma=0.2
        x0 = -2.5
        u = (sigma*np.pi)**(-0.25)*np.exp(-(self.x-x0)**2/(2*sigma))
        return u
        
    def setupHamiltonian(self,t):
        self.V = self.Vt(t)
        self.dV = self.dVt(t)
        H = self.H0 - scipy.sparse.diags([self.V], [0])
        mv = lambda u : H.dot(u)
        dH = -self.dV
        dmv = lambda u : dH*u
        return mv, dmv
        
    def setupHamiltonianCFM(self,a,c,chat,t,dt):
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
        
    def applyexpV(self,sig,t,u):
        expitV = np.exp(-sig*t*self.V)
        return u*expitV


