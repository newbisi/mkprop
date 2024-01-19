import numpy as np
from scipy.special import jv
import scipy
import scipy.linalg

#######################################################################
#######################################################################
#######################################################################

class barycentricratfct():
    def __init__(self, y, b, alpha = None):
        self.y, self.beta = y, b
        self.alpha = alpha
    def __call__(self, x, usesym = False):
        y, beta, alpha = self.y, self.beta, self.alpha
        if alpha is None:
            if usesym:
                return _evalr_unisym(x, y, beta)
            else:
                return evalr_unitary(x, y, beta)
        else:
            return evalr_std(x, y, alpha, beta)

    def getpoles(self):
        y, wj = self.y, self.beta
        m = len(y)
        if (m<=1):
            return []
        else:
            B = np.eye(m+1)
            B[0,0] = 0
            E = np.block([[0, wj],
                        [np.ones([m,1]), np.diag(y)]])
            lam = scipy.linalg.eigvals(E, B)
            lam = lam[np.isfinite(lam)]
            return lam
    
    def getpartialfractioncoef(self):
        y, beta = self.y, self.beta
        if (len(y)<=1):
            return []
        else:
            sj = self.getpoles()
            a0 = np.conj(sum(beta))/sum(beta)
            C_pol = 1.0 / (sj[:,None] - y[None,:])
            N_pol = C_pol.dot(np.conj(beta))
            Ddiff_pol = (-C_pol**2).dot(beta)
            aj = N_pol / Ddiff_pol
            return a0, aj, sj
    def coef(self):
        return 1j*self.y, self.beta
    
def interpolate_unitarysym(nodes_pos, omega):
    # nodes_pos are all strictly positive nodes, total number of nodes is 2n+1 <- always odd!!
    n = len(nodes_pos)
    
    # of 2n+1 nodes, n+1 are support nodes
    ys_pos = nodes_pos[(n+1)%2::2]
    # of 2n+1 nodes, n are test nodes
    xs_pos = nodes_pos[n%2::2]

    Fmo = 1.0 - np.exp(-1j*omega*xs_pos[:,None])
    Fmo[Fmo == 0.0] = 1j
    Rz = Fmo/np.abs(Fmo)
    (Rr, Ri) = (Rz.real, Rz.imag)

    Fmo = 1.0 - np.exp(-1j*omega*ys_pos[None,:])
    Fmo[Fmo == 0.0] = 1j
    Kz = Fmo/np.abs(Fmo)
    (Kr, Ki) = (Kz.real, Kz.imag)

    C2 = 1./(xs_pos[:,None]-ys_pos)
    C2m = -1./(xs_pos[:,None]+ys_pos)

    B1 = Ri*C2*Kr - Rr*C2*Ki
    B2 = -Ri*C2m*Kr - Rr*C2m*Ki

    if (n%2 == 0):
        B = B1 - B2
        a0 = -Rr[:,0]/xs_pos
        g0 = np.linalg.solve(B,a0)
        b0 = (-1j*Kr[0,:]+Ki[0,:])*g0
        b = np.concatenate((np.conj(b0[::-1]),[-1],b0))
        y = np.concatenate((-ys_pos[::-1],[0],ys_pos))
    else:  
        B = B1 + B2
        [U,S,V]=np.linalg.svd(B,full_matrices=True)
        b0 = (1j*Kr[0,:]-Ki[0,:])*V[-1,:]
        b = np.concatenate((-np.conj(b0[::-1]),b0))
        y = np.concatenate((-ys_pos[::-1],ys_pos))
        
    r = barycentricratfct(1j*y,1j*b)
    return r

def interpolate_unitary(nodes_pos, omega):
    # nodes_pos are all strictly positive nodes, total number of nodes is 2n+1 <- always odd!!
    n = len(nodes_pos)
    
    allnodes = np.concatenate((-nodes_pos[::-1],[0],nodes_pos))
    y = allnodes[::2]
    xs = allnodes[1::2]

    C = 1./(xs[:,None]-y)

    Fmo = 1.0 - np.exp(-1j*omega*xs[:,None]) # 1 - F.conj()
    Fmo[Fmo == 0.0] = 1j
    Rz = Fmo/np.abs(Fmo)
    (Rr, Ri) = (Rz.real, Rz.imag)

    Fmo = 1.0 - np.exp(-1j*omega*y[None,:]) # 1 - f.conj()
    Fmo[Fmo == 0.0] = 1j
    Kz = Fmo/np.abs(Fmo)
    (Kr, Ki) = (Kz.real, Kz.imag)

    A = (Ri*C*Kr-Rr*C*Ki)
    [U,S,V]=np.linalg.svd(A,full_matrices=True)
    gam = V[-1,:]
    b = (1j*Kr[0,0:n+2]-Ki[0,0:n+2])*gam   
    
    r = barycentricratfct(1j*y,1j*b)
    return r

def _evalr_unisym(x, y2, b):
    xv = np.asanyarray(x).ravel()
    
    y = y2[y2>0]
    m = len(y2)
    m2 = len(y)
    if (m%2==0):
        b2 = 1j*b[m2:]
    else:
        b2 = b[m2+1:]
        bzer = b[m2]
    D = -xv[:,None]**2 + y[None,:]**2
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    
    #with np.errstate(divide='ignore', invalid='ignore'):
    if len(node_xi) > 0:       # no zero divisors
        D[node_xi, node_zi] = 1.0
    C = np.divide(1.0, D)
    denom = 2*1j*xv*C.dot(b2.real) - 2*C.dot(y*b2.imag) 
    if (m%2==1):
        denom[xv!=0] += -1j*bzer/xv[xv!=0] # b2[m2] = -1.0 with 1/(1j*xv)
        r = -denom.conj()/denom
        r[xv==0] = 1.0
    else:
        r = denom.conj()/denom
    if len(node_xi) > 0:
        node_xi_pos = node_xi[xv[node_xi]>0]
        node_zi_pos = node_zi[xv[node_xi]>0]
        node_xi_neg = node_xi[xv[node_xi]<0]
        node_zi_neg = node_zi[xv[node_xi]<0]
        if (m%2==0):
            r[node_xi_pos] = -np.conj(b2[node_zi_pos])/b2[node_zi_pos]
            r[node_xi_neg] = -b2[node_zi_neg]/np.conj(b2[node_zi_neg])
        else:
            r[node_xi_pos] = np.conj(b2[node_zi_pos])/b2[node_zi_pos]
            r[node_xi_neg] = b2[node_zi_neg]/np.conj(b2[node_zi_neg])
        
        #r[node_xi_neg] = np.exp(1j*w*xv[node_xi_neg])
        #r[node_xi] = np.exp(1j*w*xv[node_xi])

    if np.isscalar(x):
        return r[0]
    else:
        r.shape = np.shape(x)
        return r

def evalr_unitary(x, yj, wj):
    xv = np.asanyarray(x).ravel()

    D = xv[:,None] - yj[None,:]
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    
    one = 1.0    
    with np.errstate(divide='ignore', invalid='ignore'):
        if len(node_xi) == 0:       # no zero divisors
            C = np.divide(one, D)
            denomx = C.dot(wj)
            r = np.conj(denomx) / denomx
        else:
            # set divisor to 1 to avoid division by zero
            D[node_xi, node_zi] = one
            C = np.divide(one, D)
            denomx = C.dot(wj)
            r = denomx.conj() / denomx
            # fix evaluation at nodes to corresponding fj
            r[node_xi] = -np.conj(wj[node_zi])/wj[node_zi]

    if np.isscalar(x):
        return r[0]
    else:
        r.shape = np.shape(x)
        return r
def evalr_std(x, yj, alpha, beta):
    xv = np.asanyarray(x).ravel()

    D = xv[:,None] - yj[None,:]
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    
    one = 1.0    
    with np.errstate(divide='ignore', invalid='ignore'):
        if len(node_xi) == 0:       # no zero divisors
            C = np.divide(one, D)
            numx = C.dot(alpha)
            denomx = C.dot(beta)
            r = numx / denomx
        else:
            # set divisor to 1 to avoid division by zero
            D[node_xi, node_zi] = one
            C = np.divide(one, D)
            numx = C.dot(alpha)
            denomx = C.dot(beta)
            r = numx / denomx
            r[node_xi] = alpha[node_zi]/beta[node_zi]

    if np.isscalar(x):
        return r[0]
    else:
        r.shape = np.shape(x)
        return r

def briberrest(n, w):
    nfacx = np.sum(np.log(np.arange(n+1,2*n+1)))
    efaclog = -2*nfacx-np.log(2*n+1)
    return 2*np.exp(efaclog+(2*n+1)*np.log(w/2))
def briberrest_getw(n, tol):
    nfacx = np.sum(np.log(np.arange(n+1,2*n+1)))
    efaclog = -2*nfacx-np.log(2*n+1)
    logtolh = np.log(tol/2)
    return 2*np.exp((logtolh-efaclog)/(2*n+1))
def briberrest_getn(w, tol):
    from scipy.special import lambertw
    m=-np.log(tol)/lambertw(-4*np.exp(-1)*np.log(tol/2)/w)
    return int(np.ceil((m.real-1)/2))

    
def brib(w = np.inf, n=8, tol = 1e-8, nodes_pos = None,
             maxiter=100, tolequi=1e-3, npi = -30, syminterp = False, step_factor = 0.1):
    # parts of this algorithm are taken from baryrat https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    
    # y or n, mirrored nodes, flat, sorted
    # w or tol, tol only used to specify w if needed, no stopping criteria
    
    a = -1.0
    a0 = 0.0
    b = 1.0

    ###### set parameters
    if nodes_pos is None:
        nodes_pos = PositiveChebyshevNodes(n)
    else:
        nodes_pos = np.sort(nodes_pos[nodes_pos>0].flatten())
        n = len(nodes_pos)
    
    if (w >= np.pi*(n+1)):
        # return constant one function
        print("for w>=(n+1)pi = {} the best approximartion is trivial")
        allpoints=[[0.0],[0.0]]
        allerr = [2.0]
        return barycentricratfct([0.0],[1.0]), allpoints ,allerr
    ######
    
    f = lambda x : np.exp(1j*w*x)
    
    # initial interpolation nodes
    N = 2*n+1
    if nodes_pos is None:
        nodes_pos = PositiveChebyshevNodes(n)

    errors = []
    approxerrors = []
    stepsize = np.nan
    #ni = 1
    max_step_size=0.1
    
    for ni in range(maxiter):
        # compute the new interpolant r
        if syminterp:
            r = interpolate_unitarysym(nodes_pos, w)
        else:
            r = interpolate_unitary(nodes_pos, w)
    
        # find nodes with largest error
        all_nodes_pos = np.concatenate(([0.0], nodes_pos, [1.0]))
        errfun = lambda x: abs(f(x) - r(1j*x,usesym=syminterp))
        #print(errfun(nodes_pos))
        if npi > 0:
            local_max_x, local_max = local_maxima_sample(errfun, all_nodes_pos, npi)
        else:
            local_max_x, local_max = local_maxima_golden(errfun, all_nodes_pos, num_iter=-npi)
    
        max_err = local_max.max()
        deviation = max_err / local_max.min() - 1
        errors.append((max_err, deviation, stepsize))
        approxerrors.append(max_err)
    
        converged = deviation <= tolequi
        if converged:
            # only if converged
            signed_errors = np.angle(r(1j*local_max_x)/f(local_max_x))
            max_phase_err = max(abs(signed_errors))
            signed_errors /= (-1)**np.arange(len(signed_errors)) * np.sign(signed_errors[0]) * max_phase_err
            equi_err = abs(1.0 - signed_errors).max()
            break
        # move nodes
        # test convergence criteria
    
        # global interval size adjustment
        intv_lengths = np.diff(all_nodes_pos)

        mean_err = np.mean(local_max)
        max_dev = abs(local_max - mean_err).max()
        normalized_dev = (local_max - mean_err) / max_dev
        stepsize = min(max_step_size, step_factor * max_dev / mean_err)
        scaling = (1.0 - stepsize)**normalized_dev

        intv_lengths *= scaling
        # rescale so that they add up to b-a again
        intv_lengths *= 1 / intv_lengths.sum()
        nodes_pos = np.cumsum(intv_lengths)[:-1] + a0
    
    if syminterp:
        r = interpolate_unitarysym(nodes_pos, w)
    else:
        r = interpolate_unitary(nodes_pos, w)
    # also return interpolation nodes and equioscillation points 
    allpoints = [np.concatenate((-local_max_x[::-1], local_max_x)), np.concatenate((-nodes_pos[::-1], [0], nodes_pos))]
    return r, allpoints, errors

def _piecewise_mesh(nodes, n):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    """Build a mesh over an interval with subintervals described by the array
    ``nodes``. Each subinterval has ``n`` points spaced uniformly between the
    two neighboring nodes.  The final mesh has ``(len(nodes) - 1) * n`` points.
    """
    #z = np.concatenate(([z0], nodes, [z1]))
    M = len(nodes)
    return np.concatenate(tuple(
        np.linspace(nodes[i], nodes[i+1], n, endpoint=(i==M-2))
        for i in range(M - 1)))

def local_maxima_bisect(g, nodes, num_iter=10):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    L, R = nodes[1:-2], nodes[2:-1]
    # compute 3 x m array of endpoints and midpoints
    z = np.vstack((L, (L + R) / 2, R))
    values = g(z[1])
    m = z.shape[1]

    for k in range(num_iter):
        # compute quarter points
        q = np.vstack(((z[0] + z[1]) / 2, (z[1] + z[2])/ 2))
        qval = g(q)

        # move triple of points to be centered on the maximum
        for j in range(m):
            maxk = np.argmax([qval[0,j], values[j], qval[1,j]])
            if maxk == 0:
                z[1,j], z[2,j] = q[0,j], z[1,j]
                values[j] = qval[0,j]
            elif maxk == 1:
                z[0,j], z[2,j] = q[0,j], q[1,j]
            else:
                z[0,j], z[1,j] = z[1,j], q[1,j]
                values[j] = qval[1,j]

    # find maximum per column (usually the midpoint)
    #maxidx = values.argmax(axis=0)
    # select abscissae and values at maxima
    #Z, gZ = z[maxidx, np.arange(m)], values[np.arange(m)]
    Z, gZ = np.empty(m+2), np.empty(m+2)
    Z[1:-1] = z[1, :]
    gZ[1:-1] = values
    # treat the boundary intervals specially since usually the maximum is at the boundary
    Z[0], gZ[0] = _boundary_search(g, nodes[0], nodes[1], num_iter=3)
    Z[-1], gZ[-1] = _boundary_search(g, nodes[-2], nodes[-1], num_iter=3)
    return Z, gZ

def local_maxima_golden(g, nodes, num_iter):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    # vectorized version of golden section search
    golden_mean = (3.0 - np.sqrt(5.0)) / 2   # 0.381966...
    L, R = nodes[1:-2], nodes[2:-1]     # skip boundary intervals (treated below)
    # compute 3 x m array of endpoints and midpoints
    z = np.vstack((L, L + (R-L)*golden_mean, R))
    m = z.shape[1]
    all_m = np.arange(m)
    gB = g(z[1])

    for k in range(num_iter):
        # z[1] = midpoints
        mids = (z[0] + z[2]) / 2

        # compute new nodes according to golden section
        farther_idx = (z[1] <= mids).astype(int) * 2 # either 0 or 2
        X = z[1] + golden_mean * (z[farther_idx, all_m] - z[1])
        gX = g(X)

        for j in range(m):
            x = X[j]
            gx = gX[j]

            b = z[1,j]
            if gx > gB[j]:
                if x > b:
                    z[0,j] = z[1,j]
                else:
                    z[2,j] = z[1,j]
                z[1,j] = x
                gB[j] = gx
            else:
                if x < b:
                    z[0,j] = x
                else:
                    z[2,j] = x

    # prepare output arrays
    Z, gZ = np.empty(m+2, dtype=z.dtype), np.empty(m+2, dtype=gB.dtype)
    Z[1:-1] = z[1, :]
    gZ[1:-1] = gB
    # treat the boundary intervals specially since usually the maximum is at the boundary
    # (no bracket available!)
    Z[0], gZ[0] = _boundary_search(g, nodes[0], nodes[1], num_iter=3)
    Z[-1], gZ[-1] = _boundary_search(g, nodes[-2], nodes[-1], num_iter=3)
    return Z, gZ

def _boundary_search(g, a, c, num_iter):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    X = [a, c]
    Xvals = [g(a), g(c)]
    max_side = 0 if (Xvals[0] >= Xvals[1]) else 1
    other_side = 1 - max_side

    for k in range(num_iter):
        xm = (X[0] + X[1]) / 2
        gm = g(xm)
        if gm < Xvals[max_side]:
            # no new maximum found; shrink interval and iterate
            X[other_side] = xm
            Xvals[other_side] = gm
        else:
            # found a bracket for the minimum
            return _golden_search(g, X[0], X[1], num_iter=num_iter-k)
    return X[max_side], Xvals[max_side]

def _golden_search(g, a, c, num_iter=20):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))

    b = (a + c) / 2
    gb = g(b)
    ga, gc = g(a), g(c)
    if not (gb >= ga and gb >= gc):
        # not bracketed - maximum may be at the boundary
        return _boundary_search(g, a, c, num_iter)
    for k in range(num_iter):
        mid = (a + c) / 2
        if b > mid:
            x = b + golden_mean * (a - b)
        else:
            x = b + golden_mean * (c - b)
        gx = g(x)

        if gx > gb:
            # found a larger point, use it as center
            if x > b:
                a = b
            else:
                c = b
            b = x
            gb = gx
        else:
            # point is smaller, use it as boundary
            if x < b:
                a = x
            else:
                c = x
    return b, gb

def local_maxima_sample(g, nodes, N):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    Z = _piecewise_mesh(nodes, N).reshape((-1, N))
    vals = g(Z)
    maxk = vals.argmax(axis=1)
    nn = np.arange(Z.shape[0])
    return Z[nn, maxk], vals[nn, maxk]

#######################################################################
#######################################################################
#######################################################################
def linearizedLawson(w = np.inf, y = None, n = 8, tol = 1e-8,
                              x = None, nx = 2000, nlawson = 20, idl = 1):
    # y or n, mirrored nodes, flat, sorted
    # w or tol, tol only used to specify w if needed, no stopping criteria
    # x or nx, mirrored nodes, flat, sorted, distinct to y
    # idl = 0 classical Lawson
    # idl = 1 unitary Lawson
    # idl = 2 unitary+symmetric Lawson
    # can n be nmax????
    if y is None:
        Cheb_pos = PositiveChebyshevNodes(n)
        Cheb = np.concatenate((-Cheb_pos[::-1],[0],Cheb_pos))
        y = Cheb[::2]
    else:
        y = np.sort(y.flatten())
        if sum(y+y[::-1])>0:
            print("Warning: Lawson support nodes not mirrored around zero")
        n = len(y)-1

    #print("m = {}, n = m-1 = {}".format(len(y),n))
    
    if (w >= np.pi*(n+1)):
        # return constant one function
        print("for w>=(n+1)pi = {} the best approximartion is trivial")
        return (barycentricratfct([0.0],[1.0]) ,[2.0])
        
    if x is None:
        x = np.linspace(-1,1,nx)
    else:
        x = np.sort(x.flatten())
        if sum(x+x[::-1])>0:
            print("Warning: Lawson test nodes not mirrored around zero")
    if len(np.argwhere( (x[:,None]-y) == 0.0 ) > 0):
        print("Warning: some Lawson support and test nodes are identical.")

    return _linearizedLawson(x, y, w, nlawson = nlawson, idl = idl)
        

def _linearizedLawson(x, y, w, nlawson = 20, idl = 1):
    # parts of this algorithm are taken from AAA-Lawson http://www.chebfun.org/
    N = len(x)
    xs = np.concatenate((x,y))
    n = len(xs)
    m = len(y)
    # is m is the number of support nodes to barycentric rational approximation
    # degree of the approximation is m+1
    # todo: make sure entries of y are not in x yet

    # idl = 0 classical Lawson
    # idl = 1 unitary Lawson
    # idl = 2 unitary+symmetric Lawson
        
    # Cauchy matrix
    C = np.zeros([n,m])
    C[:N,:] = 1./(x[:,None]-y)
    C[N:,:] = np.eye(m)

    F = np.exp(1j*w*xs) 
    # Loewner matrix
    if (idl==0):
        A = np.concatenate((C, -F[:,None]*C), axis=1)
        wt = np.ones([n,1])

    elif (idl==1):
        Fmo = 1.0 - np.exp(-1j*w*xs[:,None]) # 1 - F.conj()
        Fmo[Fmo == 0.0] = 1j
        Rz = Fmo/np.abs(Fmo)
        (Rr, Ri) = (Rz.real, Rz.imag)
        A = np.concatenate((Rr*C, -Ri*C), axis=1)
        wt = np.ones([n,1])
        
    elif (idl==2):
        # only positive nodes
        (y2, m2) = (y[y>0], len(y[y>0]))
        (x2, N2) = (x[x>0], len(x[x>0]))
        (xs2, n2) = (xs[xs>0], len(xs[xs>0]))
    
        # Cauchy matrix parts
        C2 = np.zeros([n2,m2])
        C2[:N2,:] = 1./(x2[:,None]-y2)
        C2[N2:,:] = np.eye(m2)
        C2m = np.zeros([n2,m2])
        C2m[:N2,:] = 1./(-x2[:,None]-y2)
        
        Fmo = 1.0 - np.exp(-1j*w*xs2[:,None])
        Fmo[Fmo == 0.0] = 1j
        Rz = Fmo/np.abs(Fmo)
        (Rr, Ri) = (Rz.real, Rz.imag)
    
        if (m%2 == 0):
            B1 = Rr*(C2+C2m)
            B2 = -Ri*(C2-C2m)
        else:
            B1 = Rr*(C2-C2m)
            B2 = -Ri*(C2+C2m)
        A = np.concatenate((B1, B2), axis=1)
        wt = np.ones([n2,1])
    
    errvec = []
    devvec = []
    stepno = 0
    while (stepno < nlawson):
        stepno += 1
        [U,S,V]=np.linalg.svd(np.sqrt(wt)*A, full_matrices=False)

        if (idl==0):
            wa = V[-1,:m].conj()
            wb = V[-1,m:].conj()
            N = np.dot(C,wa)
            D = np.dot(C,wb)
            wjs = [wa,wb]
        elif (idl==1):
            gam = V[-1,:]
            b = (gam[0:m]-1j*gam[m:])/np.sqrt(2.0)
            D = np.dot(C,b)
            N = D.conj()
            wjs = b
        elif (idl==2):
            if (m%2 == 0):
                g0 = V[-1,:]
                mh = int(m/2)
                b = np.concatenate((-g0[mh-1::-1] - 1j*g0[m:mh-1:-1], g0[:mh] - 1j*g0[mh:]))/2
                #b = (g0[:mh] - 1j*g0[mh:])/2
            else:
                a0 = np.sqrt(wt[:,0])*Rr[:,0]/xs2
                a0[N2:] = 0
                a1 = np.dot(U.transpose(),a0)
                a2 = np.dot(V.transpose(),a1/S)
                #axnorm = (2*np.linalg.norm(gf)**2+1)**0.5
                mh = int(m/2)
                gf = np.concatenate((a2[mh-1::-1], [-1], a2[:mh], -a2[m:mh-1:-1], [0], a2[mh:]))
                gf = gf/np.linalg.norm(gf)
                b = (gf[0:m]-1j*gf[m:])/np.sqrt(2.0)
                #gf = (1j*a2[:mh] + a2[mh:])/np.sqrt(2.0)/axnorm
                #b = np.concatenate((-b[::-1].conj())
            # evaluation of r at test nodes can be simplified for the symmetric case
            D = np.dot(C,b)
            N = D.conj()
            wjs = b
        errv = np.abs(N/D - F)
        maxerr = np.max(errv)
        errvec.append(maxerr)
        
        r = barycentricratfct(1j*y,1j*wjs)
        pherr = lambda x: np.angle(r(1j*x)/np.exp(1j*w*x))
        xssorted=np.sort(xs)
        phaseerrorxs = pherr(xssorted)
        ij=np.where(np.sign(phaseerrorxs[1:])-np.sign(phaseerrorxs[:-1]))[0]
        if (len(ij)==2*m-1):
            xsintn = (xssorted[ij]+xssorted[ij+1])/2
            nodes_pos = xsintn[xsintn>0]
            all_nodes_pos = np.concatenate(([0.0], nodes_pos, [1.0]))
            npi = -30
            errfun = lambda x: abs(pherr(x))
            if npi > 0:
                local_max_x, local_max = local_maxima_sample(errfun, all_nodes_pos, npi)
            else:
                local_max_x, local_max = local_maxima_golden(errfun, all_nodes_pos, num_iter=-npi)
            max_err = local_max.max()
            deviation = max_err / local_max.min() - 1
            devvec.append(deviation)
        else:
            devvec.append(0.0)


        #if maxerr<tol:
        #    break

        if (idl==2):
            wt = wt * errv[xs>0,None]
        else:
            wt = wt * errv[:,None]
        wt = wt/max(wt)
        
        if any(wt != wt): # check for nan
            break
    if (idl==0):
        r = barycentricratfct(y,wb,alpha=wa)
    else:
        r = barycentricratfct(1j*y,1j*wjs)
    return [r, errvec, devvec]


##########################################################################
##########################################################################
##########################################################################

class pade():
    # the (k,k) Pade approximation to exp(z)
    def __init__(self, k):
        a = np.ones(k+1)
        for j in range(k):
            fact = (k-j)/(2*k-j)/(j+1)
            a[k-j-1] = -fact * a[k-j] # sum of log
        a = a/a[k]
        self.coef = a
        self.k = k

    def __call__(self, x):
        a, k = self.coef, self.k
        xv = np.asanyarray(x).ravel()
        xp = xv**0
        ys = a[k]*xp
        for j in range(k):
            xp = xp*xv
            ys = ys + a[k-j-1]*xp # exp
        denom = ys
        r = np.conj(denom)/denom
        if np.isscalar(x):
            return r[0]
        else:
            r.shape = np.shape(x)
            return r

    def getpoles(self):
        a = self.coef
        poles = np.roots(a)
        return poles

#######################################################################
#######################################################################
#######################################################################

def riCheb(w, n, syminterp=True):
    # compute (n,n) rational function which interpolates exp(1j*w*x) at 2n+1 Chebyshev nodes
    cheb_nodes_pos = PositiveChebyshevNodes(n)
    if syminterp:
        return interpolate_unitarysym(cheb_nodes_pos, w)
    else:
        return interpolate_unitary(cheb_nodes_pos, w)

def PositiveChebyshevNodes(n):
    # return the strictly positive entries from the 2n+1 Chebyshev nodes, sorted in ascending order
    cheb_nodes_pos = np.cos((2*np.arange(n)+1.)/2/(2*n+1)*np.pi)
    return np.sort(cheb_nodes_pos)

##########################################################################
##########################################################################
##########################################################################

def eval_ratfrompolchebyshev(x, omega, n):
    v = np.ones(np.shape(x))
    op = lambda v : x*v
    u = eval_polynomial_chebyshev(x, omega/2, n)
    return u/np.conjugate(u)

def eval_polynomial_chebyshev(x, t, n):
    # Clenshaw Algorithm
    # polynomial Chebyshev approximation to y ~ exp(1j*t*x) 
    # we use p(A)*v ~ exp(1j*t*A)*v  
    # where op(v) = A*v applies operator on v,
    # the eigenvalues of A are assumed to be located in [-1,1]
    # m .. degree of p
    
    v = np.ones(np.shape(x))
    op = lambda v : x*v

    cm1 = (1j)**n*jv(n,t)
    dkp1 = cm1 * v
    dkp2 = 0
    for k in range(n-1,-1,-1):
        ck = (1j)**k*jv(k,t)
        dk = ck * v + 2*op(dkp1) - dkp2
        if (k>0):
            dkp2 = dkp1
            dkp1 = dk
    return dk - dkp2

##########################################################################
##########################################################################
##########################################################################
