# mkprop package
This package provides some Python routines for time integration based on Magnus-Krylov and Krylov methods.

## Krylov methods

The solution to the system of ODE's
$$\psi'(t)=\mathrm{i}H\psi(t),\qquad \psi(0)=u,$$
where $H\in\mathbb{C}^{n\times n}$, $\psi(t)\in\mathbb{C}^{n}$ and $u\in\mathbb{C}^{n}$ for a dimension $n$ which is assumed to be large,
is given by the matrix exponential applied to $u$, i.e.,
$$\psi(t) = \exp(\mathrm{i}tH)u.$$
The routine `expimv_pKry` provides an approximation to the action of the matrix exponential,
$$y_K(t,H,u) \approx \exp(\mathrm{i}H)u.$$
The approximation is computed to satisfy the error bound
$$\| y_K(t,H,u) -\exp(\mathrm{i}tH)u\|\leq \varepsilon t.$$
The approximation is using Krylov method (Arnoldi or Lanczos) and requires the matrix $H$ or the action of the matrix $\psi \mapsto H\psi$. 

## Magnus-Krylov methods
The solution to the system of ODE's
$$\psi'(t)=\mathrm{i}H(t)\psi(t),\qquad \psi(0)=u,$$
where $H(t)\in\mathbb{C}^{n\times n}$, $\psi(t)\in\mathbb{C}^{n}$ and $u\in\mathbb{C}^{n}$, can be described by a Magnus expansion
$$\psi(t) = \exp(\mathrm{i}t\Omega(t))u,$$
when $t$ is sufficiently small. This package provides adaptive Magnus-Krylov methods, namely, using the adaptive midpoint rule and CFM integrators with error estimates based on symmetrized defects and works of Auzinger et al.. Again, Magnus-Krylov approximations
$$y_{MK}(t,H,u)\approx  \exp(\mathrm{i}t\Omega(t))u,$$
are computed to satisfy the error bound 
$$\| y_{MK}(t,H,u) -\exp(\mathrm{i}tH)u\|\leq \varepsilon t.$$
## Examples
`examples/comparestepscosts.ipyn`: compare step sizes and computational cost of adaptive Magnus-Krylov methods. Step sizes:

![dt over t](https://github.com/newbisi/mkprop/blob/main/examples/stepsize.png)

Cost:

![cost per dt over t](https://github.com/newbisi/mkprop/blob/main/examples/costperstepsize.png)

`examples/testerrasymorder.ipynb`:
Errors and error estimates over the step size, asymptotic eror.

![errors over dt](https://github.com/newbisi/mkprop/blob/main/examples/asymptoticerror.png)
