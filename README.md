# mkprop package
This package provides some Python routines for time integration based on Magnus-Krylov and Krylov methods.

## Problem description
Let $\psi(t)\in\mathbb{C}^n$ refer to the solution of the system of ODE's
$$\psi'(t)=\mathrm{i}H\psi(t),\qquad \psi(0)=u,$$
where $H\in\mathbb{C}^{n\times n}$ and the initial vector $u\in\mathbb{C}^{n}$ for a dimension $n$. Then $\psi(t)$ is given by the action of the matrix exponential on $u$, i.e.,
$$\psi(t) = \exp(\mathrm{i}tH)u.$$
In a similar manner, we consider non-autonomous system of ODE's.
The solution to the system of ODE's
$$\psi'(t)=\mathrm{i}H(t)\psi(t),\qquad \psi(0)=u,$$
where $H(t)\in\mathbb{C}^{n\times n}$ depends on the time $t$. The solution $\psi(t)$ can be described by a Magnus expansion
$$\psi(t) = \exp(\mathrm{i}t\Omega(t))u,$$
for a sufficiently small $t$. 
 
## Krylov methods
The routine `expimv_pKry` provides an approximation to the action of the matrix exponential,
$$y_K(t,H,u) \approx \exp(\mathrm{i}H)u.$$
The approximation is computed to satisfy the error bound
$$\| y_K(t,H,u) -\exp(\mathrm{i}tH)u\|\leq \varepsilon t,$$
where $\varepsilon>0$ is a given tolerance.
The approximation is using Krylov method (Arnoldi or Lanczos) and requires the matrix $H$ or the action of the matrix $\psi \mapsto H\psi$. 

## Magnus-Krylov methods
This package provides adaptive Magnus-Krylov methods, namely, using the adaptive midpoint rule and CFM integrators with error estimates based on symmetrized defects and works of Auzinger et al.. Again, Magnus-Krylov approximations
$$y_{MK}(t,H,u)\approx  \exp(\mathrm{i}t\Omega(t))u,$$
are computed to satisfy the error bound 
$$\| y_{MK}(t,H,u) -\exp(\mathrm{i}t\Omega(t))u\|\leq \varepsilon t,$$
where $\varepsilon>0$ is a given tolerance.

The following methods are available:
[CFM integrators with table of coefficients](https://github.com/newbisi/mkprop/blob/main/docs/tableofcoef.ipynb)

Currently, only methods of order $p=2$ and $p=4$ are implemented with adaptive step-size control.

## Examples

`exlaser.py`: define Hamiltonian, etc..

`examples/comparestepscosts.ipyn`: compare step sizes and computational cost of adaptive Magnus-Krylov methods. Step sizes:

![dt over t](https://github.com/newbisi/mkprop/blob/main/examples/stepsize.png)

Cost:

![cost per dt over t](https://github.com/newbisi/mkprop/blob/main/examples/costperstepsize.png)

`examples/testerrasymorder.ipynb`:
Errors and error estimates over the step size, asymptotic eror.

![errors over dt](https://github.com/newbisi/mkprop/blob/main/examples/asymptoticerror.png)
