import numpy as np
from scipy.linalg import logm, expm
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.special import factorial


def solveDiffusionEq(z, n, p=4):
    """
    Solve the following differential equation
    
    -(exp(a(x;z)) * u')' = 1
    
    on the domain [0,1] with u(0) = 0 and u(1) = 1 using a
    finite difference discretization and then interpolating
    the solution on the grid points to p equally spaced 
    observation points.
    
    args:
    z - numpy.array, array of shape (d,) for the parameters
    n - int, the number of finite difference grid points
    p - int, the number of observation points
    
    returns:
    tuple(fwd, u) - tuple(numpy.array, numpy.array), tuple of
                    the solution at the observation points of
                    shape (p,) and the array for the solution
                    at the grid points of shape (n,)
    """
    d = len(z)
    
    # Observation points
    x_obs = np.linspace(0, 1, p+2)[1:p+1]

    # Set the boundary conditions and forcing function
    u0 = 0
    un = 1
    f = lambda x: np.ones(x.shape)

    # Set up the finite difference grid and staggered grid
    x = np.linspace(0, 1, n+1)
    h = 1/n           # Mesh width
    xs = x[:n] + h/2  # Staggered grid


    # Evaluate the diffusion coefficient on the staggered grid
    # kappa[0] = \kappa_{1/2},...,kappa[n-1] = \kappa_{n-1/2}
    # kappa[i] = \kappa_{i + 1/2} for i = 0,...,n-1
    kappa = np.exp(np.polyval(z/factorial(np.arange(d-1,-1,-1)), xs))
    # Construct the LHS of the discretized equation
    B = np.zeros((3, n-1))
    B[0] = np.insert(-kappa[1:n-1], n-2, 0)
    B[1] = kappa[:n-1] + kappa[1:]
    B[2] = np.insert(-kappa[1:n-1], 0, 0)
    A = n**2 * spdiags(B, [-1, 0, 1], n-1, n-1, format='csr')
    # RHS
    b = f(x[1:n])
    b[0] += kappa[0]*u0*n**2
    b[n-2] += kappa[n-2]*un*n**2
    # Solve the system
    u = spsolve(A, b)
    # Append the boundary conditions
    u = np.insert(u, [0, n-1], [u0, un])
    # Interpolate to the observation points
    fwd = np.interp(x_obs, x, u)
    return (fwd, u)