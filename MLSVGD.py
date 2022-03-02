"""
MLSVGD.py

Author: Terrence Alsup
Date:   March 1st, 2022

Implementation of the MLSVGD algorithm.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
import time
import pickle


class MLSVGD:

    def __init__(self, gradlnprobs, theta0, stepsize, bandwidth, tol, max_iters=1000, save_file=None):
        """
        Instantiate the MLSVGD set up to be run.
        
        args:
        gradlnprobs - dict, dictionary of callables where the key is the level and the value is the callable
                            that takes in a numpy array or shape (d,) and outputs an array of shape (d,)
                            * note that the levels should be in order of increasing fidelity/cost
        theta0      - numpy.ndarray, the numpy array of shape (N, d) of the initial particles
        stepsize    - float, the step size for the gradient descent update
        bandwidth   - float, the bandwidth for the SVGD kernel
        tol         - float, the tolerance on the gradient norm for when to switch levels/terminate
        max_iters   - int, the maximum number of iterations to perform __on each level__
        save_file   - str, a string for the name of the file to save the results to
        """

        # Set variables
        self.gradlnprobs = gradlnprobs
        self.theta = deepcopy(theta0)
        self.stepsize = stepsize
        self.bandwidth = bandwidth
        self.tol = tol
        self.max_iters = max_iters
        
        # Get the dimension and number of particles (samples)
        self.nsamples, self.dim = theta0.shape
        self.level_list = list(gradlnprobs.keys())

        # Create lists to track of everything we want to record at each iteration
        self.grad_norms = []     # average norm of gradient of particles
        self.levels = []         # level at each iteration
        self.runtimes = []       # cumulative measured runtime
        self.total_iters = []    # total iterations from all levels
        self.iters_at_level = [] # iterations since beginning of the level
        self.means = []          # mean of the particles
        
        # Check where to save the results
        if save_file is None:
            self.save_file = 'MLSVGD_results.pickle'
        else:
            self.save_file = save_file
        
        # Dictionary of results to save
        self.results = {'initial_particles':theta0, 'stepsize':stepsize, 
                        'bandwidth':bandwidth, 'tol':tol, 'max_iters':max_iters,
                        'nsamples':self.nsamples, 'dim':self.dim}



    def svgd_kernel(self, theta, h):
        """
        Evaluate the kernel matrix K(theta_i, theta_j) as well as the gradient of 
        the kernel w.r.t. the first argument theta_1.
        
        args:
        theta - numpy.ndarray, numpy array of shape (N, d) of the particles
        h     - float, the bandwidth (std. dev. of kernel)
        
        returns:
        (Kxy, dxkxy) - tuple(numpy.ndarray, numpy.ndarray), a tuple containing the matrix of
                                                            kernel values of shape (N, N)
                                                            Kxy[i,j] = K(theta_i, theta_j)
                                                            and the gradient w.r.t. theta_i
                                                            of shape (N, d)
                                                            dxkxy[i,:] = grad_1 K(theta_i, theta_j)
        """
        
        # Compute all distances between between points
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2

        # Compute the radial basis function kernel matrix
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i], sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
 
    def update_svgd(self, lnprob, L, verbose=-1):
        """
        The main SVGD update at level L.  Updates the particles and records the norm of the 
        gradient, current iteration, and cumulative runtime.
        
        args:
        lnprob  - callable, gradient of the log density at level L that maps numpy.array of
                            shape (d,) to a numpy array of shape (d,)
        L       - int, current level
        verbose - int, determines how frequently to print updates, if verbose > 0 then 
                       results are printed out every 'verbose' iterations (e.g. every 5)
        """

        # Set the current iteration at this level to 0
        i = 0
        gnorm = 2*self.tol + 1 # Ensure at least a single iteration
        while gnorm > self.tol and i < self.max_iters:

            # Compute the gradient of the log densities for each particle
            lnpgrad = np.apply_along_axis(lnprob, 1, self.theta)
            
            # Calculate the kernel matrix and the gradient estimate in the RKHS
            kxy, dxkxy = self.svgd_kernel(self.theta, h=self.bandwidth)  
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / self.nsamples
            
            # Gradient descent update
            self.theta = self.theta + self.stepsize * grad_theta
            
            # Compute expected norm of gradient
            gnorm = np.mean(np.apply_along_axis(np.linalg.norm, 1, grad_theta))
            self.grad_norms.append(gnorm)

            # Compute the mean of the particles
            particle_mean = np.mean(self.theta, axis=0)
            self.means.append(particle_mean)

            # Get total runtime so far
            runtime = time.time() - self.start
            self.runtimes.append(runtime)

            # Record the total iterations and iteration at this level
            self.iters_at_level.append(i)
            self.total_iters.append(self.curr_iter)
            self.curr_iter += 1
            
            # Record the level
            self.levels.append(L)

            # Print update after iteration
            if verbose > 0 and np.mod(i, verbose) == 0:
                print('iter = {:d},\t runtime [s] = {:0.2e},\t grad norm = {:0.3e}'.format(i, runtime, gnorm))
            i += 1

        # Save the results at the end of each level
        self.save_results()
    

    def run(self, verbose=-1):
        """
        Run SVGD at all of the different levels.
        
        args:
        verbose - int, determines how frequently to print updates, if verbose > 0 then 
                       results are printed out every 'verbose' iterations (e.g. every 5)
                       
        returns:
        save_results - dict, a dictionary of results containing gradient norms, iterations, runtime,
                             and other parameters     
        """

        # Get all of the levels
        lvls = self.level_list  # Make sure we start at the lowest fidelity/level

        self.curr_iter = 0        # Keep track of the total number of iterations performed
        self.start = time.time()  # Keep track of start time

        # Loop over the levels in increasing order
        for L in lvls:
            print('\nStarting level ' + str(L) + '\n')
            self.update_svgd(self.gradlnprobs[L], L, verbose)
        self.save_results()

        # Return a dictionary of the results
        print('\nMLSVGD completed\n')
        print('Results saved at\t{:s}\n'.format(self.save_file))
        return self.results


    def save_results(self):
        """
        Store the results in a dictionary and save it as a pickle file.
        """
        # Save all of the current values as well as the parameters
        results = self.results
        results['particles']      = self.theta
        results['grad_norm']      = np.array(self.grad_norms)
        results['levels']         = np.array(self.levels)
        results['total_iters']    = np.array(self.total_iters)
        results['iters_at_level'] = np.array(self.iters_at_level)
        results['runtime']        = np.array(self.runtimes)
        results['particle_mean']  = np.array(self.means)

        with open(self.save_file, 'wb') as fp:
            pickle.dump(results, fp)