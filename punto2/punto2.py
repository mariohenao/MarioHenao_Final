import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint

def loglikelihood(t_obs, r_obs_tot, sigma_r_obs, sigma, rho, beta):
    sol = odeint(pend, r0, t_obs, args=(sigma,rho,beta))
    sol_tot = np.array(list(sol[:,0]) + list(sol[:,1]) + list(sol[:,2]))
    d = r_obs_tot - sol_tot
    d = d/sigma_r_obs
    d = -0.5 * np.sum(d**2)
    return d

def divergence_loglikelihood(t_obs, r_obs, sigma_r_obs, sigma, rho, beta):
    n_param = 3
    div = np.ones(n_param)
    delta = 1E-5
    for i in range(n_param):
        delta_parameter = np.zeros(n_param)
        delta_parameter[i] = delta
        div[i] = loglikelihood(t_obs, r_obs_tot, sigma_r_obs, sigma + delta, rho + delta, beta + delta) 
        div[i] = div[i] - loglikelihood(t_obs, r_obs_tot, sigma_r_obs, sigma - delta, rho - delta, beta - delta)
        div[i] = div[i]/(2.0 * delta)
    return div

def hamiltonian(t_obs, ri_obs, sigma_r_obs, param, param_momentum):
    m = 1.0
    K = 0.5 * np.sum(param_momentum**2)/m
    V = -loglikelihood(t_obs, r_obs_tot, sigma_r_obs, param[0], param[1], param[2])     
    return K + V

def leapfrog_proposal(t_obs, r_obs_tot, sigma_r_obs, param, param_momentum):
    N_steps = 5
    delta_t = 1E-2
    m = 1.0
    new_param = param.copy()
    new_param_momentum = param_momentum.copy()
    for i in range(N_steps):
        new_param_momentum = new_param_momentum + divergence_loglikelihood(t_obs, r_obs_tot, sigma_r_obs, param[0], param[1], param[2]) * 0.5 * delta_t
        new_param = new_param + (new_param_momentum/m) * delta_t
        new_param_momentum = new_param_momentum + divergence_loglikelihood(t_obs, r_obs_tot, sigma_r_obs, param[0], param[1], param[2]) * 0.5 * delta_t
    new_param_momentum = -new_param_momentum
    return new_param, new_param_momentum

def monte_carlo(t_obs, r_obs_tot, sigma_r_obs, N=5000):
    param = [np.random.random(3)]
    param_momentum = [np.random.normal(size=3)]
    for i in range(1,N):
        propuesta_param, propuesta_param_momentum = leapfrog_proposal(t_obs, r_obs_tot, sigma_r_obs, param[i-1], param_momentum[i-1])
        energy_new = hamiltonian(t_obs, r_obs_tot, sigma_r_obs, propuesta_param, propuesta_param_momentum)
        energy_old = hamiltonian(t_obs, r_obs_tot, sigma_r_obs, param[i-1], param_momentum[i-1])
   
        r = min(1,np.exp(-(energy_new - energy_old)))
        alpha = np.random.random()
        if(alpha<r):
            param.append(propuesta_param)
        else:
            param.append(param[i-1])
        param_momentum.append(np.random.normal(size=3))    

    param = np.array(param)
    return param

arr = np.loadtxt('datos_observacionales.csv',delimiter=' ')

def pend(r, t, sigma, rho, beta):
    x, y, z = r
    drdt = [sigma*(y - x), x*(rho - z), x*y - beta*z]
    return drdt

t_obs = arr[:,0]
r0 = arr[0,1:]
r_obs = arr[:,1:]
r_obs_tot = np.array(list(r_obs[:,0]) + list(r_obs[:,1]) + list(r_obs[:,2]))
sigma_r_obs = np.ones(len(r_obs_tot))

param_chain = monte_carlo(t_obs, r_obs_tot, sigma_r_obs)
n_param  = len(param_chain[0])
best = []
for i in range(n_param):
    best.append(np.mean(param_chain[:,i]))
    
t_model = np.linspace(t_obs.min(), t_obs.max(), 100)
sol = odeint(pend, r0, t_obs, args=(best[0],best[1],best[2]))
x_model = sol[:,0]
y_model = sol[:,0]
z_model = sol[:,0]

plt.plot(t_obs,r_obs[:,0],label=r'$x_{obs}$')
plt.plot(t_obs,r_obs[:,1],label=r'$y_{obs}$')
plt.plot(t_obs,r_obs[:,2],label=r'$z_{obs}$')
plt.plot(t_obs,x_model,'--',label=r'$x_{best}$')
plt.plot(t_obs,y_model,'--',label=r'$y_{best}$')
plt.plot(t_obs,z_model,'--',label=r'$z_{best}$')
plt.legend()
plt.close()