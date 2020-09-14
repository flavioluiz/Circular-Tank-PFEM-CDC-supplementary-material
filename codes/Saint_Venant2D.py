# 2D Shallow Water Equations for a circular tank
# with feedback control

from fenics import *
import numpy as np

np.set_printoptions(threshold=np.inf)
import mshr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True

from scipy import integrate
from scipy import linalg as la
from math import ceil, floor

n = 10
deg_p = 0
deg_q = 1

g = 10
rho = 1000 # kg/m^3


# Operators and functions

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
R = 1
circle = mshr.Circle(Point(0, 0), R)
mesh = mshr.generate_mesh(circle, n)

#mesh = IntervalMesh(n, 0, L)

d = mesh.geometry().dim()
#plot(mesh)
#plt.show()


r = Expression("sqrt(x[0]*x[0]+x[1]*x[1])", degree=2)
theta = Expression("atan2(x[1],r)", r = r, degree=2)

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Rexternal(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]*x[0]+x[1]*x[1]-R**2) < DOLFIN_EPS and on_boundary


# Boundary conditions on displacement
all_boundary = AllBoundary()
# Boundary conditions on rotations
rexternal = Rexternal()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
rexternal.mark(boundaries, 1)

ds = Measure('ds', subdomain_data= boundaries)
dx = Measure('dx')

# Finite element defition

P_p = VectorElement('DG', mesh.ufl_cell(), deg_p)
P_q = FiniteElement('CG', mesh.ufl_cell(), deg_q)

Vp = FunctionSpace(mesh, P_p)
Vq = FunctionSpace(mesh, P_q)

n_Vp = Vp.dim()
n_Vq = Vq.dim()
n_V = n_Vp + n_Vq

dofVp_x = Vp.tabulate_dof_coordinates().reshape((-1, d))
dofVq_x = Vq.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vp = Vp.dofmap().dofs()
dofs_Vq = Vq.dofmap().dofs()

v_p = TestFunction(Vp)
v_q = TestFunction(Vq)

al_p = TrialFunction(Vp)
al_q = TrialFunction(Vq)

#r = Expression("x[0]", degree = 4)

m_p = inner(v_p, al_p) * dx #inner(v_p, al_p) * dx
m_q = v_q * al_q * dx #inner(v_q, al_q) * dx

j_grad = -dot(v_p, grad(al_q)) * dx # -v_p * r * al_q.dx(0) * dx
j_gradIP = dot(grad(v_q), al_p) * dx # v_q.dx(0) * al_p *r * dx

j_div = -v_q * div(al_p) * dx # -v_q * al_p.dx(0) * dx # 
j_divIP = div(v_p)* al_q * dx # v_p.dx(0) * al_q *dx # 

#j_alldiv_p = j_div
#j_alldiv_q = j_divIP
 
j_allgrad_p = j_gradIP
j_allgrad_q = j_grad

j_p = j_gradIP
j_q = j_grad

# Assemble the interconnection matrix and the mass matrix.
J_p, J_q, M_p, M_q = PETScMatrix(), PETScMatrix(), PETScMatrix(), PETScMatrix()

J_p = assemble(j_p)
J_q = assemble(j_q)

M_p = assemble(m_p).array()
M_q = assemble(m_q).array()

D_p = J_p.array()
D_q = J_q.array()

Pu = FiniteElement('DG', mesh.ufl_cell(), deg_p)
Vu = FunctionSpace(mesh, Pu)
u = TrialFunction(Vu)

B = assemble(v_q * u * ds)
Binput = B.array()
boundary_dof = np.where(Binput.any(axis=0))[0]

Binput = Binput[:,boundary_dof]

n_bd = len(boundary_dof)

# r = Expression("sqrt(r*r+x[1]*x[1])", degree=2)
# theta = Expression("atan2(x[1],r)", degree=2)
#

# Final Assemble

al_p_ = Function(Vp)
al_q_ = Function(Vq)

Hdes = 1
h_eq_ = Function(Vq)
h_eq_.vector()[:] = Hdes
Hd = 0.5*(1./rho * al_q_*dot(al_p_, al_p_) + rho*g*al_q_**2)*dx
Lyap = 0.5*(1./rho * al_q_*dot(al_p_, al_p_) + rho*g*(al_q_ - h_eq_)**2)*dx
#Hd = 0.5*(1./rho*dot(al_p_, al_p_) + rho*g*al_q_**2)*dx

e_p_ = derivative(Hd, al_p_)
e_q_ = derivative(Hd, al_q_)

M = la.block_diag(M_p, M_q)
J = np.zeros((n_V, n_V))
J[:n_Vp, n_Vp:n_V] = D_q
J[n_Vp:n_V, :n_Vp] = D_p


D_q_tilde = la.inv(M_p) @ D_q @ la.inv(M_q)
D_p_tilde = la.inv(M_q) @ D_p @ la.inv(M_p)
invM = la.inv(M)
Jtilde = invM @ J @ invM


def HamFunc(p,q):
    #this method should work for both linear and nonlinear Hamiltonian
    al_p_.vector()[:] =  1.*p
    al_q_.vector()[:] =  1.*q
    return assemble(Hd)

def LyaFunc(p,q):
    #this method should work for both linear and nonlinear Hamiltonian
    al_p_.vector()[:] = 1.*p
    al_q_.vector()[:] = 1.*q
    return assemble(Lyap)

def HamFuncXE(p,q):
    # this method should only work for linear Hamiltonian, right?
    return (q @ e_q_fenics(p,q) + p @ e_p_fenics(p,q))/2

def HamFuncMat(p,q):
    return 0.5*( p @ M_p @ p  /rho + q @ M_q @ q  * rho * g)

def numerical_der(fun, x0):
    n0 = len(x0)
    grad = np.zeros(n0)
    for i in range(n0):
        delt = np.zeros(n0)
        delt[i] = 1e-6
        grad[i] = (fun(x0+delt)-fun(x0-delt))/(2e-6)
    return grad

def e_q_fenics(p,q):
    al_p_.vector()[:] = 1.*p
    al_q_.vector()[:] = 1.*q
    return assemble(e_q_).get_local()

def e_p_fenics(p,q):
    al_p_.vector()[:] = 1.*p
    al_q_.vector()[:] = 1.*q
    return assemble(e_p_).get_local()

def e_q_numeric(p,q):
    return numerical_der(lambda x: HamFunc(p,x), q) 

def e_p_numeric(p,q):
    return numerical_der(lambda x: HamFunc(x,q), p)

def e_q_mat(p,q):
    return M_q @ q *rho * g 

def e_p_mat(p,q):
    return M_p @ p /rho


# reference for the control:
init_p = Constant(('0','0'))
init_q = Constant(Hdes)
al_p_.assign(interpolate(init_p, Vp))
al_q_.assign(interpolate(init_q, Vq))
e0_p = assemble(e_p_).get_local()
e0_q = assemble(e_q_).get_local()
e0 = np.concatenate((e0_p, e0_q), axis = 0)


# preparacao da dinâmica utilizando integrador IVP
B = np.concatenate((np.zeros((n_Vp, n_bd)), Binput), axis = 0)
#Btilde = (invM @ B).reshape(-1,)
z = 0.001
Rr = z * invM @ (B @ B.T) @ invM
def fun(t,y):
    al_p = y[:n_Vp]
    al_q = y[n_Vp:n_V]

    e_p = e_p_fenics(al_p, al_q)
    e_q = e_q_fenics(al_p, al_q)

    e = np.concatenate((e_p, e_q), axis = 0)
    dydt = Jtilde @ e - Rr @ (e-e0) * (t > 0.5)
    
    return dydt


# preparação da dinâmica usando integrador Symplectic Euler (arquivo SympEuler.py)
Btilde = la.inv(M_q) @ Binput
RR = (Btilde @ Btilde.T)
def funSE(t,p,q):
    e_p = e_p_mat(p, q)
    e_q = e_q_mat(p, q)
    dpdt = D_q_tilde @ e_q
    dqdt = D_p_tilde @ e_p - RR @ (e_q-e0_q) * 0.1 * (t > 0.5)
    Aqinv = np.eye(len(q))
    return dpdt, dqdt, Aqinv



# initial conditions for the simulation:
h = 0.1
init_p = Expression(('0','0'), degree=0)
init_q = Expression('H + h*sin(pi/R*r)*cos(2*theta)', degree=4, H=Hdes, h=h, R=R, r=r, theta=theta)
al_p_.assign(interpolate(init_p, Vp))
al_q_.assign(interpolate(init_q, Vq))
alp_0 = al_p_.vector().get_local()
alq_0 = al_q_.vector().get_local()

t0 = 0.0; t_fin = 3; n_t = 300


# simulation using Symplectic Euler;
#Rr = la.inv(M_q)@(B_r.reshape(-1,1)) @ B_r.reshape(1,-1) @ la.inv(M_q) * M_q
#R =  Rr * rho * g

#Aq = (np.eye(n_Vq) +  R *dt *0.01)
#invAq = la.inv(Aq)
#Btilde_r = (la.inv(M_q) @ B_r).reshape(-1,1)

# from SympEuler import SympEuler
#
# dt = 1e-3
# t_evSE, p_SE, q_SE = SympEuler(funSE, alp_0, alq_0, t_fin, t0, dt, n_ev = n_t)
# y_SE = np.concatenate((p_SE, q_SE), axis=0)
# al_sol = y_SE
# t_ev = t_evSE
# n_ev = len(t_ev)


# simulation using ivp:

y0 = np.concatenate((alp_0, alq_0), axis = 0)
t_span = [t0, t_fin]
t_ev = np.linspace(t0,t_fin, num = n_t)
sol = integrate.solve_ivp(fun, t_span, y0, method='RK45', vectorized=False, t_eval = t_ev, atol = 1e-6, rtol = 1e-6)
al_sol = sol.y
t_ev = sol.t
n_ev = len(t_ev)

alp_sol = al_sol[: n_Vp, :]
alq_sol = al_sol[n_Vp: n_V, :]

# plot Hamiltonian 

H_vec_int = np.zeros((n_ev, ))
H_vec = np.zeros((n_ev, ))
V_vec = np.zeros((n_ev, ))

for i in range(n_ev):
    H_vec[i] = HamFunc(alp_sol[:,i], alq_sol[:,i]) # computed with fenics
    # H_vec_lin[i] = HamFuncMat(alp_sol[:,i], alq_sol[:,i])
    V_vec[i] = LyaFunc(alp_sol[:,i], alq_sol[:,i])

fntsize = 16
path_out = "../results/2D/"

plt.figure()
plt.plot(t_ev, H_vec, 'r-')
plt.xlabel(r'{Time} (s)',fontsize=fntsize)
plt.ylabel('Total Energy (J)' ,fontsize=fntsize)

plt.savefig(path_out + "Hamiltonian.png", format="png")

plt.figure()
plt.plot(t_ev, V_vec, 'g-', label = 'Lyapunov Function (J)')
plt.xlabel(r'{Time} (s)',fontsize=fntsize)
plt.ylabel('Lyapunov Function (J)' ,fontsize=fntsize)

plt.savefig(path_out + "Lyapunov.png", format="png")


# make an animation
import matplotlib.animation as animation 
from AnimateSurf import animate2D

x_plot = dofVq_x[:,0]
y_plot = dofVq_x[:,1]

anim = animate2D(x_plot, y_plot, alq_sol, t_ev, xlabel = '$x [m]$', ylabel = '$y [m]$', zlabel = '$h [m]$', title = 'Fluid height')

#plt.show()

rallenty = 0.2
Writer = animation.writers['ffmpeg']
writer = Writer(fps=n_ev/t_fin*rallenty, metadata=dict(artist='Me'), bitrate=1800)
anim.save(path_out + 'wave.mp4', writer=writer)

minZ = alq_sol.min()
maxZ = alq_sol.max()


save_figs = True
if save_figs:
    n_fig = 7
    tol = 1e-6
    for i in range(n_fig+1):
        index = int((n_ev-1)/n_fig*i)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False, 'vmin': minZ, 'vmax': maxZ}

        ax.set_xbound(min(x_plot) - tol, max(x_plot) + tol)
        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ybound(min(y_plot) - tol, max(y_plot) + tol)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_zlabel('$h [m]$', fontsize=fntsize)
        ax.set_title('Fluid Height', fontsize=fntsize)

        ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%3.2f' ))

        ax.plot_trisurf(x_plot, y_plot, alq_sol[:,index], **surf_opts)
        plt.savefig(path_out + "Snap_n" + str(index + 1) + ".png", format="png")
        # plt.show()
