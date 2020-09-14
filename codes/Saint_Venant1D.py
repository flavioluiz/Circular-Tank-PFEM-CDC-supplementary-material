# Saint-Venant equations for a circular tank
# reduced to 1D by radial symmetry

from fenics import *
import numpy as np
from math import pi

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

n = 20
deg = 1
g = 10
rho = 1000
# kg/m^3

# Operators and functions

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
# R = 1
# circle = mshr.Circle(Point(0, 0), R)
# mesh = mshr.generate_mesh(circle, n)

L = 1
mesh = IntervalMesh(n, .0, L)

d = mesh.geometry().dim()


# plot(mesh)
# plt.show()

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - L) < DOLFIN_EPS and on_boundary


# Boundary conditions on displacement
all_boundary = AllBoundary()
# Boundary conditions on rotations
left = Left()
right = Right()
# lower = Lower()
# upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
# lower.mark(boundaries, 3)
# upper.mark(boundaries, 4)

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

# Finite element defition

P_p = FiniteElement('DG', mesh.ufl_cell(), deg-1)
P_q = FiniteElement('CG', mesh.ufl_cell(), deg)

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

r = Expression("x[0]", degree=2)
# r = 1.
m_p = v_p * al_p * r * dx  # inner(v_p, al_p) * dx
m_q = v_q * al_q * r * dx
# m_q = inner(v_q, al_q) * dx

Q_p = v_p * al_p * dx  # inner(v_p, al_p) * dx
Q_q = v_q * al_q * dx  # inner(v_q, al_q) * dx

j_grad = -v_p * (al_q).dx(0) * r * dx  # -dot(v_p, grad(al_q) * dx
j_gradIP = +v_q.dx(0) * al_p * r * dx
# j_gradIP = +v_q.dx(0) * al_p / r  *  dx # dot(grad(v_q), al_p) * dx
#
# j_div = -v_q * al_p.dx(0) * dx # -v_q * div(al_p)) * dx
# j_divIP = v_p.dx(0) * al_q *dx # div(v_p)* al_q * dx
#

B_l = assemble(-v_q * r * ds(1))
B_r = assemble(v_q * r * ds(2))

B_l = B_l.get_local()
B_lT = np.transpose(B_l)
B_r = B_r.get_local()
B_rT = np.transpose(B_r)

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

D_q_tilde = (la.inv(M_p) @ D_q @ la.inv(M_q))
D_p_tilde = (la.inv(M_q) @ D_p @ la.inv(M_p))

# r = Expression("sqrt(r*r+x[1]*x[1])", degree=2)
# theta = Expression("atan2(x[1],r)", degree=2)
#

# Final Assemble

al_p_ = Function(Vp)
al_q_ = Function(Vq)

Hdes = 1.
h_eq_ = Function(Vq)
h_eq_.vector()[:] = Hdes

Hd = 0.5 * (1. / rho * al_q_ * dot(al_p_, al_p_) + rho * g * al_q_ ** 2) * r * dx
Lyap = 0.5 * (1. / rho * al_q_ * dot(al_p_, al_p_) + rho * g * (al_q_ - Hdes) ** 2) * r * dx

e_p_ = derivative(Hd, al_p_)
e_q_ = derivative(Hd, al_q_)

M = la.block_diag(M_p, M_q)
J = np.zeros((n_V, n_V))
J[:n_Vp, n_Vp:n_V] = D_q
J[n_Vp:n_V, :n_Vp] = D_p

invM_q = la.inv(M_q)
invM_p = la.inv(M_p)
invM = la.inv(M)
Jtilde = invM @ J @ invM

# Jtilde = (Jtilde - Jtilde.T)/2.

# Stormer Verlet integrator
B = np.concatenate((np.zeros((n_Vp,)), B_r), axis=0).reshape((-1, 1))
z = 0.001
R = z * B @ B.T
Rtilde = invM @ R @ invM

Btilde = (invM @ B)[:, -1]

al_p_.assign(Constant(0))
al_q_.assign(Constant(Hdes))
e0_p = assemble(e_p_).get_local()
e0_q = assemble(e_q_).get_local()
e0 = np.concatenate((e0_p, e0_q), axis=0)


def e_q_fenics(p, q):
    al_p_.vector()[:] = 1. * p
    al_q_.vector()[:] = 1. * q
    return assemble(e_q_).get_local()


def e_p_fenics(p, q):
    al_p_.vector()[:] = 1. * p
    al_q_.vector()[:] = 1. * q
    return assemble(e_p_).get_local()


def e_q_mat(p, q):
    return M_q @ q * rho * g


def e_p_mat(p, q):
    return M_p @ p / rho


amplitude = 0.3


def fun(t, y):
    al_p = y[:n_Vp]
    al_q = y[n_Vp:n_V]

    # e_p = invM_p @ al_p / rho
    # e_q = invM_q @ al_q * rho * g

    e_p = e_p_fenics(al_p, al_q)
    e_q = e_q_fenics(al_p, al_q)

    e = np.concatenate((e_p, e_q), axis=0)

    dydt = Jtilde @ e - 0 * Rtilde @ (e - e0) * (t > 0.5) + Btilde * amplitude * sin(4 * pi * t) * (t < .25)

    return dydt


def funlin(t, y):
    al_p = y[:n_Vp]
    al_q = y[n_Vp:n_V]

    # e_p = invM_p @ al_p / rho
    # e_q = invM_q @ al_q * rho * g

    e_p = e_p_mat(al_p, al_q)
    e_q = e_q_mat(al_p, al_q)

    e = np.concatenate((e_p, e_q), axis=0)

    dydt = Jtilde @ e - 0 * Rtilde @ (e - e0) * (t > 0.5) + Btilde * amplitude * sin(4 * pi * t) * (t < .25)

    return dydt


def funSE(t, p, q):
    e_p = e_p_fenics(p, q)
    e_q = e_q_fenics(p, q)
    dpdt = D_q_tilde @ e_q
    dqdt = D_p_tilde @ e_p
    Aqinv = np.eye(len(q))
    return dpdt, dqdt, Aqinv


h = 0.01
init_p = Expression('0', degree=0)
init_q = Expression('(H + h *cos(pi/L*r)) ', degree=4, H=Hdes, h=h, L=L, r=r)

al_p_.assign(interpolate(init_p, Vp))
al_q_.assign(interpolate(init_q, Vq))

alp_0 = al_p_.vector().get_local()
alq_0 = al_q_.vector().get_local()

ep_0 = assemble(e_p_).get_local()
eq_0 = assemble(e_q_).get_local()

y0 = np.concatenate((alp_0, alq_0), axis=0)

t0 = 0.
t_fin = 0.5
n_t = 300

# simulation using IVP
t_span = [t0, t_fin]
t_ev = np.linspace(t0, t_fin, num=n_t)
sol = integrate.solve_ivp(fun, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev, \
                          atol=1e-8, rtol=1e-8)
al_sol = sol.y
t_ev = sol.t
n_ev = len(t_ev)

# linear simulation:
sollin = integrate.solve_ivp(funlin, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev, \
                             atol=1e-8, rtol=1e-8)
al_sollin = sollin.y
t_evlin = sollin.t

# simulation using Symplectic Euler;

# from SympEuler import SympEuler

# dt = 1e-3
# t_evSE, p_SE, q_SE = SympEuler(funSE, alp_0, alq_0, t_fin, t0, dt, n_ev = n_t)
# y_SE = np.concatenate((p_SE, q_SE), axis=0)
# al_sol = y_SE
# t_ev = t_evSE
# n_ev = len(t_ev)


alp_sol = al_sol[:n_Vp]
alq_sol = al_sol[n_Vp:n_V]

alp_sollin = al_sollin[:n_Vp]
alq_sollin = al_sollin[n_Vp:n_V]


# minZ = min(alq_sol)
# maxZ = max(alq_sol)


if matplotlib.is_interactive():
    plt.ioff()
plt.close('all')


def HamFunc(p, q):
    # this method should work for both linear and nonlinear Hamiltonian
    al_p_.vector()[:] = 1. * p
    al_q_.vector()[:] = 1. * q
    return assemble(Hd)


def HamFuncQuad(p, q):
    return (q.T @ M_q @ q * g * rho + p.T @ M_p @ p / rho) * 0.5


# def HamFuncQuadNew(p, q):
#     return (q.T @ invM_q @ q * g * rho + p.T @ invM_p @ p / rho) * 0.5


def Energia_q(q):
    return (q.T @ M_q @ q * rho * g) * 0.5


def Energia_p(p):
    return (p.T @ M_p @ p / rho) * 0.5


def LyaFunc(p, q):
    # this method should work for both linear and nonlinear Hamiltonian
    al_p_.vector()[:] = 1. * p
    al_q_.vector()[:] = 1. * q
    return assemble(Lyap)


H_vec = np.zeros((n_ev))
Energia_p_vec = np.zeros((n_ev))
Energia_q_vec = np.zeros((n_ev))
H_vec_quad = np.zeros((n_ev))
V_vec = np.zeros((n_ev))
for i in range(n_ev):
    H_vec[i] = HamFunc(alp_sol[:, i], alq_sol[:, i])  #
    H_vec_quad[i] = HamFuncQuad(alp_sollin[:, i], alq_sollin[:, i])  # computed with fenics
    # H_vec_quad[i] = HamFuncQuadNew( alp_sol[:,i],  alq_sol[:,i]) # computed with quadratic form
    # Energia_p_vec[i] = Energia_p(alp_sol[:,i])
    # Energia_q_vec[i] = Energia_q(alq_sol[:,i])
    # V_vec[i] = LyaFunc(invM_p @ alp_sol[:,i], invM_q @ alq_sol[:,i])
#

fntsize = 16


path_out = "../results/1D/"
plt.figure()
plt.plot(t_ev, H_vec, 'r-')
plt.plot(t_ev, H_vec_quad, 'b-')
plt.legend(['nonlinear', 'linear'])
# # plt.plot(t_ev, Energia_q_vec, 'r-')
# # plt.plot(t_ev, Energia_p_vec, 'g-')
# # plt.plot(t_ev, V_vec + rho*g/2, 'b-')
plt.xlabel(r'{Time} (s)', fontsize=fntsize)
plt.ylabel('Total Energy (J)', fontsize=fntsize)
plt.savefig(path_out + "Hamiltonian.eps", format="eps")

r_ev = dofVq_x[:, 0]

# Make data.

fig = plt.figure()
ax = fig.gca(projection='3d')
X_ev, T_ev = np.meshgrid(r_ev, t_ev)

# Customize the z axis.

# ax.set_zlim(minZ-1e-3*abs(minZ) , maxZ+1e-3*abs(maxZ))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.06f'))

# Plot the surface.
tol = 1e-4
ax.set_xlim(min(r_ev) - tol, max(r_ev) + tol)
ax.set_xlabel('Radial coordinate [m]', fontsize=fntsize)
ax.set_ylim(min(t_ev) - tol * abs(min(t_ev)), max(t_ev) + tol * abs(max(t_ev)))
ax.set_ylabel('Time [s]', fontsize=fntsize)
ax.set_zlabel('h [m]', fontsize=fntsize)
ax.set_title('Fluid Height', fontsize=fntsize)

ax.w_zaxis.set_major_locator(LinearLocator(10))
ax.w_zaxis.set_major_formatter(FormatStrFormatter('%3.2f'))

Z = np.transpose(alq_sollin)
h_plot = ax.plot_surface(X_ev, T_ev, Z, cmap=cm.jet, \
                         linewidth=0, antialiased=False, label='Wave $w$')

# Add a color bar which maps values to colors.
h_plot._facecolors2d = h_plot._facecolors3d
h_plot._edgecolors2d = h_plot._edgecolors3d

fig.colorbar(h_plot, shrink=0.5, aspect=5)

plt.savefig(path_out + "Height.eps", format="eps")
plt.show()

n_fig = 4


minZ = alq_sol.min()
maxZ = alq_sol.max()

for i in range(n_fig + 1):
    index = int((n_ev - 1) / n_fig * (i))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #    surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False, 'vmin': minZ, 'vmax': maxZ}

    # ax.set_xbound(min(x_plot) - tol, max(x_plot) + tol)
    ax.set_xlabel('Radial coordinate [m]', fontsize=fntsize)

    ax.set_ylabel('Fluid Height [m]', fontsize=fntsize)
    # ax.set_zlabel('$h [m]$', fontsize=fntsize)
    # ax.set_title('Fluid Height', fontsize=fntsize)

    # ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))
    # ax.w_zaxis.set_major_locator(LinearLocator(10))
    # ax.w_zaxis.set_major_formatter(FormatStrFormatter('%3.2f' ))
    plt.plot(r_ev, alq_sol[:, index], 'r-')
    plt.plot(r_ev, alq_sollin[:, index], 'b-')
    # ax.plot_trisurf(x_plot, y_plot, alq_sol[:,index], **surf_opts)
    plt.ylim(minZ, maxZ)
    plt.legend(['Nonlinear', 'Linear'], fontsize=fntsize)
    plt.savefig(path_out + "Snap1d_n" + str(index + 1) + "Large.eps", format="eps")
#

plt.show()
#
# plt.figure()
# plt.plot(t_ev, V_vec, 'g-', label = 'Lyapunov Function (J)')
# plt.xlabel(r'{Time} (s)',fontsize=fntsize)
# plt.ylabel('Lyapunov Function (J)' ,fontsize=fntsize)
# plt.show()
# # plt.savefig(path_out + "Lyapunov.eps", format="eps")


# plt.figure()
# plt.plot(r_ev, alq_sol[:, -1], 'r-')
# plt.plot(r_ev, alq_sollin[:, -1], 'b-')
# # plt.plot(t_ev, Energia_q_vec, 'r-')
# # plt.plot(t_ev, Energia_p_vec, 'g-')
# # plt.plot(t_ev, V_vec + rho*g/2, 'b-')
# plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# plt.ylabel('Height', fontsize=fntsize)
#
# plt.show()


#perm = np.argsort(r_ev)
#r_ev.sort()
#alq_sol = alq_sol[perm, :]
#alq_sollin = alq_sollin[perm, :]
#
#n_r = len(r_ev)
#n_th = 10
#th_ev = np.linspace(0, 2 * pi, num=n_th)
#
#r_plot = np.tile(r_ev, n_th)
#th_plot = np.repeat(th_ev, n_r)
#alq_sol_plot = np.tile(alq_sol, (n_th, 1))
#alq_sollin_plot = np.tile(alq_sollin, (n_th, 1))


# from matplotlib import animation
#
# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(0, L), ylim=(Hdes - 1.01 * h, Hdes + 1.01 * h))
# line, = ax.plot([], [], lw=2)
#
#
# # initialization function: plot the background of each frame
# def init():
#     line.set_data(r_ev, alq_0)
#     return line,
#
#
# # animation function.  This is called sequentially
# def animate(i):
#     line.set_data(r_ev, alq_sol[:, i])
#     return line,
#
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=len(t_ev), interval=20, blit=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()


# ### plot da velocidade
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X_ev, T_ev = np.meshgrid(dofVp_x, t_ev)

# # Customize the z axis.

# # ax.set_zlim(minZ-1e-3*abs(minZ) , maxZ+1e-3*abs(maxZ))
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.06f'))

# # Plot the surface.
# tol = 1e-4
# ax.set_xlim(min(r_ev) - tol, max(r_ev) + tol)
# ax.set_xlabel('Radial coordinate $[m]$', fontsize=fntsize)
# ax.set_ylim(min(t_ev) - tol* abs(min(t_ev)), max(t_ev) + tol* abs(max(t_ev)))
# ax.set_ylabel('Time $[s]$', fontsize=fntsize)
# ax.set_zlabel('$h [m]$', fontsize=fntsize)
# ax.set_title('Fluid Velocity', fontsize=fntsize)

# ax.w_zaxis.set_major_locator(LinearLocator(10))
# ax.w_zaxis.set_major_formatter(FormatStrFormatter('%3.2f'))

# Z = np.transpose(alp_sol)
# h_plot = ax.plot_surface(X_ev, T_ev, Z, cmap=cm.jet, \
#                        linewidth=0, antialiased=False, label = 'Wave $w$')

# plt.show()


# from SaintVenant_fenics.AnimateSurf import animate2D
#
# x_plot = r_plot * np.cos(th_plot)
# y_plot = r_plot * np.sin(th_plot)
#
# anim = animate2D(x_plot, y_plot, alq_sol_plot, t_ev, xlabel='$x [m]$', ylabel='$y [m]$', zlabel='$h [m]$',
#                  title='Fluid height')

# plt.show()
# rallenty = 0.2
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=n_ev / t_fin * rallenty, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('SWE1D.mp4', writer=writer)
