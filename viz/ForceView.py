import matplotlib.pyplot as plt
import numpy as np



def plot_vector_field(area, FF):
    ax = plt.gca()
    # creating a meshgrid for the vector field
    x, y = np.arange(area[0], area[1]), np.arange(area[0], area[1])
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    nx, ny = len(x), len(y)
    U, V = np.zeros(xv.shape), np.zeros(xv.shape)
    for i in range(nx):
        for j in range(ny):
            r = [xv[i, j], yv[i, j]]
            # print ('r',r)
            F = FF(r)
            U[i, j] = F[0]
            V[i, j] = F[1]

    #  showing vector field direction
    Q = ax.quiver(xv, yv, U, V, units='width')
    ax.quiverkey(Q,  X=1, Y=1, U=1, label=r'$2 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')



    # showing obstacles, start and goal locations

    if FF.start:
        x, y = FF.start
        circleS = plt.Circle((x, y), 2, color='yellow')
        ax.add_artist(circleS)


    x, y = FF.goal
    # print(FF.goal)
    circle1 = plt.Circle((x, y), 2, color='cyan')
    ax.add_artist(circle1)
    for i , obs in enumerate(FF.obstacles):
        x, y  = obs
        r = FF.obstacle_radius[i]
        circle1 = plt.Circle((x, y), r, color='r')
        ax.add_artist(circle1)


