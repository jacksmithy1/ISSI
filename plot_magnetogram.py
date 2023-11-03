import matplotlib.pyplot as plt
import numpy as np
import fieldline3D
import datetime


def plot_magnetogram_boundary(data_bz, nresol_x, nresol_y):
    x_arr = np.arange(nresol_x) * (nresol_x) / (nresol_x - 1)
    y_arr = np.arange(nresol_y) * (nresol_y) / (nresol_y - 1)
    x_plot = np.outer(y_arr, np.ones(nresol_x))
    y_plot = np.outer(x_arr, np.ones(nresol_y)).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(y_plot, x_plot, data_bz, 1000, cmap="bone")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def plot_magnetogram_boundary_3D(
    data_bz, nresol_x, nresol_y, xmin, xmax, ymin, ymax, zmin, zmax
):
    X = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    Y = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    Xgrid, Ygrid = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(
        Xgrid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        Ygrid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        data_bz[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        1000,
        cmap="bone",
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim([zmin, zmax])
    ax.view_init(30, 245)
    plt.show()


def plot_fieldlines_grid(
    data_b,
    h1,
    hmin,
    hmax,
    eps,
    nresol_x,
    nresol_y,
    nresol_z,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    a,
    b,
    alpha,
    nf_max,
):
    data_bz = data_b[:, :, 0, 2]

    X = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    Y = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    Z = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    Xgrid, Ygrid = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(
        Xgrid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        Ygrid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        data_bz[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        1000,
        cmap="bone",
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim([zmin, zmax])
    ax.view_init(30, 245)
    ax.view_init(90, 270)
    ax.set_box_aspect((xmax, ymax, 1))

    nlinesmaxx = 10
    nlinesmaxy = 10
    x_0 = 0.00000001
    y_0 = 0.00000001
    dx = 0.1
    dy = 0.1

    nlinesmaxx = 40
    nlinesmaxy = 20
    x_0 = 0.0
    y_0 = 0.0
    dx = 0.05
    dy = 0.05

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = 0.0
    boxedges[1, 0] = ymax
    boxedges[0, 1] = 0.0
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]

            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline = fieldline3D.fieldline3d(
                ystart,
                data_b,
                Y,
                X,
                Z,
                h1,
                hmin,
                hmax,
                eps,
                oneway=0,
                boxedge=boxedges,
                gridcoord=0,
                coordsystem="cartesian",
            )  # , periodicity='xy')

            # Plot fieldlines
            fieldline_x = np.zeros(len(fieldline))
            fieldline_y = np.zeros(len(fieldline))
            fieldline_z = np.zeros(len(fieldline))
            fieldline_x[:] = fieldline[:, 0]
            fieldline_y[:] = fieldline[:, 1]
            fieldline_z[:] = fieldline[:, 2]

            # Need to give row direction first/ Y, then column direction/ X
            ax.plot(
                fieldline_y,
                fieldline_x,
                fieldline_z,
                color="red",
                linewidth=0.5,
                zorder=4000,
            )

    current_time = datetime.datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    plotname = (
        "/Users/lilli/Desktop/ISSI_plots/fieldlines3D_"
        + str(a)
        + "_"
        + str(b)
        + "_"
        + str(alpha)
        + "_"
        + str(nf_max)
        + "_"
        + dt_string
        + ".png"
    )
    plt.savefig(plotname, dpi=300)
    plt.show()
