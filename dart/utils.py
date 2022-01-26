import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

def load_crosssections(path_to_file):
    """
    Load wake data from file

    Args:
        path_to_file: location where file is saved
    """
    with nc.Dataset(path_to_file, "r") as nc_file:
        z = np.asarray(nc_file.variables['z'])
        y = np.asarray(nc_file.variables['y'])
        x = np.asarray(nc_file.variables['x'])
        u = np.asarray(nc_file.variables['u_normdef'])
    return z,y,x,u

def plot_wake(wake_dict,D):
    """
    Quick plotting function that creates a cross-section

    Args:
        wake_dict: containing y,z and u_normdef data
        D: distance downstream
    """
    fig,ax = plt.subplots()
    im = plt.contourf(wake_dict["y"],wake_dict["z"],wake_dict["u_normdef"][:,:,D],15,cmap='jet',vmax=0)
    ax.set_xticks(np.arange(-1.5,2.,0.5))
    ax.set_xticklabels(np.arange(-1.5,2.,0.5))
    ax.set_yticks(np.arange(-0.5,2.,0.5))
    ax.set_yticklabels(np.arange(-0.5,2.,0.5))
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(top=1.)
    ax.set_aspect('equal')
    circ = plt.Circle((0, 0),0.5,edgecolor='grey',fill=False,linewidth=2)
    ax.add_artist(circ) ## rotor area
    plt.plot([0,0],[wake_dict["z"][0],0],'grey') ## tower
    ax.set_aspect('equal')
    plt.colorbar(im,fraction=0.026, pad=0.04,label=r'$u_{normdef}$')
    plt.ylabel('z/D')
    plt.xlabel('y/D')
    plt.gca().invert_xaxis() ## invert x-axis to view for upstream
    plt.show() 
