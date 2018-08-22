import h5py
import matplotlib.pyplot as plt
import numpy as np
def load_file(num,base='snapshot',gamma=1.0001):
    with h5py.File('{}_{:03d}.hdf5'.format(base,num),'r') as f:
        time = f['Header'].attrs['Time']
        try:
            grp = f['PartType3']
            vel = grp['Velocities'][...][0]
            pos = grp['Coordinates'][...][0]
            m = grp['Masses'][...][0]
            st = Star(mass=m,pos=pos,vel=vel)
        except:
            print('No star')
            st = Star()


        density = f['PartType0/Density'][...]
        mass = f['PartType0/Masses'][...]
        e = f['PartType0/InternalEnergy'][...]
        temp = gamma*e
        #pres = e*density *(gamma-1)
        vel = f['PartType0/Velocities'][...]
        coords = f['PartType0/Coordinates'][...]
        try:
            pot = f['PartType0/Potential'][...]
        except:
            pot = np.zeros(density.shape)
    # vel = np.array(ad[("PartType0","Velocities")])
    # coords=  np.array(ad['PartType0','Coordinates'])
    x = coords[:,0]
    y = coords[:,1]
    r = np.sqrt(x**2+ y**2)
    phi = np.arctan2(y,x)
    vr = np.cos(phi) * vel[:,0] + np.sin(phi) * vel[:,1]
    vphi = -np.sin(phi) * vel[:,0] + np.cos(phi) * vel[:,1]
    return x,y,r,phi,density,vr,vphi,temp,mass,pot,st,time
class Star():
    def __init__(self,mass=1,pos=np.zeros((3,)),vel=np.zeros((3,))):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.vx = vel[0]
        self.vy = vel[1]
        self.x = pos[0]
        self.y = pos[1]
        self.r = np.sqrt(self.x**2 + self.y**2)
    def plot_pos(self,ax=None,ms=5,c='k',**kargs):
        if ax is None:
            return
        ax.plot(self.x,self.y,'*',ms=ms,c=c,**kargs)
class Snap():
    def __init__(self,num=0,base='out/snapshot',gamma=1.0001):
        self.x,self.y,self.r,self.phi,self.dens,self.vr,self.vp,self.temp,self.mass,self.pot,self.star,self.time = load_file(num,base=base,gamma=gamma)
        self.vol = self.mass/self.dens
        self.gamma = gamma
    def plot2d(self,val='dens',func=None,rmax=3,rmin=.05,cart=True,lims=None,fig=None,ax=None,**kargs):
        import matplotlib.colors as colors
        import matplotlib.tri as tri

        if ax is None:
            fig,ax=plt.subplots(figsize=(6,6))
        ind = (self.r <= rmax)&(self.r>=rmin)



        if cart:
            x = self.x[ind].copy()
            y = self.y[ind].copy()
            lbl = ('$x$','$y$')
        else:
            x = self.r[ind].copy()
            y = self.phi[ind].copy()
            lbl = ('$r$','$\\phi$')
        triang = tri.Triangulation(x, y)
        triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                             y[triang.triangles].mean(axis=1))< rmin)

        if func is None:
            try:
                q = getattr(self,val)[ind]
            except AttributeError:
                print(val, 'not a valid field!')
                return None,None
        else:
            q = func(self)[ind]

        norm = kargs.pop('norm',colors.Normalize())
        shading = kargs.pop('shading','gouraud')
        cmap = kargs.pop('cmap','viridis')
        ax.tripcolor(triang,q,norm=norm,shading=shading,cmap=cmap,**kargs)
        _create_colorbar(ax,norm,cmap=cmap)

        if lims is not None:
            ax.set_xlim(*lims[:2])
            ax.set_ylim(*lims[-2:])

        ax.minorticks_on()
        ax.tick_params(labelsize=20)
        ax.set_xlabel(lbl[0],fontsize=20)
        ax.set_ylabel(lbl[1],fontsize=20)
        fig.tight_layout()
        return fig,ax
    def point_plot(self,rmax=10,ms=.5,fig=None,ax=None,**kargs):
        if ax is None:
            fig,ax=plt.subplots(figsize=(6,6))
        ind = (self.r <= rmax)

        ax.plot(self.x[ind],self.y[ind],'.',ms=ms)

        self.star.plot_pos(ax)
        ax.plot(0,0,'.',ms=5,c='r')

        ax.minorticks_on()
        ax.tick_params(labelsize=20)
        ax.set_xlabel('$x$',fontsize=20)
        ax.set_ylabel('$y$',fontsize=20)
        fig.tight_layout()
        return fig,ax

    def plot1d(self,val='dens',func=None,ms=.1,rmin=.1,rmax=10,fig=None,ax=None,logx=True,logy=True,ylbl='',**kargs):
        if ax is None:
            fig,ax=plt.subplots(figsize=(8,6))
        ind = (self.r <= rmax)&(self.r>=rmin)


        if func is None:
            try:
                q = getattr(self,val)[ind]
            except AttributeError:
                print(val, 'not a valid field!')
                return None,None
        else:
            q = func(self)[ind]

        ax.plot(self.r[ind],q,'.',ms=ms,**kargs)
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('$r$',fontsize=20)
        ax.tick_params(labelsize=20)
        ax.set_ylabel(ylbl,fontsize=20)
        return fig,ax
    def circ(self,r,ax=None,**kargs):
        phi = np.linspace(-np.pi,np.pi,1000)
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        if ax is not None:
            ax.plot(x,y,'--w',**kargs)
        return x,y
    def avg(self,x,val='dens'):

        try:
            q = getattr(self,val)
        except AttributeError:
            print(val, 'not a valid field!')
            return None

        nx = len(x)
        dx = np.diff(x)[0]
        bins = np.zeros(x.shape)
        tot = np.zeros(x.shape)
        for r,d in zip(self.r,q):
            i = int((r-.1)/dx)
            if (i>=0)&(i<nx):
                tot[i] += 1
                bins[i] += d
        bins /= tot
        return bins
def _create_colorbar(ax,norm,cax=None,cmap='viridis',**kargs):
    """
    Function to create a colorbar at the top of a plot

    Parameters
    ----------
    ax : matplotlib.axis.Axis
        The axis object we want to draw the colorbar over.

    vmin : float
        The minimum value for the colorbar

    vmax : float
        The maximum value for the colorbar

    log : bool
        If log==True then the colorscale is logscale
    cmap : str
        The colormap to use for the colorbar
    **kargs :
        Extra keyword arguments which are passed to matplotlib.colorbar.ColorbarBase


    Returns
    -------
    cb : matplotlib.colorbar
        The final colorbar.

    """
    import matplotlib
    import matplotlib.cm
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    labelsize = kargs.pop('labelsize',12)
    upper_lim = kargs.pop('upper_lim',False)
    lower_lim = kargs.pop('lower_lim',False)

    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top',size='3%',pad=.05)


    cmap = matplotlib.cm.get_cmap(cmap)
    cb = matplotlib.colorbar.ColorbarBase(ax=cax,cmap=cmap,norm=norm,orientation='horizontal',**kargs)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(labelsize=labelsize)



    if upper_lim:
        cb_lbls = cb.ax.get_xticklabels()
        cb_lbls[-1].set_text('$>$'+cb_lbls[-1].get_text())
        cb.ax.set_xticklabels(cb_lbls)
    if lower_lim:
        cb_lbls = cb.ax.get_xticklabels()
        cb_lbls[0].set_text('$<$'+cb_lbls[0].get_text())
        cb.ax.set_xticklabels(cb_lbls)


    return cb
