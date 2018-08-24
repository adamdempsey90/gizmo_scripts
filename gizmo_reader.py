import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
class SPHpoint():
    fields = ['dens','vr','vp','temp','pot']
    def __init__(self,x,y,h,vol,mass,val):
        self.x = x
        self.y = y
        self.r = np.sqrt(x**2 + y**2)
        self.h = h
        self.val = val
        self.vol = vol
        self.mass = mass
        norm = 40./(7*np.pi)
        self.ih3 = norm/(2*h)**2
        for i,name in enumerate(self.fields):
            setattr(self,name,val[i])
    def dist(self,x,y):
        return np.sqrt((x-self.x)**2 + (y-self.y)**2)
    def kernel(self,u):
        try:
            wk = np.zeros(u.shape)
        except AttributeError:
            return self.kernel_single(u)

        ind = (u < .5)&(u>=0)
        nind = (u>=.5)&(u<=1)
        wk[ind] = self._kernel_lt(u[ind])
        wk[nind] = self._kernel_gt(u[nind])
        return wk
    def kernel_single(self,u):
        if u<0 or u>1:
            return 0.
        if u < .5:
            return self._kernel_lt(u)

        return self._kernel_gt(u)

    def _kernel_lt(self,u):
        wk = (1.0 + 6.0 * (u - 1.0) * u * u)
        return wk*self.ih3
    def _kernel_gt(self,u):
        u2 = (1.-u)**2
        wk = 2.0*(1.0-u) * u2
        return wk*self.ih3
    def weight(self,x,y):
        u = self.dist(x,y)/(2*self.h)
        wk = self.kernel(u)
        return wk #*self.vol

def convert_to_sph(fld):
    return  [SPHpoint(fld.x[i],fld.y[i],fld.smooth[i],fld.vol[i],fld.mass[i],
              np.array([fld.dens[i],fld.vr[i],fld.vp[i],fld.temp[i],fld.pot[i]])) for i in range(len(fld.x))]

class Mesh():
    def __init__(self,ri,ro,nr):
        nphi = int(2*np.pi*nr/np.log(ro/ri))
        r = np.exp(np.linspace(np.log(ri),np.log(ro),nr))
        phi = np.linspace(-np.pi,np.pi,nphi)
        self.r = r
        self.phi = phi

    def add_points(self,points):
        rr,pp,zz = self.interpolate(points,self.r,self.phi)
        self.rr = rr
        self.pp = pp
        self.xx = rr*np.cos(pp)
        self.yy = rr*np.sin(pp)
        shape = len(self.r),len(self.phi)
        for i,name in enumerate(points[0].fields):
            setattr(self,name,zz[:,i].reshape(shape))
    def plot2d(self,val='dens',func=None,rmax=10,rmin=0,cart=True,lims=None,logx=False,logy=False,
               fig=None,ax=None,**kargs):
        import matplotlib.colors as colors

        if ax is None:
            fig,ax=plt.subplots(figsize=(6,6))


        indx = (self.r <= rmax)&(self.r>=rmin)



        if cart:
            x = self.xx[indx,:]
            y = self.yy[indx,:]
            lbl = ('$x$','$y$')
        else:
            x = self.r[indx,:]
            y = self.phi[indx,:]
            lbl = ('$r$','$\\phi$')

        if func is None:
            try:
                q = getattr(self,val)[indx,:]
            except AttributeError:
                print(val, 'not a valid field!')
                return None,None
        else:
            q = func(self)[indx,:]

        norm = kargs.pop('norm',colors.Normalize())
        shading = kargs.pop('shading','gouraud')
        cmap = kargs.pop('cmap','viridis')
        ax.pcolormesh(x,y,q,norm=norm,shading=shading,cmap=cmap,**kargs)
        _create_colorbar(ax,norm,cmap=cmap)

        if lims is not None:
            ax.set_xlim(*lims[:2])
            ax.set_ylim(*lims[-2:])

        ax.minorticks_on()
        ax.tick_params(labelsize=20)
        ax.set_xlabel(lbl[0],fontsize=20)
        ax.set_ylabel(lbl[1],fontsize=20)
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        fig.tight_layout()
        return fig,ax
    def semilogx(self,**kargs):
        logx = kargs.pop('logx',True)
        logx = True
        logy = kargs.pop('logy',False)
        logy = False

        return self.plot(logx=logx,logy=logy,**kargs)
    def semilogy(self,**kargs):
        logx = kargs.pop('logx',False)
        logx = False
        logy = kargs.pop('logy',True)
        logy = True

        return self.plot(logx=logx,logy=logy,**kargs)
    def loglog(self,**kargs):
        logx = kargs.pop('logx',True)
        logx = True
        logy = kargs.pop('logy',True)
        logy = True

        return self.plot(logx=logx,logy=logy,**kargs)
    def plot(self,val='dens',func=None,ylbl='',rmax=10,rmin=0,cart=True,logx=False,logy=False,fig=None,ax=None,**kargs):
        import matplotlib.colors as colors

        if ax is None:
            fig,ax=plt.subplots(figsize=(8,6))


        indx = (self.r <= rmax)&(self.r>=rmin)


        if func is None:
            try:
                q = getattr(self,val).mean(axis=1)[indx]
            except AttributeError:
                print(val, 'not a valid field!')
                return None,None
        else:
            q = func(self).mean(axis=1)[indx]

        ax.plot(self.r[indx],q,**kargs)
        ax.minorticks_on()
        ax.tick_params(labelsize=20)
        ax.set_xlabel('$r$',fontsize=20)
        ax.set_ylabel(ylbl,fontsize=20)


        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        fig.tight_layout()
        return fig,ax

    def get_mesh(self,r,phi):
        rr,pp = np.meshgrid(r,phi,indexing='ij')
        xx,yy = rr*np.cos(pp),rr*np.sin(pp)
        xx = xx.ravel()
        yy = yy.ravel()
        mesh_points = np.vstack((xx,yy)).T
        mesh_tree = KDTree(mesh_points)
        return xx,yy,rr,pp,mesh_tree
    def get_index_list(self,points,mesh_tree,rmin=.3,rmax=3):
        dist = lambda h: 2.1*h
        inds = [mesh_tree.query_radius([np.array([p.x,p.y])],r=dist(p.h))[0] if (p.r - dist(p.h)<= rmax)&(p.r + dist(p.h)>=rmin) else np.array([])
               for p in points]
        return inds
    def interpolate(self,points,r,phi):
        rmin = r.min()
        rmax = r.max()
        xx,yy,rr,pp,mesh_tree = self.get_mesh(r,phi)
        inds = self.get_index_list(points,mesh_tree,rmin=rmin,rmax=rmax)
        nf = len(points[0].val)
        zz = np.zeros(xx.shape+(nf,))
        for ind,p in zip(inds,points):
            if len(ind) > 0:
                w = p.weight(xx[ind],yy[ind])
                for j,wj in zip(ind,w):
                    zz[j,:] += wj*p.vol*p.val

        return rr,pp,zz

def load_file(num,base='snapshot',gamma=1.0001):
    with h5py.File('{}_{:03d}.hdf5'.format(base,num),'r') as f:
        time = f['Header'].attrs['Time']
        try:
            grp = f['PartType3']
            vel = grp['Velocities'][...]
            pos = grp['Coordinates'][...]
            masses = grp['Masses'][...]
            st = [Star(mass=m,pos=p,vel=v) for m,p,v in zip(masses,pos,vel)]
        except:
            print('No star')
            st = [Star()]


        density = f['PartType0/Density'][...]
        mass = f['PartType0/Masses'][...]
        smooth = f['PartType0/SmoothingLength'][...]
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
    return x,y,r,phi,density,vr,vphi,temp,mass,pot,smooth,st,time
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
    def __init__(self,num=0,base='out/snapshot',gamma=1.0001,shift=False):
        self.x,self.y,self.r,self.phi,self.dens,self.vr,self.vp,self.temp,self.mass,self.pot,self.smooth,self.star,self.time = load_file(num,base=base,gamma=gamma)

        if shift and len(self.star)>1:
            p0 = np.arctan2(self.star[1].pos[1],self.star[1].pos[0])
            self.phi -= p0
            self.x = self.r*np.cos(self.phi)
            self.y = self.r*np.sin(self.phi)
            for s in self.star:
                p = np.arctan2(s.y,s.x)
                p -= p0
                r = np.sqrt(s.x**2 + s.y**2)
                s.x = r*np.cos(p)
                s.y = r*np.sin(p)


        self.vol = self.mass/self.dens
        self.gamma = gamma

    def to_mesh(self,ri,ro,nr):
        print('Converting to uniform mesh')
        self.mesh = Mesh(ri,ro,nr)
        print('{:d} x {:d} points'.format(len(self.mesh.r),len(self.mesh.phi)))
        print('Interpolating...')

        self.mesh.add_points(convert_to_sph(self))

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

        for s in self.star:
            s.plot_pos(ax)
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
        xm = min(x)
        for r,d in zip(self.r,q):
            i = int((r-xm)/dx)
            if (i>=0)&(i<nx):
                tot[i] += 1
                bins[i] += d
        bins /= (2*np.pi*dx*x)
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
