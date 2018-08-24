# gizmo_scripts
Scripts to run/read/visualize Gizmo for 2D Keplerian disks. Gizmo is available at https://bitbucket.org/phopkins/gizmo-public.


To load and plot the snapshot located in, for example, out/snapshot_100.hdf5,

```
>>> import gizmo_scripts as gs
>>> fld = gs.Snap(100,base='out/snapshot')
```
To get a plot of where all of the particles are,
```
>>> fld.point_plot(ms=.5)
```
To make 2D plots of fluid quantities we can first remap to a uniform mesh, taking into account the SPH kernel. 
By default we map to a cylindrical mesh with equal aspect ratio cells. To map to a mesh from r = .3 -> 2 with 100 points,
```
>>> fld.to_mesh(.3,2,100)  
```
You can then make 2D plots, azimuthally averaged plots, take FFTs, etc.
```
>>> import matplotlib.colors as colors
>>> fld.mesh.plot2d(val='dens',norm = colors.LogNorm())
>>> fld.mesh.loglog(val='dens')
```
