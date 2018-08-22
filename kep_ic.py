################################################################################
###### This is an example script to generate HDF5-format ICs for GIZMO
######  The specific example below is obviously arbitrary, but could be generalized
######  to whatever IC you need.
################################################################################
################################################################################

## load libraries we will use
import numpy as np
import h5py as h5py

# the main routine. this specific example builds an N-dimensional box of gas plus
#   a collisionless particle species, with a specified mass ratio. the initial
#   gas particles are distributed in a uniform lattice; the initial collisionless
#   particles laid down randomly according to a uniform probability distribution
#   with a specified random velocity dispersion

def get_polar_coords(x,y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    return r,phi
def get_velocities(x,y,z,G=1,M=1,H=0,mu=0,delta=0,**kargs):
    """
        GM = omega_K^2 r^3
        Sigma = r^mu
        c^2 = h^2 * r^delta
        omega^2 = omega_K^2 (1 + h^2 * (mu + delta))
    """
    vr = np.zeros(x.shape)
    vz = np.zeros(x.shape)

    r,phi = get_polar_coords(x,y)


    r1  = np.sqrt(r**2 + kargs['soft']**2)
    omega = np.sqrt(G*M) / r1**(1.5)

    if kargs['ring']:
        grad = dprof_gauss_grad(r,phi,**kargs)
    else:
        grad = dprof_grad(r,phi,**kargs)

    omega *= (1 + H**2 * ( grad + delta))**(.5)



    vx = np.cos(phi) * vr - np.sin(phi) * omega*r
    vy = np.sin(phi) * vr + np.cos(phi) * omega*r
    return vx,vy,vz


def dprof_gauss(r,phi,ri=1,rc=.35,**kargs):
    return np.exp(-((r-ri)/rc)**2)
def dprof_gauss_grad(r,phi,ri=1,rc=1./np.sqrt(8.),**kargs):
    return -2*(r-ri)/rc**2

def dprof(r,phi,mu=-.5,ri=.05,rc=2.,**kargs):
    res =   (1 - np.sqrt(ri/r)) * np.exp(-r/rc) * r**(mu)
    return res
def dprof_grad(r,phi,mu=-.5,ri=.05,rc=2.,**kargs):
# d ln Sigma/d ln r
    res = mu
    res += -r/rc
    res += 1./(-2 + 2*np.sqrt(r/ri))
    return res
def get_density_pressure(x,y,z,H=0,delta=0,sig0=1,sig_floor=.001,**kargs):
    r,phi = get_polar_coords(x,y)
    if kargs['ring']:
        print(sig0)
        density = sig0*dprof_gauss(r,phi,**kargs)
        density += sig_floor
    else:
        density = sig0 * dprof(r,phi,**kargs) / dprof(1.,0.,**kargs)
        density += sig_floor

    c2 = H**2 * r**(delta)
    pressure = c2 * density

    return density,pressure

def makeIC_box(**kargs):
    DIMS=2; # number of dimensions
    N_1D=kargs['N']; # 1D particle number (so total particle number is N_1D^DIMS)
    fname = kargs['fname']

    Lbox = kargs['L'] # box side length
    gamma_eos = kargs['gamma'] # polytropic index of ideal equation of state the run will assume

    # first we set up the gas properties (particle type 0)

    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
   # x0=np.arange(-.5,.5,1./N_1D); x0+=0.5*(0.5-x0[-1]);

    ri = kargs['rinner']
    ro = kargs['router']
    lr = np.linspace(np.log(ri),np.log(ro),N_1D)
    dlr = np.log(ro/ri)/(N_1D-1)

    r0 = np.exp(lr)
    p0 = np.arange(np.ceil(2*np.pi/dlr))*dlr


    rr,pp = np.meshgrid(r0,p0,indexing='xy')

    rr = rr.ravel()
    pp = pp.ravel()
    dA = (rr*dlr)**2

    xv_g = rr*np.cos(pp)
    yv_g = rr*np.sin(pp)
    zv_g = np.zeros(xv_g.shape)


    # now extend that to a full lattice in DIMS dimensions
  #  if(DIMS==3):
  #      xv_g, yv_g, zv_g = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
  #  if(DIMS==2):
  #      xv_g, yv_g = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_g = 0.0*xv_g
  #  if(DIMS==1):
  #      xv_g=x0; yv_g = 0.0*xv_g; zv_g = 0.0*xv_g;
    # the gas particle number is the lattice size: this should be the gas particle number
    Ngas = xv_g.size
    # flatten the vectors (since our ICs should be in vector, not matrix format): just want a
    #  simple list of the x,y,z positions here. Here we multiply the desired box size in
    #xv_g=xv_g.flatten()*Lbox; yv_g=yv_g.flatten()*Lbox; zv_g=zv_g.flatten()*Lbox;
    # set the initial velocity in x/y/z directions (here zero)
    vx_g, vy_g, vz_g = get_velocities(xv_g,yv_g,zv_g,**kargs)
    rho_g, pres_g = get_density_pressure(xv_g,yv_g,zv_g,**kargs)
    #vx_g=0.*xv_g; vy_g=0.*xv_g; vz_g=0.*xv_g;
    # set the initial magnetic field in x/y/z directions (here zero).
    #  these can be overridden (if constant field values are desired) by BiniX/Y/Z in the parameterfile
    bx_g=0.*xv_g; by_g=0.*xv_g; bz_g=0.*xv_g;
    # set the particle masses. Here we set it to be a list the same length, with all the same mass
    #   since their space-density is uniform this gives a uniform density, at the desired value
    #dA = Lbox**2/Ngas
    #dV = dA / Lbox
    mv_g=rho_g * dA
    # set the initial internal energy per unit mass. recall gizmo uses this as the initial 'temperature' variable
    #  this can be overridden with the InitGasTemp variable (which takes an actual temperature)
    uv_g=pres_g/((gamma_eos-1.)*rho_g)
    # set the gas IDs: here a simple integer list
    id_g=np.arange(1,Ngas+1)

    # now we set the properties of the collisionless particles: we will assign these to particle type '3',
    #   but (barring special compile-time flags being set) GIZMO will treat all collisionless particle types
    #   the same. so the setup would be identical for any of the particle types 1,2,3,4,5

    # set the desired number of particles (here to about twice as many as the gas particles, because we feel like it)
# Central star
    #Ngrains = 1
    #xv_d, yv_d, zv_d = 0,0,0
    #vx_d, vy_d, vz_d = 0,0,0
    #mv_d = M
    #id_d = Ngas+1




    # now we get ready to actually write this out
    #  first - open the hdf5 ics file, with the desired filename

    Ngrains = 0
    if kargs['star']:
        Ngrains = 1

    with h5py.File(fname,'w') as file:

        # set particle number of each type into the 'npart' vector
        #  NOTE: this MUST MATCH the actual particle numbers assigned to each type, i.e.
        #   npart = np.array([number_of_PartType0_particles,number_of_PartType1_particles,number_of_PartType2_particles,
        #                     number_of_PartType3_particles,number_of_PartType4_particles,number_of_PartType5_particles])
        #   or else the code simply cannot read the IC file correctly!
        #
        npart = np.array([Ngas,0,0,Ngrains,0,0]) # we have gas and particles we will set for type 3 here, zero for all others

        # now we make the Header - the formatting here is peculiar, for historical (GADGET-compatibility) reasons
        h = file.create_group("Header");
        # here we set all the basic numbers that go into the header
        # (most of these will be written over anyways if it's an IC file; the only thing we actually *need* to be 'correct' is "npart")
        h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
                                             #  if we make a multi-part IC file. with this simple script, we aren't equipped to do that.
        h.attrs['NumPart_Total'] = npart; # npart set as above
        h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
        h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
                                            # we set masses explicitly by-particle this should be zero. that is more flexible anyways, as it
                                            # allows for physics which can change particle masses
        ## all of the parameters below will be overwritten by whatever is set in the run-time parameterfile if
        ##   this file is read in as an IC file, so their values are irrelevant. they are only important if you treat this as a snapshot
        ##   for restarting. Which you shouldn't - it requires many more fields be set. But we still need to set some values for the code to read
        h.attrs['Time'] = 0.0;  # initial time
        h.attrs['Redshift'] = 0.0; # initial redshift
        h.attrs['BoxSize'] = 8.0; # box size
        h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
        h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
        h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
        h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
        h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
        h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
        h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
        h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
        h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
        h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
        h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs
        ## ok, that ends the block of 'useless' parameters

        # Now, the actual data!
        #   These blocks should all be written in the order of their particle type (0,1,2,3,4,5)
        #   If there are no particles of a given type, nothing is needed (no block at all)
        #   PartType0 is 'special' as gas. All other PartTypes take the same, more limited set of information in their ICs

        # start with particle type zero. first (assuming we have any gas particles) create the group
        p = file.create_group("PartType0")
        # now combine the xyz positions into a matrix with the correct format
        q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
        # write it to the 'Coordinates' block
        p.create_dataset("Coordinates",data=q)
        # similarly, combine the xyz velocities into a matrix with the correct format
        q=np.zeros((Ngas,3)); q[:,0]=vx_g; q[:,1]=vy_g; q[:,2]=vz_g;
        # write it to the 'Velocities' block
        p.create_dataset("Velocities",data=q)
        # write particle ids to the ParticleIDs block
        p.create_dataset("ParticleIDs",data=id_g)
        # write particle masses to the Masses block
        p.create_dataset("Masses",data=mv_g)
        # write internal energies to the InternalEnergy block
        p.create_dataset("InternalEnergy",data=uv_g)
        # combine the xyz magnetic fields into a matrix with the correct format
        q=np.zeros((Ngas,3)); q[:,0]=bx_g; q[:,1]=by_g; q[:,2]=bz_g;
        # write magnetic fields to the MagneticField block. note that this is unnecessary if the code is compiled with
        #   MAGNETIC off. however, it is not a problem to have the field there, even if MAGNETIC is off, so you can
        #   always include it with some dummy values and then use the IC for either case
        p.create_dataset("MagneticField",data=q)

        # no PartType1 for this IC
        # no PartType2 for this IC

        # now assign the collisionless particles to PartType3. note that this block looks exactly like
        #   what we had above for the gas. EXCEPT there are no "InternalEnergy" or "MagneticField" fields (for
        #   obvious reasons).

        if kargs['star']:
            p = file.create_group("PartType3")
            q=np.zeros((Ngrains,3));
            p.create_dataset("Coordinates",data=q)
            q=np.zeros((Ngrains,3));
            p.create_dataset("Velocities",data=q)
            p.create_dataset("ParticleIDs",data=Ngas+1)
            p.create_dataset("Masses",data=np.ones((1,)))

        # no PartType4 for this IC
        # no PartType5 for this IC

        # close the HDF5 file, which saves these outputs
    # all done!

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N',type=int,default=150,help='N^2 particles')
    parser.add_argument('-L',type=float,default=8.,help='Box length')
    parser.add_argument('-fname',type=str,default='mykep.hdf5',help='Output file name')
    parser.add_argument('-H',type=float,default=.01,help='Aspect ratio')
    parser.add_argument('-gamma',type=float,default=1.0001,help='EOS gamma')
    parser.add_argument('-mu',type=float,default=0,help='Density power law')
    parser.add_argument('-delta',type=float,default=0,help='c^2 power law')
    parser.add_argument('-G',type=float,default=1,help='Gravitational constant')
    parser.add_argument('-M',type=float,default=1,help='Central Mass')
    parser.add_argument('-sig0',type=float,default=.001,help='Density normalization')
    parser.add_argument('-ri',type=float,default=.05,help='Inner disk radius')
    parser.add_argument('-rc',type=float,default=2.,help='Outer cutoff radius')
    parser.add_argument('-sig_floor',type=float,default=1e-5,help='Density floor')
    parser.add_argument('--ring',action='store_true',help='Use Gaussian ring Exp[-((r-ri)/rc)^2')
    parser.add_argument('--star',action='store_true',help='Use a star instead of an analytic potential')
    parser.add_argument('-soft',type=float,default=.01,help='Potential softening')
    parser.add_argument('-rinner',type=float,default=.2,help='Inner radius')
    parser.add_argument('-router',type=float,default=3,help='Outer radius')
    args = vars(parser.parse_args())

    print(args)

    makeIC_box(**args)


