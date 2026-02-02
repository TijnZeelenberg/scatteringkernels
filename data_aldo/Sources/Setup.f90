MODULE Setup
  USE Precision
  USE Particles
  USE Molprops
  USE Random
  USE Macrovars
  IMPLICIT NONE
  REAL(dp), DIMENSION(nspecies)::molfrac,massfrac,numdens
  REAL(dp), DIMENSION(nspecies,nspecies)::num_coll,coll_freq
  REAL(dp)::Tt0,Tr0,Teq,P0,rho0,numdens0
  REAL(dp)::tstart,deltat,tstop,mean_free_time
  REAL(dp)::deltat_shrink
  REAL(dp)::erot
  REAL(dp)::shear_rate
  REAL(dp):: divergence,volume_ratio
  INTEGER::nsteps,s,nsave,restart
  !
  ! h is the trajectory integration time step. NOT to be confused with
  ! deltat
  !
  REAL(dp)::h                      
  !
CONTAINS
  !
  SUBROUTINE Init
    INTEGER::jp
    !
    OPEN(Unit=1,file='ct_dsmc.inp')
    OPEN(Unit=2,file='Setup.dat')
    OPEN(Unit=3,file='ct_dsmc.dat')
    OPEN(Unit=4,file='muv.dat')
    OPEN(Unit=7,file='pressures.dat')
    OPEN(Unit=8,file='collisions.dat')
    OPEN(Unit=9,file='coll_freqs.dat')
    !
    OPEN(Unit=14,file='Diagnostica_N2_N2.dat')
    OPEN(Unit=15,file='Diagnostica_N2_O2.dat')
    OPEN(Unit=16,file='Diagnostica_O2_O2.dat')
    OPEN(Unit=17,file='Restart.dat')
    !
    READ(1,*) Tt0
    READ(1,*) Tr0
    READ(1,*) P0
    READ(1,*) molfrac(2) ! frazione molare dell'ossigeno
    READ(1,*) npart
    READ(1,*) seme
    READ(1,*) nsteps
    READ(1,*) nsave
    READ(1,*) restart
    READ(1,*) divergence
    READ(1,*) volume_ratio
    READ(1,*) shear_rate
    READ(1,*) h
    READ(1,*) deltat_shrink
    !
    Teq=(3.0D0*Tt0+2.0D0*Tr0)/5.0D0
    !
    numdens0=P0/(kb*Teq)
    !
    Vol_box=Npart/numdens0
    lx=Vol_box**(1.0D0/3.0D0)
    ly=lx
    lz=ly
    xmin=-lx/20.D0
    xmax=-xmin
    ymin=-ly/20.D0
    ymax=-ymin
    zmin=-lz/20.D0
    zmax=-zmin
    !
    tstart=0.0D0
    !
    ! np sono pari
    !
    np(2)=INT(Npart*molfrac(2))
    IF(MOD(np(2),2).EQ.0) THEN
       np(1)=Npart-np(2)
    ELSE
       np(2)=np(2)+1
       np(1)=Npart-np(2)
    END IF
    numdens=np/Vol_box
    !
    ALLOCATE(x(npart))
    ALLOCATE(y(npart))
    ALLOCATE(z(npart))
    ALLOCATE(vx(npart))
    ALLOCATE(vy(npart))
    ALLOCATE(vz(npart))
    ALLOCATE(omegax(npart))
    ALLOCATE(omegay(npart))
    ALLOCATE(omegaz(npart))
    ALLOCATE(species(npart))
    ALLOCATE(mass(npart))
    !
    DO jp=1,np(1)-1,2
       x(jp)=-xmin+lx*rf(seme)
       y(jp)=-ymin+ly*rf(seme)
       z(jp)=-zmin+lz*rf(seme)
       x(jp+1)=-x(jp)
       y(jp+1)=-y(jp)
       z(jp+1)=-z(jp)
       species(jp)=1
       species(jp+1)=species(jp)
       mass(jp)=molmass(1)
       mass(jp+1)=mass(jp)
       CALL Maxwell(vx(jp),vy(jp),vz(jp))
       vx(jp)=vx(jp)*SQRT(Rgas(1)*Tt0)
       vy(jp)=vy(jp)*SQRT(Rgas(1)*Tt0)
       vz(jp)=vz(jp)*SQRT(Rgas(1)*Tt0)
       vx(jp+1)=-vx(jp)
       vy(jp+1)=-vy(jp)
       vz(jp+1)=-vz(jp)
       !
       CALL Genk(omegax(jp),omegay(jp),omegaz(jp))
       erot=-kb*Tr0*LOG(1.0D0-rf(seme))
       omegax(jp)= omegax(jp)*SQRT(2.0D0*erot/momin(1))
       omegay(jp)= omegay(jp)*SQRT(2.0D0*erot/momin(1))
       omegaz(jp)= omegaz(jp)*SQRT(2.0D0*erot/momin(1))
       omegax(jp+1)=-omegax(jp)
       omegay(jp+1)=-omegay(jp)
       omegaz(jp+1)=-omegaz(jp)
       !
    END DO
    !
    DO jp=np(1)+1,Npart-1,2
       x(jp)=-xmin+lx*rf(seme)
       y(jp)=-ymin+ly*rf(seme)
       z(jp)=-zmin+lz*rf(seme)
       x(jp+1)=-x(jp)
       y(jp+1)=-y(jp)
       z(jp+1)=-z(jp)
       species(jp)=2
       species(jp+1)=species(jp)
       mass(jp)=molmass(2)
       mass(jp+1)=mass(jp)
       CALL Maxwell(vx(jp),vy(jp),vz(jp))
       vx(jp)=vx(jp)*SQRT(Rgas(2)*Tt0)
       vy(jp)=vy(jp)*SQRT(Rgas(2)*Tt0)
       vz(jp)=vz(jp)*SQRT(Rgas(2)*Tt0)
       vx(jp+1)=-vx(jp)
       vy(jp+1)=-vy(jp)
       vz(jp+1)=-vz(jp)
       !
       CALL Genk(omegax(jp),omegay(jp),omegaz(jp))
       erot=-kb*Tr0*LOG(1.0D0-rf(seme))
       omegax(jp)= omegax(jp)*SQRT(2.0D0*erot/momin(2))
       omegay(jp)= omegay(jp)*SQRT(2.0D0*erot/momin(2))
       omegaz(jp)= omegaz(jp)*SQRT(2.0D0*erot/momin(2))
       omegax(jp+1)=-omegax(jp)
       omegay(jp+1)=-omegay(jp)
       omegaz(jp+1)=-omegaz(jp)
    END DO
    !
    CALL Inst_averages
    !
    vx(1:np(1))=vx(1:np(1))*SQRT(Tt0/temp_trasl(1))
    vy(1:np(1))=vy(1:np(1))*SQRT(Tt0/temp_trasl(1))
    vz(1:np(1))=vz(1:np(1))*SQRT(Tt0/temp_trasl(1))
    vx(1+np(1):Npart)=vx(1+np(1):Npart)*SQRT(Tt0/temp_trasl(2))
    vy(1+np(1):Npart)=vy(1+np(1):Npart)*SQRT(Tt0/temp_trasl(2))
    vz(1+np(1):Npart)=vz(1+np(1):Npart)*SQRT(Tt0/temp_trasl(2))
    !
    omegax(1:np(1))=omegax(1:np(1))*SQRT(Tr0/temp_rot(1))
    omegay(1:np(1))=omegay(1:np(1))*SQRT(Tr0/temp_rot(1))
    omegaz(1:np(1))=omegaz(1:np(1))*SQRT(Tr0/temp_rot(1))
    omegax(1+np(1):Npart)=omegax(1+np(1):Npart)*SQRT(Tr0/temp_rot(2))
    omegay(1+np(1):Npart)=omegay(1+np(1):Npart)*SQRT(Tr0/temp_rot(2))
    omegaz(1+np(1):Npart)=omegaz(1+np(1):Npart)*SQRT(Tr0/temp_rot(2))
    !
    CALL Inst_averages
    CALL Total_Energies
    Initial_Energy=Energy
    !
    num_coll(1,1)=2.0D0*SQRT(Rgas(1)*pi*Tt0)*diamol(1,1)**2*numdens(1)**2
    num_coll(2,2)=2.0D0*SQRT(Rgas(2)*pi*Tt0)*diamol(2,2)**2*numdens(2)**2
    num_coll(1,2)=2.0D0**(3.0D0/2.0D0)*SQRT(pi*kb*Tt0/mr)*diamol(1,2)**2*numdens(1)*numdens(2)
    num_coll(2,1)=num_coll(1,2)
    !
    coll_freq(1,1)=2.0D0*num_coll(1,1)/numdens(1)
    coll_freq(1,2)=num_coll(1,2)/numdens(1)
    coll_freq(2,1)=num_coll(2,1)/numdens(2)
    coll_freq(2,2)=2.0D0*num_coll(2,2)/numdens(2)
    !
    deltat=0.1/MAXVAL(coll_freq)
    deltat=deltat*deltat_shrink
    nsteps=nsteps/deltat_shrink
    !
    IF(divergence.NE.0.0D0) THEN
       tstop=LOG(Volume_ratio)/divergence
       nsteps=INT(tstop/deltat)+1
!!$       nsteps=12000
!!$       deltat=tstop/nsteps
    END IF
    IF(ABS(divergence)*deltat/3.0D0.GT.1.0D0)THEN
       WRITE(*,*) 'Delta_t or divergence too large !!!'
       WRITE(*,*) 'Modify computational parameters !!!'
       STOP
    END IF
    !
    ! Stampa i dati iniziali
    !
    WRITE(2,*)' Pressure [Pa] = ', P0
    WRITE(2,*)' Initial Translational Temperature [K] = ', Tt0
    WRITE(2,*)' Initial Rotational Temperature [K] = ', Tr0
    WRITE(2,*)' Number density [1/m^3] = ',numdens0
    WRITE(2,*)' Oxygen molar fraction = ', molfrac(2)
    WRITE(2,*)' ----------------------------------------- '
    WRITE(2,*)' Radius of protection sphere = ', bmax
    WRITE(2,*)' ----------------------------------------- '
    WRITE(2,*)' Number of simulation particles = ', Npart
    WRITE(2,*)' Number of N_2 simulation particles = ', np(1)
    WRITE(2,*)' Number of O_2 simulation particles = ', np(2)
    WRITE(2,*)' ----------------------------------------- '
    WRITE(2,*)' Volume of simulation box [m^3] = ', Vol_box
    WRITE(2,*)' Side of simulation box [m] = ', lx
    WRITE(2,*)' ----------------------------------------- '
    WRITE(2,*)' Frequency of N_2-N_2 collisions [1/s]  ', coll_freq(1,1)
    WRITE(2,*)' Frequency of N_2-O_2 collisions [1/s]  ', coll_freq(1,2)
    WRITE(2,*)' Frequency of O_2-N_2 collisions [1/s]  ', coll_freq(2,1)
    WRITE(2,*)' Frequency of O_2-O_2 collisions [1/s]  ', coll_freq(2,2)
    WRITE(2,*)' ----------------------------------------- '
    WRITE(2,*)' Time step [s]  ', deltat
    WRITE(2,*)' ----------------------------------------- '
    WRITE(2,*)' Number of N_2-N_2 collisions in the box, per time step  ', num_coll(1,1)*Vol_box*deltat
    WRITE(2,*)' Number of O_2-O_2 collisions in the box, per time step  ', num_coll(2,2)*Vol_box*deltat
    WRITE(2,*)' Number of N_2-O_2 collisions in the box, per time step  ', num_coll(1,2)*Vol_box*deltat
    WRITE(2,*)' ----------------------------------------- '
    WRITE(2,*)' Flow divergence [1/s] ', divergence
    WRITE(2,*)' Volume ratio ', volume_ratio
    WRITE(2,*)' Shear rate ', shear_rate
    WRITE(2,*)' Simulation duration [s] ', tstop
    WRITE(2,*)' Number of time steps ', nsteps
    FLUSH(2)
!!$100 FORMAT(12(E12.5,2X))
200 FORMAT(I1,9(E13.6,2X))
  END SUBROUTINE Init
END MODULE Setup
