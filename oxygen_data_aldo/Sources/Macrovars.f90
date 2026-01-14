MODULE Macrovars
  USE Precision
  USE Particles
  USE Molprops
  IMPLICIT NONE
  REAL(dp)::vol_box,lx,ly,lz,xmin,xmax,ymin,ymax,zmin,zmax
  REAL(dp), DIMENSION(nspecies)::ndens,rho,temp_trasl,temp_rot,ux,uy,uz
  REAL(dp)::rho_mix,umix_x,umix_y,umix_z
  REAL(dp), DIMENSION(nspecies)::Pxx,Pxy,Pxz,Pyy,Pyz,Pzz
  REAL(dp)::Pmix_xx,Pmix_xy,Pmix_xz,Pmix_yy,Pmix_yz,Pmix_zz
  REAL(dp)::Energy,Initial_Energy,Translational_Energy,Rotational_Energy
  !
CONTAINS
  SUBROUTINE Inst_averages
    INTEGER::jp,s
    !
    ndens=0.0D0
    temp_trasl=0.0D0
    temp_rot=0.0D0
    ux=0.0D0
    uy=0.0D0
    uz=0.0D0
    Pxx=0.0D0;Pxy=0.0D0;Pxz=0.0D0;Pyy=0.0D0;Pyz=0.0D0;Pzz=0.0D0
    !
    DO jp=1,Npart
       s=species(jp)
       ndens(s)=ndens(s)+1.0D0
       ux(s)=ux(s)+vx(jp)
       uy(s)=uy(s)+vy(jp)
       uz(s)=uz(s)+vz(jp)
       temp_trasl(s)=temp_trasl(s)+vx(jp)**2+vy(jp)**2+vz(jp)**2
       temp_rot(s)=temp_rot(s)+omegax(jp)**2+omegay(jp)**2+omegaz(jp)**2
       Pxx(s)=Pxx(s)+vx(jp)*vx(jp)
       Pxy(s)=Pxy(s)+vx(jp)*vy(jp)
       Pxz(s)=Pxz(s)+vx(jp)*vz(jp)
       Pyy(s)=Pyy(s)+vy(jp)*vy(jp)
       Pyz(s)=Pyz(s)+vy(jp)*vz(jp)
       Pzz(s)=Pzz(s)+vz(jp)*vz(jp)
    END DO
    !
    DO s=1,nspecies
       rho(s)=molmass(s)*ndens(s)/vol_box
       ux(s)=ux(s)/ndens(s)
       uy(s)=uy(s)/ndens(s)
       uz(s)=uz(s)/ndens(s)
       temp_trasl(s)=temp_trasl(s)/ndens(s)-ux(s)**2-uy(s)**2-uz(s)**2
       temp_trasl(s)=temp_trasl(s)/(3.0D0*Rgas(s))
       temp_rot(s)=temp_rot(s)/ndens(s)
       temp_rot(s)=0.5D0*momin(s)*temp_rot(s)/kb
       Pxx(s)=Pxx(s)/ndens(s)
       Pxy(s)=Pxy(s)/ndens(s)
       Pxz(s)=Pxz(s)/ndens(s)
       Pyy(s)=Pyy(s)/ndens(s)
       Pyz(s)=Pyz(s)/ndens(s)
       Pzz(s)=Pzz(s)/ndens(s)
    END DO
    !
    rho_mix=0.0D0
    umix_x=0.0D0;umix_y=0.0D0;umix_z=0.0D0
    Pmix_xx=0.0D0;Pmix_xy=0.0D0;Pmix_xz=0.0D0;Pmix_yy=0.0D0;Pmix_yz=0.0D0;Pmix_zz=0.0D0
    !
    DO s=1,nspecies
       rho_mix=rho_mix+rho(s)
       umix_x= umix_x+rho(s)*ux(s)
       umix_y= umix_y+rho(s)*uy(s)
       umix_z= umix_z+rho(s)*uz(s)
       Pmix_xx=Pmix_xx+Pxx(s)*rho(s)
       Pmix_xy=Pmix_xy+Pxy(s)*rho(s)
       Pmix_xz=Pmix_xz+Pxz(s)*rho(s)
       Pmix_yy=Pmix_yy+Pyy(s)*rho(s)
       Pmix_yz=Pmix_yz+Pyz(s)*rho(s)
       Pmix_zz=Pmix_zz+Pzz(s)*rho(s)
    END DO
    !
    umix_x= umix_x/rho_mix
    umix_y= umix_y/rho_mix
    umix_z= umix_z/rho_mix
    !
    Pmix_xx=Pmix_xx-rho_mix*umix_x*umix_x
    Pmix_xy=Pmix_xy-rho_mix*umix_x*umix_y
    Pmix_xz=Pmix_xz-rho_mix*umix_x*umix_z
    Pmix_yy=Pmix_yy-rho_mix*umix_y*umix_y
    Pmix_yz=Pmix_yz-rho_mix*umix_y*umix_z
    Pmix_zz=Pmix_zz-rho_mix*umix_z*umix_z
    !
  END SUBROUTINE Inst_averages
  !
  SUBROUTINE Total_Energies
    INTEGER::jp,s
    !
    Energy=0.0D0;Translational_Energy=0.0D0;Rotational_Energy=0.0D0
    !
    DO jp=1,Npart
       s=species(jp)
       Translational_Energy=Translational_Energy+0.5*molmass(s)*(vx(jp)**2+vy(jp)**2+vz(jp)**2)
       Rotational_Energy=Rotational_Energy+0.5*momin(s)*(omegax(jp)**2+omegay(jp)**2+omegaz(jp)**2)
    END DO
    Energy=Translational_Energy+Rotational_Energy
  END SUBROUTINE Total_Energies
END MODULE macrovars
