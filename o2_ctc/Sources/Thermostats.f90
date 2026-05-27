MODULE Thermostats
  USE Precision
  USE Particles
  USE Setup
  USE Macrovars
CONTAINS
  SUBROUTINE Gauss
    CALL Inst_averages
    vx(1:np(1))=vx(1:np(1))*SQRT(Tt0/temp_trasl(1))
    vy(1:np(1))=vy(1:np(1))*SQRT(Tt0/temp_trasl(1))
    vz(1:np(1))=vz(1:np(1))*SQRT(Tt0/temp_trasl(1))
    vx(1+np(1):Npart)=vx(1+np(1):Npart)*SQRT(Tt0/temp_trasl(2))
    vy(1+np(1):Npart)=vy(1+np(1):Npart)*SQRT(Tt0/temp_trasl(2))
    vz(1+np(1):Npart)=vz(1+np(1):Npart)*SQRT(Tt0/temp_trasl(2))
  END SUBROUTINE Gauss
  !
  SUBROUTINE Vel_Rescale
    vx(1:Npart)=vx(1:Npart)*(1.0D0-divergence*deltat/3.0D0)
    vy(1:Npart)=vy(1:Npart)*(1.0D0-divergence*deltat/3.0D0)
    vz(1:Npart)=vz(1:Npart)*(1.0D0-divergence*deltat/3.0D0) 
  END SUBROUTINE Vel_Rescale
  !
  SUBROUTINE Shear_vel
    INTEGER:: jp
    DO jp=1,npart
       vy(jp)=vy(jp)-shear_rate*vx(jp)*deltat
    END DO
  END SUBROUTINE Shear_vel
  !
  SUBROUTINE Energy_reset
    REAL(dp)::f
    !
    CALL Total_Energies
    !
    f=SQRT(Initial_energy/Energy)
    !
    vx(1:Npart)=vx(1:Npart)*f
    vy(1:Npart)=vy(1:Npart)*f
    vz(1:Npart)=vz(1:Npart)*f
    !
    omegax(1:Npart)=omegax(1:Npart)*f
    omegay(1:Npart)=omegay(1:Npart)*f
    omegaz(1:Npart)=omegaz(1:Npart)*f
    !
  END SUBROUTINE Energy_reset
END MODULE Thermostats
