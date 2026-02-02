MODULE CT_Collider
USE Precision
  USE Particles
  USE Molprops
  USE Random
  USE Setup
  USE Coll_N2_N2
  USE Coll_N2_O2
  USE Coll_O2_O2
  IMPLICIT NONE
  REAL(dp), DIMENSION(nspecies,nspecies)::avncoll
  !
CONTAINS
  SUBROUTINE Collision(v1,omega1,v2,omega2,s1,s2)
    IMPLICIT NONE
    INTEGER::s1,s2
    REAL(dp), DIMENSION(3)::v1,v2,vcm,vr
    REAL(dp), DIMENSION(3)::vr_save
    REAL(dp), DIMENSION(3)::omega1,omega2
    REAL(dp), DIMENSION(3)::omega1_save,omega2_save
    REAL(dp)::vref,omegaref
    !---------------------------------------------------------------------------------------------
    ! Variabili per la diagnostica delle collisioni
    !
    REAL(dp):: etr_in,etr_out,erot1_in,erot1_out,erot2_in,erot2_out,e_in,e_out,errore
    !
    !---------------------------------------------------------------------------------------------
    vr=v2-v1
    vcm=(molmass(s1)*v1+molmass(s2)*v2)/(molmass(s1)+molmass(s2))
    !
    ! Collisioni N2-N2
    !
    IF(s1.EQ.1.AND.s2.EQ.1) THEN
       !-----------------------------------------------------------------------------------
!!$       WRITE(14,*)'============================ Inizio ========================================='
!!$       WRITE(14,*)'molmass(1)= ',molmass(1), 'momin(1)= ',momin(1), 'eps_pot(1)= ',eps_pot(1), 'Rm(1)= ',Rm(1)
       etr_in=0.5D0*molmass(1)*molmass(1)/(molmass(1)+molmass(1))*DOT_PRODUCT(vr,vr)
       erot1_in=0.5D0*momin(1)*DOT_PRODUCT(omega1,omega1)
       erot2_in=0.5D0*momin(1)*DOT_PRODUCT(omega2,omega2)
       e_in=etr_in+erot1_in+erot2_in
!!$       WRITE(14,*)'e_in= ',e_in
       !-----------------------------------------------------------------------------------
       !
       ! Normalizza velocita lineari ed angolari
       !
       vref=SQRT(eps_pot(1)/molmass(1))
       omegaref=vref/Rm(1)
       vr=vr/vref
       vr_save=vr
       omega1=omega1/omegaref;omega2=omega2/omegaref;
       omega1_save=omega1;omega2_save=omega2;
!!$       WRITE(14,*)'vref= ',vref,'omegaref= ',omegaref
       !   
1111   CALL CT_N2_N2(vr,omega1,omega2)
       !
       ! Riscala velocita lineari ed angolari in unita fisiche 
       ! 
       vr=vr*vref
       omega1=omega1*omegaref;omega2=omega2*omegaref;
       !-----------------------------------------------------------------------------------
       etr_out=0.5D0*molmass(1)*molmass(1)/(molmass(1)+molmass(1))*DOT_PRODUCT(vr,vr)
       erot1_out=0.5D0*momin(1)*DOT_PRODUCT(omega1,omega1)
       erot2_out=0.5D0*momin(1)*DOT_PRODUCT(omega2,omega2)
       e_out=etr_out+erot1_out+erot2_out
       errore=2.0D0*ABS(e_out-e_in)/(e_out+e_in)
!!$       WRITE(14,*)'e_out= ',e_out,' errore= ',errore
!!$       WRITE(14,*)'============================== Fine ========================================='
       IF(errore.GT.0.01D0) THEN
          WRITE(14,*)'N2-N2'
          WRITE(14,*)'e_out= ',e_out,' errore= ',errore
          vr=vr_save
          omega1=omega1_save;omega2=omega2_save;
!!$          STOP
          GO TO 1111
       END IF
    END IF
    !-----------------------------------------------------------------------------------
    !
    ! Collisioni N2-O2
    !
    IF(s1.EQ.1.AND.s2.EQ.2) THEN
       !-----------------------------------------------------------------------------------
       !WRITE(*,*)'N2-O2'
       etr_in=0.5D0*molmass(1)*molmass(2)/(molmass(1)+molmass(2))*DOT_PRODUCT(vr,vr)
       erot1_in=0.5D0*momin(1)*DOT_PRODUCT(omega1,omega1)
       erot2_in=0.5D0*momin(2)*DOT_PRODUCT(omega2,omega2)
       e_in=etr_in+erot1_in+erot2_in
       !-----------------------------------------------------------------------------------
       !
       ! Normalizza velocita lineari ed angolari
       !
       vref=SQRT(eps_pot(3)/molmass(1))
       omegaref=vref/Rm(3)
       vr=vr/vref
       vr_save=vr
       omega1=omega1/omegaref;omega2=omega2/omegaref;
       omega1_save=omega1;omega2_save=omega2;
       !   
2222   CALL CT_N2_O2(vr,omega1,omega2)
       !
       ! Riscala velocita lineari ed angolari in unita fisiche 
       ! 
       vr=vr*vref
       omega1=omega1*omegaref;omega2=omega2*omegaref;
       !-----------------------------------------------------------------------------------
       etr_out=0.5D0*molmass(1)*molmass(2)/(molmass(1)+molmass(2))*DOT_PRODUCT(vr,vr)
       erot1_out=0.5D0*momin(1)*DOT_PRODUCT(omega1,omega1)
       erot2_out=0.5D0*momin(2)*DOT_PRODUCT(omega2,omega2)
       e_out=etr_out+erot1_out+erot2_out
       errore=2.0D0*ABS(e_out-e_in)/(e_out+e_in)
       IF(errore.GT.0.1D0) THEN
           WRITE(15,*)'N2-O2'
           WRITE(15,*)' e_in= ',e_in,' e_out= ',e_out,' errore= ',errore
           vr=vr_save
           omega1=omega1_save;omega2=omega2_save;
          GO TO 2222
       END IF
    END IF
    !
    ! Collisioni O2-O2
    !
    IF(s1.EQ.2.AND.s2.EQ.2) THEN
       !-----------------------------------------------------------------------------------
       !WRITE(*,*)'O2-O2'
       etr_in=0.5D0*molmass(2)*molmass(2)/(molmass(2)+molmass(2))*DOT_PRODUCT(vr,vr)
       erot1_in=0.5D0*momin(2)*DOT_PRODUCT(omega1,omega1)
       erot2_in=0.5D0*momin(2)*DOT_PRODUCT(omega2,omega2)
       e_in=etr_in+erot1_in+erot2_in
       !-----------------------------------------------------------------------------------
       !
       ! Normalizza velocita lineari ed angolari
       !
       vref=SQRT(eps_pot(2)/molmass(2))
       omegaref=vref/Rm(2)
       vr=vr/vref
       vr_save=vr
       omega1=omega1/omegaref;omega2=omega2/omegaref;
       omega1_save=omega1;omega2_save=omega2;
       !
3333   CALL CT_O2_O2(vr,omega1,omega2)
       !
       ! Riscala velocita lineari ed angolari in unita fisiche 
       ! 
       vr=vr*vref
       omega1=omega1*omegaref;omega2=omega2*omegaref;
       !-----------------------------------------------------------------------------------
       etr_out=0.5D0*molmass(2)*molmass(2)/(molmass(2)+molmass(2))*DOT_PRODUCT(vr,vr)
       erot1_out=0.5D0*momin(2)*DOT_PRODUCT(omega1,omega1)
       erot2_out=0.5D0*momin(2)*DOT_PRODUCT(omega2,omega2)
       e_out=etr_out+erot1_out+erot2_out
       errore=2.0D0*ABS(e_out-e_in)/(e_out+e_in)
 !!$      WRITE(*,*)' e_in= ',e_in,' e_out= ',e_out,' errore= ',errore
       IF(errore.GT.0.1D0) THEN
           WRITE(16,*)'O2-O2'
           WRITE(16,*)' e_in= ',e_in,' e_out= ',e_out,' errore= ',errore
           vr=vr_save
           omega1=omega1_save;omega2=omega2_save;
          GO TO 3333
       END IF
    END IF
    !
    v1=vcm-molmass(s2)*vr/(molmass(s1)+molmass(s2))
    v2=vcm+molmass(s1)*vr/(molmass(s1)+molmass(s2))
    !
  END SUBROUTINE Collision
  !
  SUBROUTINE Hom_relax
    INTEGER::jp1,jp2
    REAL(dp), DIMENSION(nspecies)::vxmin,vxmax
    REAL(dp), DIMENSION(nspecies)::vymin,vymax
    REAL(dp), DIMENSION(nspecies)::vzmin,vzmax
    REAL(dp), DIMENSION(nspecies)::numdens
    REAL(dp), DIMENSION(nspecies,nspecies)::vrmax
    REAL(dp), DIMENSION(nspecies,nspecies)::ncoll
    REAL(dp), DIMENSION(3)::v1,v2,omega1,omega2
    REAL(dp)::fncoll,vr
    INTEGER::nncoll,jcoll,nreal
    !
    !-------------------------- Diagnostica-------------------------------
    !
    REAL(dp)::e_in,e_out
    !
    !---------------------------------------------------------------------
    !---------------------------------------------------------------------------------------------
    !---------------------------------------------------------------------------------------------
    vxmin(1)=MINVAL(vx(1:np(1)))
    vxmax(1)=MAXVAL(vx(1:np(1)))
    vymin(1)=MINVAL(vy(1:np(1)))
    vymax(1)=MAXVAL(vy(1:np(1)))
    vzmin(1)=MINVAL(vz(1:np(1)))
    vzmax(1)=MAXVAL(vz(1:np(1)))
    !
    vxmin(2)=MINVAL(vx(1+np(1):npart))
    vxmax(2)=MAXVAL(vx(1+np(1):npart))
    vymin(2)=MINVAL(vy(1+np(1):npart))
    vymax(2)=MAXVAL(vy(1+np(1):npart))
    vzmin(2)=MINVAL(vz(1+np(1):npart))
    vzmax(2)=MAXVAL(vz(1+np(1):npart))
    !
    vrmax(1,1)=SQRT((vxmax(1)-vxmin(1))**2+(vymax(1)-vymin(1))**2+(vzmax(1)-vzmin(1))**2)
    vrmax(2,2)=SQRT((vxmax(2)-vxmin(2))**2+(vymax(2)-vymin(2))**2+(vzmax(2)-vzmin(2))**2)
    vrmax(1,2)=SQRT((vxmax(2)-vxmin(1))**2+(vymax(2)-vymin(1))**2+(vzmax(2)-vzmin(1))**2)
    vrmax(2,1)=vrmax(1,2)
    !
    numdens=np/vol_box
    !
    ncoll(1,1)=1.0D0/2.0D0*vrmax(1,1)*xsect(1,1)*numdens(1)*np(1)*deltat
    ncoll(2,2)=1.0D0/2.0D0*vrmax(2,2)*xsect(2,2)*numdens(2)*np(2)*deltat
    ncoll(1,2)=vrmax(1,2)*xsect(1,2)*numdens(2)*np(1)*deltat
    ncoll(2,1)=ncoll(1,2)
    !
    ! Collisioni N2-N2
    !
    nncoll=INT(ncoll(1,1))
    fncoll=ncoll(1,1)-nncoll
    IF(rf(seme).LT.fncoll)nncoll=nncoll+1
    !
    DO jcoll=1,nncoll
       !
       jp1=1+INT(np(1)*rf(seme))
       !
       v1(1)=vx(jp1)
       v1(2)=vy(jp1)
       v1(3)=vz(jp1)
       omega1(1)=omegax(jp1)
       omega1(2)=omegay(jp1)
       omega1(3)=omegaz(jp1)
       !
       jp2=1+INT(np(1)*rf(seme))
       !
       v2(1)=vx(jp2)
       v2(2)=vy(jp2)
       v2(3)=vz(jp2)
       omega2(1)=omegax(jp2)
       omega2(2)=omegay(jp2)
       omega2(3)=omegaz(jp2)
       !
       vr=SQRT(DOT_PRODUCT(v2-v1,v2-v1))
       !
       IF(rf(seme).LT.vr/vrmax(1,1)) THEN
          e_in=0.5D0*molmass(1)*DOT_PRODUCT(v1,v1)
          e_in=e_in+0.5D0*molmass(1)*DOT_PRODUCT(v2,v2)
          e_in=e_in+0.5D0*momin(1)*DOT_PRODUCT(omega1,omega1)
          e_in=e_in+0.5D0*momin(1)*DOT_PRODUCT(omega2,omega2)
          !
          CALL Collision(v1,omega1,v2,omega2,1,1)
          !
          e_out=0.5D0*molmass(1)*DOT_PRODUCT(v1,v1)
          e_out=e_out+0.5D0*molmass(1)*DOT_PRODUCT(v2,v2)
          e_out=e_out+0.5D0*momin(1)*DOT_PRODUCT(omega1,omega1)
          e_out=e_out+0.5D0*momin(1)*DOT_PRODUCT(omega2,omega2)
 !!$         WRITE(*,*)'N2-N2 ',e_in,e_out,2.0D0*(e_out-e_in)/(e_out+e_in)
          vx(jp1)=v1(1)
          vy(jp1)=v1(2)
          vz(jp1)=v1(3)
          omegax(jp1)=omega1(1)
          omegay(jp1)=omega1(2)
          omegaz(jp1)=omega1(3)
          !
          vx(jp2)=v2(1)
          vy(jp2)=v2(2)
          vz(jp2)=v2(3)
          omegax(jp2)=omega2(1)
          omegay(jp2)=omega2(2)
          omegaz(jp2)=omega2(3)
          !
          avncoll(1,1)=avncoll(1,1)+1.0D0
          !
       END IF
    END DO
    !
    ! Collisioni N2-O2
    !
    nncoll=INT(ncoll(1,2))
    fncoll=ncoll(1,2)-nncoll
    IF(rf(seme).LT.fncoll)nncoll=nncoll+1
    !
    nreal=0
    DO jcoll=1,nncoll
       !
       jp1=1+INT(np(1)*rf(seme))
       !
       v1(1)=vx(jp1)
       v1(2)=vy(jp1)
       v1(3)=vz(jp1)
       omega1(1)=omegax(jp1)
       omega1(2)=omegay(jp1)
       omega1(3)=omegaz(jp1)
       !
       jp2=np(1)+1+INT(np(2)*rf(seme))
       !
       v2(1)=vx(jp2)
       v2(2)=vy(jp2)
       v2(3)=vz(jp2)
       omega2(1)=omegax(jp2)
       omega2(2)=omegay(jp2)
       omega2(3)=omegaz(jp2)
       !
       vr=SQRT(DOT_PRODUCT(v2-v1,v2-v1))
       !
       IF(rf(seme).LT.vr/vrmax(1,2)) THEN
          nreal=nreal+1
          !
          CALL Collision(v1,omega1,v2,omega2,1,2)
          !
!!$          ! Per il calcolo del coefficiente di diffusione dell'ossigeno in Azoto
!!$          ! viene cancellata la variazione di velocità sull'Azoto
!!$          vx(jp1)=v1(1)
!!$          vy(jp1)=v1(2)
!!$          vz(jp1)=v1(3)
!!$          omegax(jp1)=omega1(1)
!!$          omegay(jp1)=omega1(2)
!!$          omegaz(jp1)=omega1(3)
          !
          vx(jp2)=v2(1)
          vy(jp2)=v2(2)
          vz(jp2)=v2(3)
          omegax(jp2)=omega2(1)
          omegay(jp2)=omega2(2)
          omegaz(jp2)=omega2(3)
          !
          avncoll(1,2)=avncoll(1,2)+1.0D0
          avncoll(2,1)=avncoll(2,1)+1.0D0
       END IF
    END DO
    !WRITE(*,*) nncoll,nreal
    !
    ! Collisioni O2-O2
    !
    nncoll=INT(ncoll(2,2))
    fncoll=ncoll(2,2)-nncoll
    IF(rf(seme).LT.fncoll)nncoll=nncoll+1
    !
    ! Soppresse le collisioni O2-O2
    !
    nncoll=0
    DO jcoll=1,nncoll
       !
       jp1=np(1)+1+INT(np(2)*rf(seme))
       !
       v1(1)=vx(jp1)
       v1(2)=vy(jp1)
       v1(3)=vz(jp1)
       omega1(1)=omegax(jp1)
       omega1(2)=omegay(jp1)
       omega1(3)=omegaz(jp1)
       !
       jp2=np(1)+1+INT(np(2)*rf(seme))
       !
       v2(1)=vx(jp2)
       v2(2)=vy(jp2)
       v2(3)=vz(jp2)
       omega2(1)=omegax(jp2)
       omega2(2)=omegay(jp2)
       omega2(3)=omegaz(jp2)
       !
       vr=SQRT(DOT_PRODUCT(v2-v1,v2-v1))
       !
       IF(rf(seme).LT.vr/vrmax(2,2)) THEN
          !
          CALL Collision(v1,omega1,v2,omega2,2,2)
          !
          vx(jp1)=v1(1)
          vy(jp1)=v1(2)
          vz(jp1)=v1(3)
          omegax(jp1)=omega1(1)
          omegay(jp1)=omega1(2)
          omegaz(jp1)=omega1(3)
          !
          vx(jp2)=v2(1)
          vy(jp2)=v2(2)
          vz(jp2)=v2(3)
          omegax(jp2)=omega2(1)
          omegay(jp2)=omega2(2)
          omegaz(jp2)=omega2(3)
          !
          avncoll(2,2)=avncoll(2,2)+1.0D0
          !
       END IF
    END DO
  END SUBROUTINE Hom_relax

END MODULE CT_Collider

