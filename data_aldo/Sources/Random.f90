MODULE Random
USE Precision
  INTEGER :: Seme
  REAL(dp), PARAMETER :: Pi=4.0D0*ATAN(1.0D0),Pi2=2.0D0*Pi
CONTAINS
!  
  FUNCTION rf(idum)
    INTEGER, PARAMETER :: MBIG=1000000000,MSEED=161803398,MZ=0
    REAL(dp), PARAMETER :: FAC=1.d0/MBIG
    INTEGER :: i,iff,ii,inext,inextp,k
    INTEGER :: mj,mk,ma(55)
    INTEGER :: idum
    REAL(dp) :: rf
!
    SAVE iff,inext,inextp,ma
    DATA iff /0/
    
      IF(idum.LT.0.OR.iff.EQ.0)THEN
        iff=1
        mj=MSEED-iabs(idum)
        mj=MOD(mj,MBIG)
        ma(55)=mj
        mk=1
        DO i=1,54
           ii=MOD(21*i,55)
           ma(ii)=mk
           mk=mj-mk
           IF(mk.LT.MZ)mk=mk+MBIG
           mj=ma(ii)
        END DO
        DO k=1,4
          DO i=1,55
            ma(i)=ma(i)-ma(1+MOD(i+30,55))
            IF(ma(i).LT.MZ)ma(i)=ma(i)+MBIG
         END DO
      END DO
        inext=0
        inextp=31
        idum=1
      ENDIF
      inext=inext+1
      IF(inext.EQ.56)inext=1
      inextp=inextp+1
      IF(inextp.EQ.56)inextp=1
      mj=ma(inext)-ma(inextp)
      IF(mj.LT.MZ)mj=mj+MBIG
      ma(inext)=mj
      rf=mj*FAC
    END FUNCTION rf
    !
    SUBROUTINE Maxwell(vx,vy,vz)                  
      IMPLICIT NONE
      REAL(dp)::r,ro,teta,vx,vy,vz
!                                                                               
      r=1.d0-rf(seme)                                                           
      ro=SQRT(-2.d0*LOG(r))                                                   
      r=rf(seme)                                                                
      teta=pi2*r                                                                
      vy=ro*COS(teta)                                                          
      vz=ro*SIN(teta)                                                          
      r=1.d0-rf(seme)                                                           
      ro=SQRT(-2.d0*LOG(r))                                                   
      r=rf(seme)                                                                
      teta=pi2*r                                                                
      vx=ro*COS(teta)                                                          
!                                                                         
    END SUBROUTINE Maxwell
    !
    SUBROUTINE genk(kx,ky,kz)
      IMPLICIT NONE
      REAL(dp):: kx,ky,kz,phi
!
      kx=2.0d0*rf(seme)-1.0
      ky=SQRT(1.0d0-kx*kx)
      phi=pi2*rf(seme)
      kz=ky*SIN(phi)
      ky=ky*COS(phi)
      !
    END SUBROUTINE Genk
!    
  END MODULE Random
