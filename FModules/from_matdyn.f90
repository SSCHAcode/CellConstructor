!
!-----------------------------------------------------------------------
SUBROUTINE frc_blk(dyn,q,tau,nat,nr1,nr2,nr3,frc,at,rws,nrws, nrwsx)
  !-----------------------------------------------------------------------
  ! calculates the dynamical matrix at q from the (short-range part of the)
  ! force constants
  !
  USE io_global,  ONLY : stdout
  !
  IMPLICIT NONE
  INTEGER nr1, nr2, nr3, nat, n1, n2, n3, &
          ipol, jpol, na, nb, m1, m2, m3, nint, i,j, nrws, nrwsx
  double complex, intent(out) :: dyn(3,3,nat,nat)
  double precision, intent(in):: frc(nr1,nr2,nr3,3,3,nat,nat), tau(3,nat), q(3), &
               at(3,3), rws(4,nrwsx)
               
  double precision ::  r(3), weight, r_ws(3), total_weight, arg
  double precision, EXTERNAL :: wsweight
  double precision,ALLOCATABLE :: wscache(:,:,:,:,:)
  
  double precision, parameter :: tpi = 6.283185307179586
  LOGICAL, PARAMETER :: first=.true.
  !
  FIRST_TIME : IF (first) THEN
    !first=.false.
    ALLOCATE( wscache(-2*nr3:2*nr3, -2*nr2:2*nr2, -2*nr1:2*nr1, nat,nat) )
    DO na=1, nat
       DO nb=1, nat
          total_weight=0.0d0
          !
          DO n1=-2*nr1,2*nr1
             DO n2=-2*nr2,2*nr2
                DO n3=-2*nr3,2*nr3
                   DO i=1, 3
                      r(i) = n1*at(i,1)+n2*at(i,2)+n3*at(i,3)
                      r_ws(i) = r(i) + tau(i,na)-tau(i,nb)
                   END DO
                   wscache(n3,n2,n1,nb,na) = wsweight(r_ws,rws,nrws, nrwsx)
                ENDDO
             ENDDO
          ENDDO
      ENDDO
    ENDDO
  ENDIF FIRST_TIME
  !
  DO na=1, nat
     DO nb=1, nat
        total_weight=0.0d0
        DO n1=-2*nr1,2*nr1
           DO n2=-2*nr2,2*nr2
              DO n3=-2*nr3,2*nr3
                 !
                 ! SUM OVER R VECTORS IN THE SUPERCELL - VERY VERY SAFE RANGE!
                 !
                 DO i=1, 3
                    r(i) = n1*at(i,1)+n2*at(i,2)+n3*at(i,3)
                 END DO

                 weight = wscache(n3,n2,n1,nb,na) 
                 IF (weight .GT. 0.0d0) THEN
                    !
                    ! FIND THE VECTOR CORRESPONDING TO R IN THE ORIGINAL CELL
                    !
                    m1 = MOD(n1+1,nr1)
                    IF(m1.LE.0) m1=m1+nr1
                    m2 = MOD(n2+1,nr2)
                    IF(m2.LE.0) m2=m2+nr2
                    m3 = MOD(n3+1,nr3)
                    IF(m3.LE.0) m3=m3+nr3
                    !
                    ! FOURIER TRANSFORM
                    !
                    arg = tpi*(q(1)*r(1) + q(2)*r(2) + q(3)*r(3))
                    DO ipol=1, 3
                       DO jpol=1, 3
                          dyn(ipol,jpol,na,nb) =                 &
                               dyn(ipol,jpol,na,nb) +            &
                               frc(m1,m2,m3,ipol,jpol,na,nb)     &
                               *DCMPLX(COS(arg),-SIN(arg))*weight
                       END DO
                    END DO
                 END IF
                 total_weight=total_weight + weight
              END DO
           END DO
        END DO
        IF (ABS(total_weight-nr1*nr2*nr3).GT.1.0d-8) THEN
           WRITE(stdout,*) "Total weight:", total_weight
           WRITE(stdout,*) "NR1,2,3 = ", nr1, nr2, nr3
           CALL errore ('frc_blk','wrong total_weight',1)
        END IF
     END DO
  END DO
  !
  RETURN
END SUBROUTINE frc_blk
!
!-----------------------------------------------------------------------
subroutine wsinit(rws,nrwsx,nrws,atw)
!-----------------------------------------------------------------------
!
  implicit none
  double precision, intent(inout) :: rws(4,nrwsx)
  double precision, intent(in) :: atw(3,3)
  integer, intent(out) :: nrws
  integer, intent(in) ::  nrwsx
  
  integer i, ii, ir, jr, kr , nx
  double precision eps
  parameter (eps=1.0d-6,nx=2)
  ii = 1
  do ir=-nx,nx
     do jr=-nx,nx
        do kr=-nx,nx
           do i=1,3
              rws(i+1,ii) = atw(i,1)*ir + atw(i,2)*jr + atw(i,3)*kr
           end do
           rws(1,ii)=rws(2,ii)*rws(2,ii)+rws(3,ii)*rws(3,ii)+            &
                               rws(4,ii)*rws(4,ii)
           rws(1,ii)=0.5d0*rws(1,ii)
           if (rws(1,ii).gt.eps) ii = ii + 1
           if (ii.gt.nrwsx) call errore('wsinit', 'ii.gt.nrwsx',1)
        end do
     end do
  end do
  nrws = ii - 1
  return
end subroutine wsinit
!-----------------------------------------------------------------------
function wsweight(r,rws,nrws, nrwsx)
!-----------------------------------------------------------------------
!
  implicit none
  integer ir, nreq, nrws, nrwsx
  double precision, dimension(4, nrwsx), intent(in) :: rws
  double precision r(3), rrt, ck, eps, wsweight
  parameter (eps=1.0d-7)
!
  wsweight = 0.d0
  nreq = 1
  do ir =1,nrws
     rrt = r(1)*rws(2,ir) + r(2)*rws(3,ir) + r(3)*rws(4,ir)
     ck = rrt-rws(1,ir)
     if ( ck .gt. eps ) return
     if ( abs(ck) .lt. eps ) nreq = nreq + 1
  end do
  wsweight = 1.d0/DBLE(nreq)
  return
end function wsweight
