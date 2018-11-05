SUBROUTINE q_gen(nsc,qbid,at_blk,bg_blk,at,bg)
  !-----------------------------------------------------------------------
  ! generate list of q (qbid) that are G-vectors of the supercell
  ! but not of the bulk
  ! FROM MATDYN.f90
  !
  !
  IMPLICIT NONE
  INTEGER :: nsc
  DOUBLE PRECISION, INTENT(in)::  at_blk(3,3), bg_blk(3,3), at(3,3), bg(3,3)
  DOUBLE PRECISION, intent(out) ::  qbid(3,nsc)
  !
  INTEGER, PARAMETER:: nr1=4, nr2=4, nr3=4, &
                       nrm=(2*nr1+1)*(2*nr2+1)*(2*nr3+1)
  DOUBLE PRECISION, PARAMETER:: eps=1.0d-7
  INTEGER :: i, j, k,i1, i2, i3, idum(nrm), iq
  DOUBLE PRECISION :: qnorm(nrm), qbd(3,nrm) ,qwork(3), delta
  LOGICAL lbho
  !
  i = 0
  DO i1=-nr1,nr1
     DO i2=-nr2,nr2
        DO i3=-nr3,nr3
           i = i + 1
           DO j=1,3
              qwork(j) = i1*bg(j,1) + i2*bg(j,2) + i3*bg(j,3)
           END DO ! j
           !
           qnorm(i)  = qwork(1)**2 + qwork(2)**2 + qwork(3)**2
           !
           DO j=1,3
              !
              qbd(j,i) = at_blk(1,j)*qwork(1) + &
                         at_blk(2,j)*qwork(2) + &
                         at_blk(3,j)*qwork(3)
           END DO ! j
           !
           idum(i) = 1
           !
        END DO ! i3
     END DO ! i2
  END DO ! i1
  !
  DO i=1,nrm-1
     IF (idum(i).EQ.1) THEN
        DO j=i+1,nrm
           IF (idum(j).EQ.1) THEN
              lbho=.TRUE.
              DO k=1,3
                 delta = qbd(k,i)-qbd(k,j)
                 lbho = lbho.AND. (ABS(NINT(delta)-delta).LT.eps)
              END DO ! k
              IF (lbho) THEN
                 IF(qnorm(i).GT.qnorm(j)) THEN
                    qbd(1,i) = qbd(1,j)
                    qbd(2,i) = qbd(2,j)
                    qbd(3,i) = qbd(3,j)
                    qnorm(i) = qnorm(j)
                 END IF
                 idum(j) = 0
              END IF
           END IF
        END DO ! j
     END IF
  END DO ! i
  !
  iq = 0
  DO i=1,nrm
     IF (idum(i).EQ.1) THEN
        iq=iq+1
        qbid(1,iq)= bg_blk(1,1)*qbd(1,i) +  &
                    bg_blk(1,2)*qbd(2,i) +  &
                    bg_blk(1,3)*qbd(3,i)
        qbid(2,iq)= bg_blk(2,1)*qbd(1,i) +  &
                    bg_blk(2,2)*qbd(2,i) +  &
                    bg_blk(2,3)*qbd(3,i)
        qbid(3,iq)= bg_blk(3,1)*qbd(1,i) +  &
                    bg_blk(3,2)*qbd(2,i) +  &
                    bg_blk(3,3)*qbd(3,i)
     END IF
  END DO ! i
  !
  IF (iq.NE.nsc) CALL errore('q_gen',' probably nr1,nr2,nr3 too small ', iq)
  RETURN
END SUBROUTINE q_gen
!
