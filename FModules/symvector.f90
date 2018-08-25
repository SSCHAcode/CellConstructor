   SUBROUTINE symvector (nat, nsym, irt, s, at, bg, vect)
     !-----------------------------------------------------------------------
     ! Symmetrize a function f(i,na), i=cartesian component, na=atom index
     ! e.g. : forces (in cartesian axis) 
     !   
     IMPLICIT NONE
     !   
     INTEGER, INTENT(IN) :: nat, nsym
     INTEGER, INTENT(IN) :: irt(48,nat)  
     INTEGER, INTENT(IN) :: s(3,3,48)  
     double precision, INTENT(IN) :: at(3,3), bg(3,3)
     double precision, intent(INOUT) :: vect(3,nat)
     !   
     INTEGER :: na, isym, nar 
     double precision, ALLOCATABLE :: work (:,:)
     !   
     IF (nsym == 1) RETURN
     !   
     ALLOCATE (work(3,nat))
     !   
     ! bring vector to crystal axis
     !   
     DO na = 1, nat 
        work(:,na) = vect(1,na)*at(1,:) + & 
                     vect(2,na)*at(2,:) + & 
                     vect(3,na)*at(3,:)
     END DO
     !   
     ! symmetrize in crystal axis
     !   
     vect (:,:) = 0.0d0 
     DO na = 1, nat 
        DO isym = 1, nsym
           nar = irt (isym, na) 
           vect (:, na) = vect (:, na) + & 
                          s (:, 1, isym) * work (1, nar) + & 
                          s (:, 2, isym) * work (2, nar) + & 
                          s (:, 3, isym) * work (3, nar)
        END DO
     END DO
     work (:,:) = vect (:,:) / DBLE(nsym)
     !   
     ! bring vector back to cartesian axis
     !   
     DO na = 1, nat 
        vect(:,na) = work(1,na)*bg(:,1) + & 
                     work(2,na)*bg(:,2) + & 
                     work(3,na)*bg(:,3)
     END DO
     !   
     DEALLOCATE (work)
     !   
   END SUBROUTINE symvector
