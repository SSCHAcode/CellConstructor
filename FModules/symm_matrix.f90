SUBROUTINE symmatrix ( matr, s, nsym, at, bg)
   !-----------------------------------------------------------------------
   ! Symmetrize a function f(i,j), i,j=cartesian components
   ! e.g. : stress, dielectric tensor (in cartesian axis) 
   !
   IMPLICIT NONE
   !
   double precision, intent(INOUT) :: matr(3,3)
   integer, intent(IN) :: nsym
   integer, intent(IN) :: s(3,3,48)
   double precision, dimension(3,3), intent(in) :: at, bg
   !
   INTEGER :: isym, i,j,k,l
   double precision :: work (3,3)
   !
   IF (nsym == 1) RETURN
   !
   ! bring matrix to crystal axis
   !
   CALL cart_to_crys_mat ( matr, at )
   !
   ! symmetrize in crystal axis
   !
   work (:,:) = 0.0d0
   DO isym = 1, nsym
      DO i = 1, 3
         DO j = 1, 3
            DO k = 1, 3
               DO l = 1, 3
                    work (i,j) = work (i,j) + &
                        s (i,k,isym) * s (j,l,isym) * matr (k,l)
               END DO
            END DO
         END DO
      END DO
   END DO
   matr (:,:) = work (:,:) / DBLE(nsym)
   !
   ! bring matrix back to cartesian axis
   !
   CALL crys_to_cart_mat ( matr, bg )
   !
 END SUBROUTINE symmatrix
 

 SUBROUTINE cart_to_crys_mat ( matr, at )
    !-----------------------------------------------------------------------
    !     
    IMPLICIT NONE
    !
    double precision, intent(INOUT) :: matr(3,3)
    double precision, intent(IN) :: at(3,3)
    !
    double precision:: work(3,3)
    INTEGER :: i,j,k,l
    !
    work(:,:) = 0.0d0
    DO i = 1, 3
        DO j = 1, 3
        DO k = 1, 3
            DO l = 1, 3
                work(i,j) = work(i,j) + matr(k,l) * at(k,i) * at(l,j)
            END DO
        END DO
        END DO
    END DO
    !
    matr(:,:) = work(:,:)
    !
END SUBROUTINE cart_to_crys_mat
!
SUBROUTINE crys_to_cart_mat ( matr, bg)
    !-----------------------------------------------------------------------
    !
    IMPLICIT NONE
    !
    double precision, intent(INOUT) :: matr(3,3)
    double precision, intent(in) :: bg(3,3)
    !
    double precision :: work(3,3)
    INTEGER :: i,j,k,l
    !
    work(:,:) = 0.0d0
    DO i = 1, 3
        DO j = 1, 3
        DO k = 1, 3
            DO l = 1, 3
                work(i,j) = work(i,j) + &
                            matr(k,l) * bg(i,k) * bg(j,l)
            END DO
        END DO
        END DO
    END DO
    matr(:,:) = work(:,:)
    !
END SUBROUTINE crys_to_cart_mat
!