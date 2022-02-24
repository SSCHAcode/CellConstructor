module third_order_interpol

contains

subroutine interpol(fc,R2,R3,q2,q3,fc_interp,n_blocks,nat)
    IMPLICIT NONE
    !
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    INTEGER, intent(IN) :: nat, n_blocks
    REAL(DP), intent(IN) :: R2(3,n_blocks),R3(3,n_blocks)
    COMPLEX(DP),INTENT(in)   :: fc(3*nat,3*nat,3*nat,n_blocks)
    REAL(DP),INTENT(in) :: q2(3), q3(3)
    COMPLEX(DP),INTENT(out) :: fc_interp(3*nat, 3*nat, 3*nat)
    !
    REAL(DP), parameter :: tpi=3.14159265358979323846_DP*2.0_DP
    REAL(DP) :: arg
    COMPLEX(DP) :: phase
    INTEGER :: i_block, a,b,c
    !
    fc_interp = (0._dp, 0._dp)
    !

    DO i_block = 1, n_blocks
      arg = tpi * SUM(q2(:)*R2(:,i_block) + q3(:)*R3(:,i_block))
      phase = CMPLX(Cos(arg),Sin(arg), kind=DP)
      !
      DO c = 1,3*nat
      DO b = 1,3*nat
      DO a = 1,3*nat
        fc_interp(a,b,c) = fc_interp(a,b,c) + phase*fc(a,b,c,i_block)
      ENDDO
      ENDDO
      ENDDO
      ! 
    END DO
end subroutine interpol
!
end module third_order_interpol
