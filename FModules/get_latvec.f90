
! This subroutine gets the translation vectors of the
! supercell

subroutine get_latvec ( tau_sc, tau, itau, latvec, nat, nat_sc, nr )

    implicit none
  
    double precision, dimension(3,nat_sc), intent(in) :: tau_sc
    double precision, dimension(3, nat), intent(in) :: tau
    integer, dimension(:), intent(in) :: itau
    double precision, dimension(nr,3), intent(out) :: latvec
    logical, parameter :: debug = .true.
  
    integer :: ka, i, nat_sc, nat, nr
  
    !nat_sc = size(tau_sc(1,:))
    if (debug) then
        print *, "=== DEBUG GET_LATVEC ==="
        print *, "NAT:", nat 
        print *, "NAT_SC:", nat_sc
        print *, "NR (NAT_SC/NAT):", nr
        call flush() 
    endif
  
    ! Prepare list of lattice vectors
  
    ka = 0
  
    do i = 1, nat_sc
      if (itau(i) .ne. 1) cycle
      ka = ka + 1
      latvec(ka,:) = tau_sc(:,i) - tau(:,1)
    end do
  
  end subroutine get_latvec
  