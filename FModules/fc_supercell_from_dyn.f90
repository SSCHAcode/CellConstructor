!
! This code transform in q space the real space dynamical matrix
! Made by Ion Errea
! Originally part of the sscha.x code 
!
subroutine fc_supercell_from_dyn (phitot, q, tau, tau_sc, itau, phitot_sc, nat, nq) 

  implicit none
  
  integer, intent(in) :: nq, nat

  double complex, dimension(nq,3,3,nat,nat), intent(in) :: phitot
  double precision, dimension(3,nq), intent(in) :: q
  double precision, dimension(3,nat), intent(in):: tau
  double precision, dimension(3, nat*nq), intent(in)  ::tau_sc
  ! integer, dimension(nat), intent(in) :: ityp
  integer, dimension(nat*nq), intent(in) ::  itau
  double precision, dimension(3,3,nat*nq,nat*nq), intent(out) :: phitot_sc

  integer :: natsc
  integer :: i, j, alpha, beta, qtot
  double precision, dimension(3) :: latvec
  double complex :: im, one, complex_number
  double precision :: twopi

  one    = (1.0d0,0.0d0)
  im     = (0.0d0,1.0d0)
  twopi  = 6.283185307179586d0 

  natsc = nq * nat

  do i = 1, natsc
    do j = 1, natsc
      latvec(:) = tau_sc(:,i) - tau(:,itau(i)) - tau_sc(:,j) + tau(:,itau(j))
      do alpha = 1, 3
        do beta = 1, 3
          complex_number = (0.0d0,0.0d0)
          do qtot = 1, nq
            print * , "THE Q POINT", qtot, "IS", q(:, qtot), &
                      "THE LATTICE VECTOR", i,j, "IS", latvec(:)
             
            complex_number = complex_number +  &
                             exp( im * twopi * dot_product(q(:,qtot),latvec)) * &
                             phitot(qtot,alpha,beta,itau(i),itau(j)) / &
                             dble(nq) 
          end do 
          if (abs(aimag(complex_number)) .gt. 1.0d-5) then
            print *, complex_number
            print *, ''
            print *, ' ERROR: There are force constants in the supercell that   '
            print *, '        are complex. This is not possible.              '
            print *, '        Stopping...                                     '
            print *, ' 2 '             
            stop
          end if
          phitot_sc(alpha,beta,i,j) = real(complex_number)
        end do
      end do
    end do
  end do

end subroutine fc_supercell_from_dyn



! 
! The following subroutine instead perform the inverse transform
subroutine dyn_from_fc ( phitot_sc, q, tau, tau_sc, itau, dyn, nq, nat) 

  implicit none


  integer :: nq, nat
  double precision, dimension(3,nq), intent(in) :: q
  double precision, dimension(3,nat), intent(in) :: tau
  double precision, dimension(3,3,nq*nat,nq*nat), intent(in) :: phitot_sc
  double precision, dimension(3,nq*nat), intent(in) ::  tau_sc
  !integer, dimension(:), intent(in) :: nqs
  !integer, dimension(:), intent(in) :: ityp
  integer, dimension(nq*nat), intent(in) ::  itau
  double complex, dimension(nq,3,3,nat,nat), intent(out) :: dyn 

  integer :: natsc
  integer :: i, j, k, alpha, beta, qtot, R
  integer :: ka
  double precision, dimension(:,:), allocatable :: latvec
  double precision, dimension(3) :: vecaux
  double complex :: im, one, complex_number
  double precision :: twopi, prec

  one    = (1.0d0,0.0d0)
  im     = (0.0d0,1.0d0)
  twopi  = 6.283185307179586d0 

  prec = 1.0d-6

  natsc = nq * nat

  allocate(latvec(nq,3))

  ! Prepare list of lattice vectors

  ka = 0

  do i = 1, natsc
    if (itau(i) .ne. 1) cycle
    ka = ka + 1
    latvec(ka,:) = tau_sc(:,i) - tau(:,1)
  end do 

  ! Print list of lattice vectors

  do i = 1, nq
    print *, latvec(i,:)
  end do 

  do qtot = 1, nq
    do i = 1, nat
      do j = 1, nat
        do alpha = 1, 3
          do beta = 1, 3
            complex_number = (0.0d0,0.0d0)
            do R = 1, nq
              ! Check what the atom in the supercell is
              do k = 1, natsc
                vecaux = tau(:,i) + latvec(R,:) - tau_sc(:,k)
                if ( sqrt(dot_product(vecaux,vecaux)) .lt. prec ) then
                  complex_number = complex_number +  &
                                  exp( - im * twopi * dot_product(q(:,qtot),latvec(R,:))) * &
                                  phitot_sc(alpha,beta,k,j)
                end if
              end do
            end do
            dyn(qtot,alpha,beta,i,j) = complex_number
          end do
        end do
      end do
    end do
  end do

end subroutine dyn_from_fc

!
!function dot_product(v1, v2) result x
!    implicit none
!    double precision, dimension(:), intent(in) :: v1
!    double precision, dimension(:), intent(in) :: v2
!    double precision :: x 
!    
!    integer n, i
!    
!    n = size(v1)
!    
!    x = 0.0d0
!    do i = 1, n
!        x = x + v1(i) * v2(i)
!    end do
!end function dot_product