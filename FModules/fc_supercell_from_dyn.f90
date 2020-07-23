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
      latvec(:) = -( tau_sc(:,i) - tau(:,itau(i)) - tau_sc(:,j) + tau(:,itau(j)))
      do alpha = 1, 3
        do beta = 1, 3
          complex_number = (0.0d0,0.0d0)
          do qtot = 1, nq
            !print * , "THE Q POINT", qtot, "IS", q(:, qtot), &
            !          "THE LATTICE VECTOR", i,j, "IS", latvec(:)
             
            complex_number = complex_number +  &
                             exp( - im * twopi * dot_product(q(:,qtot),latvec)) * &
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
            !stop
          end if
          phitot_sc(alpha,beta,i,j) = real(complex_number)
        end do
      end do
    end do
  end do

end subroutine fc_supercell_from_dyn

! This is a fast version of the Fourier transform
! Equal to the one implemented in python
! But much faster
subroutine fast_ft_real_space_from_dynq(unit_cell_coords, super_cell_coords, itau, nat, nat_sc, nq, q_tot, dynq, fc_supercell)
  
  integer, intent(in) :: nat, nat_sc, nq 
  integer, intent(in), dimension(nat_sc) :: itau
  double precision, intent(in) :: unit_cell_coords(nat, 3), super_cell_coords(nat_sc, 3)
  double precision, intent(in), dimension(nq, 3) :: q_tot 
  double complex, intent(in), dimension(nq, 3*nat, 3*nat) :: dynq

  double complex, intent(out), dimension(3*nat_sc, 3*nat_sc) :: fc_supercell


  integer :: i, j, iq, i_uc, j_uc, h, k
  double precision :: R(3), arg, twopi

  double complex :: im, phase

  im     = (0.0d0,1.0d0)
  twopi  = 6.283185307179586d0 

  fc_supercell(:,:) = 0.0d0

  do i = 1, nat_sc
    i_uc = itau(i)
    do j = 1, nat_sc 
      j_uc = itau(j)

      ! Get the distance vector between the two atoms
      R(:) = super_cell_coords(i, :) - unit_cell_coords(i_uc,:)
      R(:) = R(:) - super_cell_coords(j, :) + unit_cell_coords(j_uc, :)

      ! Perform the Fourier transform
      do iq = 1, nq
        arg = twopi * sum(q_tot(iq, :) * R)
        phase = exp(im * arg) / nq

        do h = 1, 3
          do k = 1, 3
            fc_supercell(3*(i-1) + h, 3*(j-1) + k) = fc_supercell(3*(i-1) + h, 3*(j-1) + k) + &
              dynq(iq, 3*(i_uc-1) + h, 3*(j_uc-1) + k) * phase
          enddo
        enddo
      enddo
    enddo
  enddo

  ! Check if the fc supercell has an imaginary value

end subroutine fast_ft_real_space_from_dynq

!
!logical function eqvect1 (x, y)
!  !-----------------------------------------------------------------------
!  !
!  !   This function test if the difference x-y-f is an integer.
!  !   x, y = 3d vectors in crystal axis, f = fractionary translation
!  !
!  implicit none
!  double precision, intent(in) :: x (3), y (3)
!  double precision, parameter :: accep = 1.0d-4 ! acceptance parameter
!
!  !
!  !
!  eqvect1 = abs( x(1)-y(1) - nint(x(1)-y(1)) ) < accep .and. &
!           abs( x(2)-y(2) - nint(x(2)-y(2) )) < accep .and. &
!           abs( x(3)-y(3) - nint(x(3)-y(3) ) ) < accep
!  !
!  return
!end function eqvect1
! 

! by Lorenzo Monacelli
! This subrouitne impose the translation in the supercell
! for the force constant matrix.
! Tau_sc must be in crystal coordinates with respect to the supercell basis!
subroutine impose_trans_sc(fc_sc, tau_sc_cryst, itau, nat_sc)
    implicit none
    
    
    integer, intent(in) :: nat_sc
    double precision, dimension(3, 3, nat_sc, nat_sc), intent(inout) :: fc_sc
    double precision, dimension(3, nat_sc), intent(in) :: tau_sc_cryst
    integer, dimension(nat_sc), intent(in) :: itau
    
    ! ----- HERE THE CODE -----
    double precision, dimension(3, 3) :: fc_tmp
    double precision, dimension(3, 3, nat_sc, nat_sc) :: fc_new

    double precision, dimension(3) :: latvec_1, latvec_2, zero_vec
    integer :: i, j, h, k, counter
    double precision, parameter :: small_value = 1d-6
    logical :: is_equivalent
    
    fc_new = 0.0d0
    do i = 1, nat_sc
        do j = 1, nat_sc
        
            ! Average the force constant matrix for all lattice vectors in the supercell.
            counter = 0
            fc_tmp = 0.0d0
            do h = 1, nat_sc
                if (itau(h) /= itau(i)) cycle ! Check if they are the same atom
                ! Get the lattice vector
                latvec_1 = tau_sc_cryst(:, i) - tau_sc_cryst(:, h)
                
                do k = 1, nat_sc
                    if (itau(k) /= itau(j)) cycle ! Check if they are the same atom
                    
                    ! Get the second lattice vector
                    latvec_2 = tau_sc_cryst(:, j) - tau_sc_cryst(:, k)
                    
                    zero_vec = latvec_1 - latvec_2
                    !print *, "DISTANCE:", zero_vec
                    
                    ! Check if the two lattice vectors are the same apart from
                    ! an unit cell vector
                    ! In that case the fc should be equal
                    is_equivalent = abs( latvec_1(1)-latvec_2(1) - nint(latvec_1(1)-latvec_2(1)) ) < small_value .and. &
                       abs( latvec_1(2)-latvec_2(2) - nint(latvec_1(2)-latvec_2(2) )) < small_value .and. &
                       abs( latvec_1(3)-latvec_2(3) - nint(latvec_1(3)-latvec_2(3) ) ) < small_value
                    if ( is_equivalent ) then
                         fc_tmp = fc_tmp + fc_sc(:, :, h, k)
                         counter  = counter + 1
                         print *, "ATOMS EQ TO:", i, j, "ARE:", h, k
                    end if
                end do
            end do
            
            ! Copy the symmetrized matrix into the original one
            print *, "COUNTER:", counter
            fc_new(:, :, i, j) = fc_tmp / counter
        end do
    end do
    fc_sc = fc_new
end subroutine impose_trans_sc          
    
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
            do R = 1, ka
              ! Check what the atom in the supercell is
              do k = 1, natsc
                vecaux = tau(:,j) + latvec(R,:) - tau_sc(:,k)
                if ( sqrt(dot_product(vecaux,vecaux)) .lt. prec ) then
                  complex_number = complex_number +  &
                                  exp(  im * twopi * dot_product(q(:,qtot),latvec(R,:))) * &
                                  phitot_sc(alpha,beta,i,k)
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
