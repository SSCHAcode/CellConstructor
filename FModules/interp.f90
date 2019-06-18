!
! INTERPOLATION MODULE (BY ION ERREA)
! We use the quantum espresso subroutine taken from the matdyn code
! to interpolate a dynamical matrix in a finer grid.
! This is from interpol.f90 source of the original sscha.x code
!


! ------------------ INTERNAL SUBROUTINES --------------------------
subroutine get_frc( phi_sc, tau, tau_sc, at, itau, frc, &
        size1, size2, size3,nat, natsc)

  implicit none

  integer, intent(in) :: nat, natsc, size1, size2, size3
  double precision, dimension(3,3,natsc,natsc), intent(in) :: phi_sc
  double precision, dimension(3, nat), intent(in) :: tau
  double precision, dimension(3, natsc), intent(in) :: tau_sc
  double precision, dimension(3,3), intent(in) :: at
  integer, dimension(natsc), intent(in) :: itau  
  double precision, dimension(size1, size2, size3,3,3,nat,nat), intent(out) :: frc
  
  
  
  integer :: alpha, beta, i, j, l, m, n, sup1, sup2, sup3
  integer, dimension(3) :: supercell_size
  double precision, dimension(3) :: vect
  double precision, dimension(:,:,:,:,:), allocatable :: phi_auxx

  logical, parameter :: debug = .true.
  
  !natsc = size(tau_sc(1,:))
  !nat   = size(tau(1,:))
  supercell_size(1) = size1
  supercell_size(2) = size2
  supercell_size(3) = size3
  
  ! Print some debugging info
  if (debug) then
     print *, "SUPERCELL SIZE:", supercell_size(:)
     print *, "NAT:", nat
     print *, "The unit cell:"
     do i = 1, 3
        print *, at(:, i)
     enddo
  end if

  allocate(phi_auxx(nat,nat,supercell_size(1),supercell_size(2),supercell_size(3)))

  do alpha = 1, 3
    do beta = 1, 3
      do i = 1, natsc
        do j = 1, nat
          vect(:) = tau_sc(:,i)-tau(:,itau(i))
          call asign_supercell_index_new(vect,at,l,m,n)
          phi_auxx(itau(i),j,l,m,n) = phi_sc(alpha,beta,i,j)
        end do
      end do
      do i = 1, nat 
        do j = 1, nat
          do sup3 = 1, supercell_size(3) 
            do sup2 = 1, supercell_size(2) 
              do sup1 = 1, supercell_size(1)
                frc(sup1,sup2,sup3,alpha,beta,i,j) = phi_auxx(i,j,sup1,sup2,sup3) 
              end do
            end do
          end do 
        end do  
      end do 
    end do
  end do
  

  deallocate(phi_auxx)

end subroutine get_frc


subroutine asign_supercell_index_new(vect,at,l,m,n)

  double precision, dimension(3), intent(in) :: vect
  double precision, dimension(3,3), intent(in) :: at               
  integer, intent(out) :: l
  integer, intent(out) :: m
  integer, intent(out) :: n

  double precision, dimension(3,3) :: matrix
  double precision, dimension(3) :: vector
  integer, dimension(3) :: ipiv
  integer :: i, j, info
  character (len=3) :: logi

  matrix = at           
  call dgetrf(3,3,matrix,3,ipiv,info)
  call dgetrs('N',3,3,matrix,3,ipiv,vect,3,info) 
  l = nint(vect(1)) + 1
  m = nint(vect(2)) + 1
  n = nint(vect(3)) + 1

end subroutine asign_supercell_index_new

!
!subroutine asign_supercell_index(x_atom,x_atoms_prim,primlatt_vec,supercell_size,s,l,m,n,natoms_prim)
!
!  double precision, dimension(3), intent(in) :: x_atom
!  double precision, dimension(3,natoms_prim), intent(in) :: x_atoms_prim     
!  double precision, dimension(3,3), intent(in) :: primlatt_vec     
!  integer, dimension(3), intent(in) :: supercell_size
!  integer, intent(in) :: natoms_prim
!  integer, intent(out) :: s
!  integer, intent(out) :: l
!  integer, intent(out) :: m
!  integer, intent(out) :: n
!
!  double precision, dimension(3,3) :: matrix
!  double precision, dimension(3) :: vector
!  integer, dimension(3) :: ipiv
!  integer :: i, j, natoms_prim, info
!  character (len=3) :: logi
!
!  !natoms_prim = size(x_atoms_prim(1,:))
!  
!  do j = 1, natoms_prim
!!    matrix = transpose(primlatt_vec)
!    matrix = primlatt_vec
!    vector(:) = x_atom(:) - x_atoms_prim(:,j)
!    call dgetrf(3,3,matrix,3,ipiv,info)
!    call dgetrs('N',3,3,matrix,3,ipiv,vector,3,info) 
!    call integer_test(vector,supercell_size,logi)
!    if (logi .eq. 'yes') then
!      s = j
!      l = int(vector(1)) + 1
!      m = int(vector(2)) + 1
!      n = int(vector(3)) + 1
!    end if
!  end do
!
!end subroutine asign_supercell_index

!
!subroutine integer_test(vector,supercell_size,logi)
!
!  double precision, dimension(3), intent(in) :: vector
!  integer, dimension(3), intent(in) :: supercell_size
!  character (len=3), intent(out) :: logi
!
!  integer :: i, j, k
!
!  logi = 'non'
!
!  do i = 0, supercell_size(1)
!    if (vector(1)-dble(i) .eq. 0.0d0) then
!      do j = 0, supercell_size(2)
!        if (vector(2)-dble(j) .eq. 0.0d0) then
!          do k = 0, supercell_size(3)
!            if (vector(3)-dble(k) .eq. 0.0d0) then
!              logi = 'yes'
!            end if
!          end do
!        end if
!      end do
!    end if
!  end do    
!
!end subroutine integer_test

