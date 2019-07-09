
! This subroutine calculates which atom in the supercell
! is related by a translation vector of the supercell 
! to another atom of the supercell

subroutine get_tau_sc_latvec ( tau_sc, latvec, at_sc, tau_sc_latvec, nat_sc, nr )

  implicit none

  double precision, dimension(3,nat_sc), intent(in) :: tau_sc
  double precision, dimension(nr, 3), intent(in) :: latvec 
  double precision, dimension(3,3), intent(in) :: at_sc
  integer, dimension(nat_sc,nr), intent(out) :: tau_sc_latvec

  integer :: nr, nat_sc
  double precision, dimension(3) :: diff
  double precision, dimension(27,3) :: superlatvec
  double precision :: prec
  logical, parameter :: debug = .true.
   
  integer :: ka, i, j, k, r

  ! Define precision for scalar product that
  ! decides if two positions are the same

  if (debug) then
    print *, "=== DEBUG get_tau_sc_latvec ==="
    print *, "NAT_SC:", nat_sc
    print *, "NR:", NR
    print *, ""
    call flush()
  endif

  prec = 1.0d-6

  ! Get integers

  !nr = size(latvec(:,1))
  !nat_sc = size(tau_sc(1,:))

  ! Create the supercell lattice vectors

  ka = 0

  do i = -1, 1
    do j = -1, 1
      do k = -1, 1
        ka = ka + 1
        superlatvec(ka,:) = dble(i) * at_sc(:,1) + dble(j) * at_sc(:,2) + dble(k) * at_sc(:,3)
      end do
    end do
  end do

  ! Calculate which is the atom of the supercell related to a given 
  ! lattice vector

  do i = 1, nat_sc
    do r = 1, nr
      do j = 1, nat_sc
        do ka = 1, 27   
          diff(:) = tau_sc(:,i) + latvec(r,:) - tau_sc(:,j) + superlatvec(ka,:)
          if (dot_product(diff,diff) .lt. prec) then
            tau_sc_latvec(i,r) = j
            print *, ''
            print '(a,i3,a,3f16.8)', ' Supercell atom ', i, ' : ', tau_sc(:,i)
            print '(a,i3,a,3f16.8)', ' Translation    ', r, ' : ', latvec(r,:)
            print '(a,i3,a,3f16.8)', ' Translate atom ', j, ' : ', tau_sc(:,j)
          end if
        end do
      end do
    end do
  end do

end subroutine get_tau_sc_latvec


! This subroutine imposes the permutation symmetry in the 
! third order force constant matrices. The input is given
! with three indices, where each index represents an atom and
! a Cartesian index

subroutine permute_v3 (v3,n)

  implicit none

  double precision, dimension(n,n,n), intent(inout) :: v3

  integer :: n
  integer :: a, b, c

  ! Assign permutation symmetry

  do a = 1, n
    do b = 1, n
      do c = 1, n
        v3(a,b,c) = (v3(a,b,c) + v3(a,c,b) + v3(b,a,c) + v3(b,c,a) + v3(c,a,b) + v3(c,b,a)) / 6.0d0
        v3(a,c,b) = v3(a,b,c)
        v3(b,a,c) = v3(a,b,c)
        v3(b,c,a) = v3(a,b,c)
        v3(c,a,b) = v3(a,b,c)
        v3(c,b,a) = v3(a,b,c)
      end do
    end do
  end do  

end subroutine permute_v3 

! This subroutine imposes the permutation symmetry in the 
! fourth order force constant matrices. The input is given
! with four indices, where each index represents an atom and
! a Cartesian index

subroutine permute_v4 (v4, n)

  implicit none

  double precision, dimension(n,n,n,n), intent(inout) :: v4

  integer :: n
  integer :: a, b, c, d


  ! Assign permutation symmetry

  do a = 1, n
    do b = 1, n
      do c = 1, n
        do d = 1, n
          v4(a,b,c,d) = ( v4(a,b,c,d) +  v4(a,b,d,c) +  v4(a,c,b,d) +  v4(a,c,d,b) +  v4(a,d,b,c) +  v4(a,d,c,b) &
                        + v4(b,a,c,d) +  v4(b,a,d,c) +  v4(b,d,a,c) +  v4(b,d,c,a) +  v4(b,c,a,d) +  v4(b,c,d,a) &
                        + v4(c,a,b,d) +  v4(c,a,d,b) +  v4(c,b,a,d) +  v4(c,b,d,a) +  v4(c,d,a,b) +  v4(c,d,b,a) &
                        + v4(d,a,b,c) +  v4(d,a,c,b) +  v4(d,b,a,c) +  v4(d,b,c,a) +  v4(d,c,a,b) +  v4(d,c,b,a) ) / 24.0d0
          v4(a,b,d,c) = v4(a,b,c,d)
          v4(a,c,b,d) = v4(a,b,c,d)
          v4(a,c,d,b) = v4(a,b,c,d)
          v4(a,d,b,c) = v4(a,b,c,d)
          v4(a,d,c,b) = v4(a,b,c,d)
          !
          v4(b,a,c,d) = v4(a,b,c,d)
          v4(b,a,d,c) = v4(a,b,c,d)
          v4(b,d,a,c) = v4(a,b,c,d)
          v4(b,d,c,a) = v4(a,b,c,d)
          v4(b,c,a,d) = v4(a,b,c,d)
          v4(b,c,d,a) = v4(a,b,c,d)
          !
          v4(c,a,b,d) = v4(a,b,c,d)
          v4(c,a,d,b) = v4(a,b,c,d)
          v4(c,b,a,d) = v4(a,b,c,d)
          v4(c,b,d,a) = v4(a,b,c,d)
          v4(c,d,a,b) = v4(a,b,c,d)
          v4(c,d,b,a) = v4(a,b,c,d)
          !
          v4(d,a,b,c) = v4(a,b,c,d)
          v4(d,a,c,b) = v4(a,b,c,d)
          v4(d,b,a,c) = v4(a,b,c,d)
          v4(d,b,c,a) = v4(a,b,c,d)
          v4(d,c,a,b) = v4(a,b,c,d)
          v4(d,c,b,a) = v4(a,b,c,d)
        end do
      end do
    end do
  end do

end subroutine permute_v4


! This subroutine imposes the translational symmetry to the 
! second order force constants.
!

subroutine trans_v2 ( v2, tau_sc_latvec, nat_sc, nr )

  implicit none

  double precision, dimension(3, 3, nat_sc,nat_sc), intent(inout) :: v2
  integer, dimension(nat_sc,nr), intent(in) :: tau_sc_latvec
  integer :: nat_sc, nr
  
  integer :: ka, i, j, k, l, r, is, js, la, r1, r2
  double precision, dimension(3,3) :: mat_aux
  logical, parameter :: debug = .true.

  !nat    = size(tau(1,:))
  !nat_sc = size(tau_sc(1,:))

  if (debug) then
    print *, "=== DEBUG TRANS V2 ==="
    print *, "NAT_SC:", nat_sc
    print *, "NR:", nr 
    call flush()
  endif


  ! Impose translational symmetry

  do i = 1, nat_sc
    do j = 1, nat_sc
        mat_aux = 0.0d0
        do r = 1, nr
           mat_aux(:,:) = mat_aux(:,:) &
                          + v2(:, :, tau_sc_latvec(i,r),tau_sc_latvec(j,r))
        end do
        mat_aux(:,:) = mat_aux(:,:) / dble(nr)
        do r = 1, nr
           v2(:, :, tau_sc_latvec(i,r),tau_sc_latvec(j,r)) = mat_aux(:,:)
        end do
      end do
  end do

end subroutine trans_v2

! This subroutine imposes the translational symmetry to the 
! third order force constants.
!
! Both in the input and output the third order force constants
! are given only with three indexes, each representing both
! an atom and a Cartesian index, but inside it is used
! with 6 indexes, separating cartesian and atom indexes.
subroutine trans_v3 ( v3, tau_sc_latvec, nat_sc, nr)!tau, tau_sc, itau, at_sc, nat, nat_sc )

  implicit none

  double precision, dimension(nat_sc*3,nat_sc*3,nat_sc*3), intent(inout) :: v3
  integer, dimension(nat_sc,nr), intent(in) :: tau_sc_latvec
  !double precision, dimension(3,nat), intent(in) :: tau
  !double precision, dimension(3,nat_sc), intent(in) :: tau_sc
  !integer, dimension(nat_sc), intent(in) :: itau
  !double precision, dimension(3,3), intent(in) :: at_sc

  integer :: nat_sc, nr
  !double precision, dimension(3) :: cholat, vect, diff
  double precision, dimension(:,:,:,:,:,:), allocatable :: v3_6
  !double precision, dimension(:,:), allocatable :: latvec
  double precision :: prec
  !integer, dimension(:,:), allocatable :: tau_sc_latvec
  !logical, dimension(:), allocatable :: assigned
  integer :: ka, i, j, k, l, r, is, js, la, r1, r2
  double precision, dimension(3,3,3) :: mat_aux
  logical, parameter :: debug = .true.

  prec = 1.0d-6

  !nat    = size(tau(1,:))
  !nat_sc = size(tau_sc(1,:))

  !nr = nat_sc / nat

  if (debug) then
    print *, "=== DEBUG TRANS V3 ==="
    print *, "NAT_SC:", nat_sc
    !print *, "NAT:", nat 
    print *, "NR:", nr 
    call flush()
  endif

  !allocate( assigned(nr) )
  allocate( v3_6(nat_sc,nat_sc,nat_sc,3,3,3) )
  !allocate( latvec(nr,3) )
  !allocate( tau_sc_latvec(nat_sc,nr) )

  ! Get the lattice vectors of the supercell

  !call get_latvec ( tau_sc, tau, itau, latvec, nat, nat_sc, nr )

  ! Build the 3rd order force-constant matrices 
  ! in 6 rank tensor

  call threetosix_real ( v3, v3_6, nat_sc)

  ! Assign which is the transformed atom in the supercell
  ! given a particular translation vector

  !call get_tau_sc_latvec ( tau_sc, latvec, at_sc, tau_sc_latvec, nat_sc, nr )

  ! Impose translational symmetry

  do i = 1, nat_sc
    do j = 1, nat_sc
      do k = 1, nat_sc
        mat_aux = 0.0d0
        do r = 1, nr
           mat_aux(:,:,:) = mat_aux(:,:,:) &
                          + v3_6(tau_sc_latvec(i,r),tau_sc_latvec(j,r),tau_sc_latvec(k,r),:,:,:)
        end do
        mat_aux(:,:,:) = mat_aux(:,:,:) / dble(nr)
        do r = 1, nr
           v3_6(tau_sc_latvec(i,r),tau_sc_latvec(j,r),tau_sc_latvec(k,r),:,:,:) = mat_aux(:,:,:)
        end do
      end do
    end do
  end do

  ! Return to the rank 3 tensor of the third order force constants
  ! matrices

  call sixtothree_real ( v3_6, v3, nat_sc)

end subroutine trans_v3

! This subroutine imposes the translational symmetry to the 
! fourth order force constants.
!
! Both in the input and output the third order force constants
! are given only with three indexes, each representing both
! an atom and a Cartesian index, but inside it is used
! with 6 indexes, separating cartesian and atom indexes.
! TODO: TO BE CONVERTED IN PYTHONIC
!( v3, tau_sc_latvec, nat_sc, nr)
subroutine trans_v4 ( v4, tau_sc_latvec, nat_sc, nr )

  implicit none

  double precision, dimension(3*nat_sc,3*nat_sc,3*nat_sc,3*nat_sc), intent(inout) :: v4
  integer, dimension(nat_sc,nr), intent(in) :: tau_sc_latvec

  ! double precision, dimension(:,:), intent(in) :: tau, tau_sc
  ! integer, dimension(:), intent(in) :: itau
  ! double precision, dimension(3,3), intent(in) :: at_sc

  integer :: nat, nat_sc, nr

  ! double precision, dimension(3) :: cholat, vect, diff
  double precision, dimension(:,:,:,:,:,:), allocatable :: v3_6
  ! double precision, dimension(:,:), allocatable :: latvec
  double precision :: prec
  logical, dimension(:), allocatable :: assigned
  integer :: ka, i, j, k, l, r, is, js, la, r1, r2
  double precision, dimension(3,3,3,3) :: mat_aux
  logical, parameter :: debug = .true.

  prec = 1.0d-6

  nat = nat_sc / nr

  if (debug) then
    print *, "=== DEBUG TRANS_V4 ==="
    print *, "NAT_SC:", nat_sc
    print *, "NR:", nr 
    print *, "NAT:", nat
    call flush()
  endif


  ! allocate( assigned(nr) )
  ! allocate( latvec(nr,3) )
  ! allocate( tau_sc_latvec(nat_sc,nr) )

  ! ! Get the lattice vectors of the supercell

  ! call get_latvec ( tau_sc, tau, itau, latvec, nat, nat_sc, nr )

  ! ! Assign which is the transformed atom in the supercell
  ! ! given a particular translation vector

  ! call get_tau_sc_latvec ( tau_sc, latvec, at_sc, tau_sc_latvec )

  ! Impose translational symmetry

  do i = 1, nat_sc
    do j = 1, nat_sc
      do k = 1, nat_sc
        do l = 1, nat_sc
          mat_aux = 0.0d0
          do r = 1, nr
             mat_aux(:,:,:,:) = mat_aux(:,:,:,:) &
                              + v4((3*(tau_sc_latvec(i,r)-1)+1):(3*(tau_sc_latvec(i,r)-1)+3), &
                                   (3*(tau_sc_latvec(j,r)-1)+1):(3*(tau_sc_latvec(j,r)-1)+3), &
                                   (3*(tau_sc_latvec(k,r)-1)+1):(3*(tau_sc_latvec(k,r)-1)+3), &
                                   (3*(tau_sc_latvec(l,r)-1)+1):(3*(tau_sc_latvec(l,r)-1)+3))  
          end do
          mat_aux(:,:,:,:) = mat_aux(:,:,:,:) / dble(nr)
          do r = 1, nr
             v4((3*(tau_sc_latvec(i,r)-1)+1):(3*(tau_sc_latvec(i,r)-1)+3), &
                (3*(tau_sc_latvec(j,r)-1)+1):(3*(tau_sc_latvec(j,r)-1)+3), &
                (3*(tau_sc_latvec(k,r)-1)+1):(3*(tau_sc_latvec(k,r)-1)+3), &
                (3*(tau_sc_latvec(l,r)-1)+1):(3*(tau_sc_latvec(l,r)-1)+3)) = mat_aux(:,:,:,:)
          end do
        end do
      end do
    end do
  end do

end subroutine trans_v4

! This subroutine imposes the point group symmetry in the second-order
! force-constants

subroutine sym_v2 ( v2, at_sc, bg_sc, s, irt, nsym, nat_sc)

  implicit none

  double precision, dimension(3,3,nat_sc,nat_sc), intent(inout) :: v2
  double precision, dimension(3,3), intent(in) :: at_sc
  double precision, dimension(3,3), intent(in) :: bg_sc
  ! Symmetry stuff

  integer, dimension(3,3,48), intent(in) :: s
  integer, dimension(48,nat_sc), intent(in) :: irt
  integer :: nsym, nat_sc

  INTEGER :: na, nb, nc, isym, nar, nbr, ncr
  double precision, ALLOCATABLE :: work (:,:,:,:)
  !double precision, dimension(3,3) :: bg_sc

  integer :: iq, i, j, k, alpha, beta, gamm
  logical, parameter :: debug = .true.

  if (debug) then
    print *, "=== DEBUG SYM_V2 ==="
    print *, "NSYM:", nsym 
    print *, "NAT_SC:", nat_sc 
    call flush()
  endif

  !logical :: prnt_sym

  ! Extract integers

  !nat_sc = size(tau_sc(1,:))

  ! Allocate variables

  ! Create reciprocal lattice vectors of supercell

  !CALL recips(at_sc(1,1),at_sc(1,2),at_sc(1,3),bg_sc(1,1),bg_sc(1,2),bg_sc(1,3))

  ! Assign values to print fake dynamical matrix in the supercell

  ! Write fake dynamical matrix

!  call write_dyn (phitot, q, ntyp, nqs, ityp_sc, amass, &
!                            fildyn_prefix, ibrav, celldm, tau_sc, &
!                            type_name, at_sc, lrigid, epsil, zeu)
 
  ! Extract all information about symmetries


  ! Symmetrize the third order force constant matrix

     ALLOCATE (work(3,3,nat_sc,nat_sc))
     !   
     ! bring third-order matrix to crystal axis
     !   
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         do alpha = 1, 3  
           do beta = 1, 3  
             work(alpha,beta,na,nb) = 0.0d0
             do i = 1, 3
               do j = 1, 3
                 work(alpha,beta,na,nb) = work(alpha,beta,na,nb) + &
                                          v2(i,j,na,nb)*at_sc(i,alpha)*at_sc(j,beta)
               end do
             end do
           end do
         END DO
       END DO
     END DO
     !   
     ! symmetrize in crystal axis
     !   
     v2 = 0.0d0
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         do alpha = 1, 3  
           do beta = 1, 3  
             DO isym = 1, nsym
               nar = irt (isym, na)
               nbr = irt (isym, nb)
               do i = 1, 3
                 do j = 1, 3
                   v2(alpha,beta,na,nb) = v2(alpha,beta,na,nb) + &
                                          work(i,j,nar,nbr)*s(alpha,i,isym)*s(beta,j,isym)
                 end do
               end do
             end do
           end do
         END DO
       END DO
     END DO
     work (:,:,:,:) = v2 (:,:,:,:) / DBLE(nsym)
     !   
     ! bring vector back to cartesian axis
     !   
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         do alpha = 1, 3
           do beta = 1, 3
             v2(alpha,beta,na,nb) = 0.0d0
             do i = 1, 3
               do j = 1, 3
                 v2(alpha,beta,na,nb) = v2(alpha,beta,na,nb) + &
                                        work(i,j,na,nb)*bg_sc(alpha,i)*bg_sc(beta,j)
               end do
             end do
           end do
         END DO
       END DO
     END DO
     !   
     DEALLOCATE (work)

end subroutine sym_v2

! =========================
! This subroutine imposes the point group symmetry in the third-order
! force-constants

subroutine sym_v3 ( v3, at_sc, s, irt, nsym, nat_sc )

  implicit none

  double precision, dimension(nat_sc*3,nat_sc*3,nat_sc*3), intent(inout) :: v3
  !integer, dimension(nat_sc), intent(in) :: ityp_sc
  !double precision, dimension(ntyp), intent(in) :: amass
  !integer, intent(in) :: ibrav
  !double precision, dimension(6), intent(in) :: celldm
  !double precision, dimension(3,nat_sc), intent(in) :: tau_sc
  !character (len=3), dimension(:), intent(in) :: type_name
  double precision, dimension(3,3), intent(in) :: at_sc
  !integer, dimension(nat_sc), intent(in) :: itau

  ! The symmetries
  integer, dimension(3,3,48), intent(in) :: s
  integer, dimension(48, nat_sc), intent(in) :: irt

  integer :: nsym
  integer :: nat_sc

  !double complex, dimension(:,:,:,:,:), allocatable  :: phitot
  !double precision, dimension(3,1) :: q
  !integer, dimension(1) :: nqs
  !logical :: lrigid
  !character (len=50) :: fildyn_prefix
  !character (len=512) :: fildyn
  !double precision, dimension(3,3) :: epsil
  !double precision, dimension(:,:,:), allocatable :: zeu
  !character(len=6), EXTERNAL :: int_to_char

  ! Symmetry stuff

  !integer, dimension(48) :: invs, irgq, isq
  !double precision, dimension(3,48) :: sxq
  !double precision, dimension(:,:,:), allocatable :: rtau
  !integer :: nsymq, irotmq, nsym, imq
  !logical :: minus_q

  INTEGER :: na, nb, nc, isym, nar, nbr, ncr
  double precision, ALLOCATABLE :: work (:,:,:,:,:,:)
  double precision, dimension(3,3) :: bg_sc

  double precision, dimension(:,:,:,:,:,:), allocatable :: v32

  integer :: iq, i, j, k, alpha, beta, gamm

  ! Extract integers

  !nat_sc = size(tau_sc(1,:))

  ! Allocate variables

  allocate(v32(nat_sc,nat_sc,nat_sc,3,3,3))

  !allocate(phitot(1,3,3,nat_sc,nat_sc))
  !allocate(zeu(3,3,nat_sc))

  !allocate(rtau(3,48,nat_sc))
  !allocate(irt(48,nat_sc))

  ! Create reciprocal lattice vectors of supercell

  CALL recips(at_sc(1,1),at_sc(1,2),at_sc(1,3),bg_sc(1,1),bg_sc(1,2),bg_sc(1,3))

  ! Write third-order force constants with 6 indexes

  call threetosix_real (v3,v32, nat_sc)

  ! Assign values to print fake dynamical matrix in the supercell

!   q = 0.0d0
!   nqs = 1
!   lrigid = .false.
!   epsil = 0.0d0
!   phitot = (0.0d0,0.0d0)
!   zeu = 0.0d0
!   fildyn_prefix = 'fake_dyn'

!   ! Write fake dynamical matrix

! !  call write_dyn (phitot, q, ntyp, nqs, ityp_sc, amass, &
! !                            fildyn_prefix, ibrav, celldm, tau_sc, &
! !                            type_name, at_sc, lrigid, epsil, zeu)
!   call write_dyn (phitot, q, ntyp, nqs, ityp_sc, amass, &
!                             fildyn_prefix, 0, celldm, tau_sc, &
!                             type_name, at_sc, lrigid, epsil, zeu)

!   ! Extract all information about symmetries

!   print *, ''
!   print *, ' Extracting symmetries of the supercell... '
!   print *, ''

!   iq = 1

!   fildyn = trim(fildyn_prefix) // int_to_char(iq)

!   call symmdynmat ( fildyn, nat_sc, phitot(1,:,:,:,:),  q, &
!                       s, invs, rtau, irt, irgq, &
!                       nsymq, irotmq, minus_q, nsym, nqs, isq, &
!                       imq, sxq, lrigid, epsil, zeu )

  call print_symm ( s, nsym, irt, .true., nat_sc)

  ! Symmetrize the third order force constant matrix

     ALLOCATE (work(nat_sc,nat_sc,nat_sc,3,3,3))
     !   
     ! bring third-order matrix to crystal axis
     !   
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         DO nc = 1, nat_sc
           do alpha = 1, 3  
             do beta = 1, 3  
               do gamm = 1, 3  
                 work(na,nb,nc,alpha,beta,gamm) = 0.0d0
                 do i = 1, 3
                   do j = 1, 3
                     do k = 1, 3
                       work(na,nb,nc,alpha,beta,gamm) = work(na,nb,nc,alpha,beta,gamm) + &
                                  v32(na,nb,nc,i,j,k)*at_sc(i,alpha)*at_sc(j,beta)*at_sc(k,gamm)  
                     end do
                   end do
                 end do
               end do
             end do
           end do
         END DO
       END DO
     END DO
     !   
     ! symmetrize in crystal axis
     !   
     v32 = 0.0d0
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         DO nc = 1, nat_sc
           do alpha = 1, 3  
             do beta = 1, 3  
               do gamm = 1, 3  
                 DO isym = 1, nsym
                   nar = irt (isym, na)
                   nbr = irt (isym, nb)
                   ncr = irt (isym, nc)
                   do i = 1, 3
                     do j = 1, 3
                       do k = 1, 3
                         v32(na,nb,nc,alpha,beta,gamm) = v32(na,nb,nc,alpha,beta,gamm) + &
                                    work(nar,nbr,ncr,i,j,k)*s(alpha,i,isym)*s(beta,j,isym)*s(gamm,k,isym)  
                       end do
                     end do
                   end do
                 END DO 
               end do
             end do
           end do
         END DO
       END DO
     END DO
     work (:,:,:,:,:,:) = v32 (:,:,:,:,:,:) / DBLE(nsym)
     !   
     ! bring vector back to cartesian axis
     !   
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         DO nc = 1, nat_sc
           do alpha = 1, 3
             do beta = 1, 3
               do gamm = 1, 3
                 v32(na,nb,nc,alpha,beta,gamm) = 0.0d0
                 do i = 1, 3
                   do j = 1, 3
                     do k = 1, 3
                       v32(na,nb,nc,alpha,beta,gamm) = v32(na,nb,nc,alpha,beta,gamm) + &
                                  work(na,nb,nc,i,j,k)*bg_sc(alpha,i)*bg_sc(beta,j)*bg_sc(gamm,k)
                     end do
                   end do
                 end do
               end do
             end do
           end do
         END DO
       END DO
     END DO
     !   
     DEALLOCATE (work)

  ! Write third-order force constants back with 3 indexes

  call sixtothree_real (v32,v3, nat_sc)

  ! Deallocate stuff

  ! deallocate(phitot)
  ! deallocate(zeu)
  ! deallocate(rtau)
  ! deallocate(irt)
end subroutine sym_v3

! This subroutine imposes the point group symmetry in the fourth-order
! force-constants

subroutine sym_v4 ( v4, at_sc, s, irt, nsym, nat_sc )

  implicit none

  double precision, dimension(3*nat_sc,3*nat_sc,3*nat_sc,3*nat_sc), intent(inout) :: v4
  !integer, intent(in) :: ntyp
  !integer, dimension(:), intent(in) :: ityp_sc
  !double precision, dimension(:), intent(in) :: amass
  !integer, intent(in) :: ibrav
  !double precision, dimension(6), intent(in) :: celldm
  !double precision, dimension(:,:), intent(in) :: tau_sc
  !character (len=3), dimension(:), intent(in) :: type_name
  double precision, dimension(3,3), intent(in) :: at_sc
  integer, dimension(3,3,48), intent(in) :: s
  integer, dimension(48, nat_sc), intent(in) :: irt
  integer, intent(in) :: nsym
  !integer, dimension(:), intent(in) :: itau


  !double complex, dimension(:,:,:,:,:), allocatable  :: phitot
  !double precision, dimension(3,1) :: q
  !integer, dimension(1) :: nqs
  !logical :: lrigid
  !character (len=50) :: fildyn_prefix
  !character (len=512) :: fildyn
  !double precision, dimension(3,3) :: epsil
  !double precision, dimension(:,:,:), allocatable :: zeu
  !character(len=6), EXTERNAL :: int_to_char

  ! Symmetry stuff

  ! integer, dimension(48) :: invs, irgq, isq
  ! double precision, dimension(3,48) :: sxq
  ! double precision, dimension(:,:,:), allocatable :: rtau
  ! integer :: nsymq, irotmq, nsym, imq
  ! logical :: minus_q

  INTEGER :: na, nb, nc, nd, isym, nar, nbr, ncr, ndr
  double precision, ALLOCATABLE :: work (:,:,:,:)
  double precision, dimension(3,3) :: bg_sc

  integer :: nat_sc

  integer :: iq, i, j, k, l, alpha, beta, gamm, delt 
  logical, parameter :: debug = .true.

  ! Extract integers

  !nat_sc = size(tau_sc(1,:))
  if (debug) then
    print *, "=== DEBUG SYM_V4 ==="
    print *, "NAT_SC:", nat_sc 
    print *, "NSYM:", nsym 
    call flush()
  end if  

  ! Allocate variables

  ! allocate(phitot(1,3,3,nat_sc,nat_sc))
  ! allocate(zeu(3,3,nat_sc))

  ! allocate(rtau(3,48,nat_sc))
  ! allocate(irt(48,nat_sc))

  ! Create reciprocal lattice vectors of supercell

  CALL recips(at_sc(1,1),at_sc(1,2),at_sc(1,3),bg_sc(1,1),bg_sc(1,2),bg_sc(1,3))

  ! Assign values to print fake dynamical matrix in the supercell

!   q = 0.0d0
!   nqs = 1
!   lrigid = .false.
!   epsil = 0.0d0
!   phitot = (0.0d0,0.0d0)
!   zeu = 0.0d0
!   fildyn_prefix = 'fake_dyn'

!   ! Write fake dynamical matrix

! !  call write_dyn (phitot, q, ntyp, nqs, ityp_sc, amass, &
! !                            fildyn_prefix, ibrav, celldm, tau_sc, &
! !                            type_name, at_sc, lrigid, epsil, zeu)
!   call write_dyn (phitot, q, ntyp, nqs, ityp_sc, amass, &
!                             fildyn_prefix, 0, celldm, tau_sc, &
!                             type_name, at_sc, lrigid, epsil, zeu)

!   ! Extract all information about symmetries

!   print *, ''
!   print *, ' Extracting symmetries of the supercell... '
!   print *, ''

!   iq = 1

!   fildyn = trim(fildyn_prefix) // int_to_char(iq)

!   call symmdynmat ( fildyn, nat_sc, phitot(1,:,:,:,:),  q, &
!                       s, invs, rtau, irt, irgq, &
!                       nsymq, irotmq, minus_q, nsym, nqs, isq, &
!                       imq, sxq, lrigid, epsil, zeu )

  call print_symm ( s, nsym, irt, .true., nat_sc)

  ! Symmetrize the fourth order force constant matrix

     ALLOCATE (work(3*nat_sc,3*nat_sc,3*nat_sc,3*nat_sc))
     !   
     ! bring third-order matrix to crystal axis
     !   
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         DO nc = 1, nat_sc
           DO nd = 1, nat_sc
             do alpha = 1, 3  
               do beta = 1, 3  
                 do gamm = 1, 3  
                   do delt = 1, 3
                     work(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) = 0.0d0
                     do i = 1, 3
                       do j = 1, 3
                         do k = 1, 3
                           do l = 1, 3
                             work(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) = &
                                work(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) + &
                                v4(3*(na-1)+i,3*(nb-1)+j,3*(nc-1)+k,3*(nd-1)+l)* & 
                                at_sc(i,alpha)*at_sc(j,beta)*at_sc(k,gamm)*at_sc(l,delt)
                           end do
                         end do
                       end do
                     end do
                   end do
                 end do
               end do
             end do
           end do
         END DO
       END DO
     END DO
     !   
     ! symmetrize in crystal axis
     !   
     v4 = 0.0d0
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         DO nc = 1, nat_sc
           DO nd = 1, nat_sc
             do alpha = 1, 3  
               do beta = 1, 3  
                 do gamm = 1, 3  
                   do delt = 1, 3
                     DO isym = 1, nsym
                       nar = irt (isym, na)
                       nbr = irt (isym, nb)
                       ncr = irt (isym, nc)
                       ndr = irt (isym, nd)
                       do i = 1, 3
                         do j = 1, 3
                           do k = 1, 3
                             do l = 1, 3
                               v4(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) = &
                                  v4(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) + &
                                  work(3*(nar-1)+i,3*(nbr-1)+j,3*(ncr-1)+k,3*(ndr-1)+l) * &
                                  s(alpha,i,isym)*s(beta,j,isym)*s(gamm,k,isym)*s(delt,l,isym)  
                             end do
                           end do
                         end do
                       end do
                     end do
                   end do
                 END DO 
               end do
             end do
           end do
         END DO
       END DO
     END DO
     work (:,:,:,:) = v4 (:,:,:,:) / DBLE(nsym)
     !   
     ! bring vector back to cartesian axis
     !   
     DO na = 1, nat_sc
       DO nb = 1, nat_sc
         DO nc = 1, nat_sc
           DO nd = 1, nat_sc
             do alpha = 1, 3
               do beta = 1, 3
                 do gamm = 1, 3
                   do delt = 1, 3
                     v4(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) = 0.0d0
                     do i = 1, 3
                       do j = 1, 3
                         do k = 1, 3
                           do l = 1, 3
                             v4(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) = &
                                v4(3*(na-1)+alpha,3*(nb-1)+beta,3*(nc-1)+gamm,3*(nd-1)+delt) + &
                                work(3*(na-1)+i,3*(nb-1)+j,3*(nc-1)+k,3*(nd-1)+l) * &
                                bg_sc(alpha,i)*bg_sc(beta,j)*bg_sc(gamm,k)*bg_sc(delt,l)
                           end do
                         end do
                       end do
                     end do
                   end do
                 end do
               end do
             end do
           end do
         END DO
       END DO
     END DO
     !   
     DEALLOCATE (work)

  ! Deallocate stuff

  ! deallocate(phitot)
  ! deallocate(zeu)
  ! deallocate(rtau)
  ! deallocate(irt)

end subroutine sym_v4

! This subroutine creates replicas for one vector (i.e. forces or displacements)
! in the supercell based on the pointgroup symmetries. The subroutine inputs
! the vector for all the random configuration 
! and it outputs in a new array the vector replicated in the new structures

! subroutine sym_replica (v, at_sc, irt, s, nsym, tau_sc_latvec, trans_replica, vr) 

!   implicit none

!   double precision, dimension(:,:,:), intent(in) :: v   ! Input vector
!                                                         ! Dimension (n_random,nat_sc,3)
!   double precision, dimension(3,3), intent(in) :: at_sc ! Lattice vectors of the supercell
!   integer, dimension(:,:), intent(in) :: irt            ! Rotated atom by a symmetry operation
!   integer, dimension(3,3,48), intent(in) :: s           ! Symmetry operation matrix
!   integer, intent(in) :: nsym                           ! Number of symmetry operations
!   integer, dimension(:,:), allocatable :: tau_sc_latvec ! Tells which is the transformed atom by a particular translation
!   logical, intent(in) :: trans_replica                  ! Logical variable to determine if translational replica are included
!   double precision, dimension(:,:,:), intent(out) :: vr ! Output vector
!                                                         ! Dimension (n_random*nsym,nat_sc,3)
  
!   ! Work variables for the subroutine
!   integer :: n_random, nat_sc, ntrans
!   integer :: na, nar, i, alpha, ran, nr, isym, nt 
!   double precision, dimension(3,3) :: bg_sc
!   double precision, dimension(:,:), allocatable :: work1, work2
!   double precision, dimension(3) :: work_aux


!   ! Get integers

!   n_random = size (v(:,1,1))
!   nat_sc   = size (v(1,:,1))
!   ntrans   = size (tau_sc_latvec(1,:))
  
!   ! Allcoate arrays 

!   allocate ( work1 (nat_sc,3) )
!   allocate ( work2 (nat_sc,3) )

!   ! Create reciprocal lattice vectors of supercell

!   CALL recips(at_sc(1,1),at_sc(1,2),at_sc(1,3),bg_sc(1,1),bg_sc(1,2),bg_sc(1,3))

!   ! Do loop on random configurations and create replicas

!   nr = 0 ! Counter on replicas

!   do ran = 1, n_random
!     ! Now we create replicas based on point group symmetries
!     do isym = 1, nsym
!       ! Bring vector to crystal axis
!       work1 = 0.0d0
!       do na = 1, nat_sc
!         do alpha = 1, 3
!           work1(na,alpha) = 0.0d0
!           do i = 1, 3
!             work1(na,alpha) = work1(na,alpha) + v(ran,na,i) * at_sc(i,alpha)
!           end do
!         end do
!       end do
!       work2 = 0.0d0
!       ! Make replica
!       do na = 1, nat_sc
!         do alpha = 1, 3
!           nar = irt (isym, na)
!           do i = 1, 3
!             work2(na,alpha) = work2(na,alpha) + work1(nar,i) * s(alpha,i,isym)
!           end do
!         end do
!       end do
!       ! Bring replica to cartesian units
!       work1 = 0.0d0
!       do na = 1, nat_sc
!         do alpha = 1, 3
!           do i = 1, 3
!             work1(na,alpha) = work1(na,alpha) + work2(na,i) * bg_sc (alpha,i)
!           end do
!         end do 
!       end do      
!       ! Now we create replicas based on translations
!       if (trans_replica) then
!         do nt = 1, ntrans
!           nr = nr + 1
!           do na = 1, nat_sc
!             vr(nr,tau_sc_latvec(na,nt),:) =  work1(na,:)
!           end do
!         end do
!       else
!         nr = nr + 1
!         vr(nr,:,:) = work1(:,:)     
!       end if
!     end do
!   end do

!   ! Deallcoate arrays

!   deallocate ( work1, work2 )

! end subroutine sym_replica

! ! This subroutine creates replicas for one vector (i.e. forces or displacements)
! ! in the supercell based on the pointgroup symmetries. The subroutine inputs
! ! the vector for all the random configuration 
! ! and it outputs in a new array the vector replicated in the new structures

! subroutine sym_replica2 (v, at_sc, irt, s, nsym, tau_sc_latvec, trans_replica, vr) 

!   implicit none

!   double precision, dimension(:,:,:), intent(in) :: v   ! Input vector
!                                                         ! Dimension (n_random,nat_sc,3)
!   double precision, dimension(3,3), intent(in) :: at_sc ! Lattice vectors of the supercell
!   integer, dimension(:,:), intent(in) :: irt            ! Rotated atom by a symmetry operation
!   integer, dimension(3,3,48), intent(in) :: s           ! Symmetry operation matrix
!   integer, intent(in) :: nsym                           ! Number of symmetry operations
!   integer, dimension(:,:), allocatable :: tau_sc_latvec ! Tells which is the transformed atom by a particular translation
!   logical, intent(in) :: trans_replica                  ! Logical variable to determine if translational replica are included
!   double precision, dimension(:,:,:), intent(out) :: vr ! Output vector
!                                                         ! Dimension (n_random*nsym,nat_sc,3)
  
!   ! Work variables for the subroutine
!   integer :: n_random, nat_sc, ntrans
!   integer :: na, nar, i, alpha, ran, nr, isym, nt 
!   double precision, dimension(3,3) :: bg_sc
!   double precision, dimension(:,:), allocatable :: work1, work2
!   double precision, dimension(3) :: work_aux


!   ! Get integers

!   n_random = size (v(:,1,1))
!   nat_sc   = size (v(1,:,1))
!   ntrans   = size (tau_sc_latvec(1,:))
  
!   ! Allcoate arrays 

!   allocate ( work1 (nat_sc,3) )
!   allocate ( work2 (nat_sc,3) )

!   ! Create reciprocal lattice vectors of supercell

!   CALL recips(at_sc(1,1),at_sc(1,2),at_sc(1,3),bg_sc(1,1),bg_sc(1,2),bg_sc(1,3))

!   ! Do loop on random configurations and create replicas

!   nr = 0 ! Counter on replicas

!   do ran = 1, n_random
!     ! Now we create replicas based on point group symmetries
!     do isym = 1, nsym
!       ! Bring vector to crystal axis
!       work1 = 0.0d0
!       do na = 1, nat_sc
!         do alpha = 1, 3
!           work1(na,alpha) = 0.0d0
!           do i = 1, 3
!             work1(na,alpha) = work1(na,alpha) + v(ran,na,i) * bg_sc(i,alpha)
!           end do
!         end do
!       end do
!       work2 = 0.0d0
!       ! Make replica
!       do na = 1, nat_sc
!         do alpha = 1, 3
!           nar = irt (isym, na)
!           do i = 1, 3
!             work2(na,alpha) = work2(na,alpha) + work1(nar,i) * s(alpha,i,isym)
!           end do
!         end do
!       end do
!       ! Bring replica to cartesian units
!       work1 = 0.0d0
!       do na = 1, nat_sc
!         do alpha = 1, 3
!           do i = 1, 3
!             work1(na,alpha) = work1(na,alpha) + work2(na,i) * at_sc (alpha,i)
!           end do
!         end do 
!       end do      
!       ! Now we create replicas based on translations
!       if (trans_replica) then
!         do nt = 1, ntrans
!           nr = nr + 1
!           do na = 1, nat_sc
!             vr(nr,tau_sc_latvec(na,nt),:) =  work1(na,:)
!           end do
!         end do
!       else
!         nr = nr + 1
!         vr(nr,:,:) = work1(:,:)     
!       end if
!     end do
!   end do

!   ! Deallcoate arrays

!   deallocate ( work1, work2 )

! end subroutine sym_replica2


! This subroutine prints out the symmetries. It prints point group matrix
! and which atom is related to that in the unit cell

subroutine print_symm ( s, nsym, irt, supercell, nat)

  implicit none 

  integer, dimension(3,3,48), intent(in) :: s
  integer, intent(in) :: nsym
  integer, dimension(48,nat), intent(in) :: irt
  logical, intent(in) :: supercell
  !integer, dimension(nat), intent(in) :: itau

  integer :: isym, alpha, na
  integer :: nat

  !nat = size(irt(1,:))

  print *, ''
  print *, ' Printing symmetries... ' 
  print *, ''
  print '(a,i3)', '   Symmetries found : ', nsym
  print *, ''

  do isym = 1, nsym
    print '(a,i3)', '     Symmetry ', isym
    print '(a)',    '     -------- '
    print *, ''
    print *, '            point group matrix:'
    do alpha = 1, 3
      print '(3i3)', s(alpha,1:3,isym) 
    end do 
    print *, '            rotated atoms:'
    do na = 1, nat
      print '(i3,a,i3)', na , ' -> ', irt(isym,na)
    end do
    ! if ( supercell ) then
    !   print *, '            rotated atoms brought to unit cell:'
    !   do na = 1, nat
    !     print '(i3,a,i3)', itau(na) , ' -> ', itau(irt(isym,na))
    !   end do
    ! end if
  end do

end subroutine print_symm  
