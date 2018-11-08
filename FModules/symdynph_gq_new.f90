!
! Copyright (C) 2001-2012 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------

! Source edited from the original QE version
! to be wrapped into a python library
subroutine symdynph_gq_new (xq, phi, s, invs, rtau, irt, irotmq, minus_q, &
     nsymq, nat)
  !-----------------------------------------------------------------------
  !
  !     This routine receives as input an unsymmetrized dynamical
  !     matrix expressed on the crystal axes and imposes the symmetry
  !     of the small group of q. Furthermore it imposes also the symmetry
  !     q -> -q+G if present.
  !
  !
  implicit none
  !
  !    The dummy variables
  !
  integer, intent(in) :: nat, nsymq, irotmq
  integer, intent(in) :: s (3, 3, 48), irt (48, nat), invs (48)
  ! input: the number of atoms
  ! input: the symmetry matrices
  ! input: the rotated of each vector
  ! input: the small group of q
  ! input: the inverse of each matrix
  ! input: the order of the small gro
  ! input: the rotation sending q ->
  double precision, intent(in) :: xq (3), rtau (3, 48, nat)
  ! input: the q point
  ! input: the R associated at each t

  logical, intent(in) :: minus_q
  ! input: true if a symmetry q->-q+G
  
  double complex, intent(inout) :: phi (3, 3, nat, nat)
  ! inp/out: the matrix to symmetrize
  !
  !   local variables
  !
  integer :: isymq, sna, snb, irot, na, nb, ipol, jpol, lpol, kpol, &
       iflb (nat, nat)
  ! counters, indices, work space

  double precision :: arg
  ! the argument of the phase

  double complex :: phip (3, 3, nat, nat), work (3, 3), fase, faseq (nsymq)

  ! Define the 2pi constant
  double precision :: tpi
  tpi = 6.283185307179586


  ! Print all the input variables
  ! print *, "-------------------------------------"
  ! print *, "xq:", xq
  ! print *, "PHI:"
  ! do na = 1, nat
  !    do nb = 1, nat
  !       print *, na, nb
  !       do jpol = 1, 3
  !          print *, phi(:, jpol, na, nb)
  !       end do
  !    end do
  ! end do
  ! print *, "N sym:", nsymq
  ! print *, "Symmetries:"
  ! do isymq = 1, nsymq
  !    print *, "  ", isymq
  !    do jpol = 1, 3
  !       print *, s(:, jpol, isymq)
  !    end do
  !    print *, "irt:", irt(isymq, :)
  ! end do
  ! print *, "INVS:"
  ! print *, invs(:)
  ! print *, "MINUS Q:", minus_q, "IROTMQ:", irotmq
  ! print *, "-------------------------------------"
          
  
  
  ! work space, phase factors
  !
  !    We start by imposing hermiticity
  !
  do na = 1, nat
     do nb = 1, nat
        do ipol = 1, 3
           do jpol = 1, 3
              phi (ipol, jpol, na, nb) = 0.5d0 * (phi (ipol, jpol, na, nb) &
                   + CONJG(phi (jpol, ipol, nb, na) ) )
              phi (jpol, ipol, nb, na) = CONJG(phi (ipol, jpol, na, nb) )
           enddo
        enddo
     enddo
  enddo
  !
  !    If no other symmetry is present we quit here
  !
  if ( (nsymq == 1) .and. (.not.minus_q) ) return
  !
  !    Then we impose the symmetry q -> -q+G if present
  !
  if (minus_q) then
     do na = 1, nat
        do nb = 1, nat
           do ipol = 1, 3
              do jpol = 1, 3
                 work(:,:) = (0.d0, 0.d0)
                 sna = irt (irotmq, na)
                 snb = irt (irotmq, nb)
                 arg = 0.d0
                 do kpol = 1, 3
                    arg = arg + (xq (kpol) * (rtau (kpol, irotmq, na) - &
                                              rtau (kpol, irotmq, nb) ) )
                 enddo
                 arg = arg * tpi
                 fase = DCMPLX(cos (arg), sin (arg))
                 do kpol = 1, 3
                    do lpol = 1, 3
                       work (ipol, jpol) = work (ipol, jpol) + &
                            s (ipol, kpol, irotmq) * s (jpol, lpol, irotmq) &
                            * phi (kpol, lpol, sna, snb) * fase
                    enddo
                 enddo
                 phip (ipol, jpol, na, nb) = (phi (ipol, jpol, na, nb) + &
                      CONJG( work (ipol, jpol) ) ) * 0.5d0
              enddo
           enddo
        enddo
     enddo
     phi = phip
  endif

  !
  !    Here we symmetrize with respect to the small group of q
  !
  if (nsymq == 1) return

  iflb (:, :) = 0
  do na = 1, nat
     do nb = 1, nat
        if (iflb (na, nb) == 0) then
           work(:,:) = (0.d0, 0.d0)
           do isymq = 1, nsymq
              ! Print the symmetry
              ! if (na .eq. 1 .and. nb .eq. 1) then
              !    print *, "Symmetry", isymq
              !    print *, s(:, 1, isymq)
              !    print *, s(:, 2, isymq)
              !    print *, s(:, 3, isymq)
              !    print *, ""
              ! end if
              
              
              irot = isymq
              sna = irt (irot, na)
              snb = irt (irot, nb)
              arg = 0.d0
              do ipol = 1, 3
                 arg = arg + (xq (ipol) * (rtau (ipol, irot, na) - &
                                           rtau (ipol, irot, nb) ) )
              enddo
              arg = arg * tpi
              faseq (isymq) = DCMPLX(cos (arg), sin (arg))
              do ipol = 1, 3
                 do jpol = 1, 3
                    do kpol = 1, 3
                       do lpol = 1, 3
                          work (ipol, jpol) = work (ipol, jpol) + &
                               s (ipol, kpol, irot) * s (jpol, lpol, irot) &
                               * phi (kpol, lpol, sna, snb) * faseq (isymq)
                       enddo
                    enddo
                 enddo
              enddo
           enddo

           ! ! Print the Phi matrix
           ! print "(A10, I8,A10,I8)", "NA = ", na, "NB =", nb
           ! print *, "Phi:"
           ! print *, phi(:, 1, na, nb)
           ! print *, phi(:, 2, na, nb)
           ! print *, phi(:, 3, na, nb)

           ! print *, ""
           ! print *, "Work:"
           ! print *, work(:, 1)
           ! print *, work(:, 2)
           ! print *, work(:, 3)
           
           do isymq = 1, nsymq
              irot = isymq
              sna = irt (irot, na)
              snb = irt (irot, nb)
              do ipol = 1, 3
                 do jpol = 1, 3
                    phi (ipol, jpol, sna, snb) = (0.d0, 0.d0)
                    do kpol = 1, 3
                       do lpol = 1, 3
                          phi (ipol, jpol, sna, snb) = phi (ipol, jpol, sna, snb) &
                               + s (ipol, kpol, invs (irot) ) * s (jpol, lpol, invs (irot) ) &
                               * work (kpol, lpol) * CONJG(faseq (isymq) )
                       enddo
                    enddo
                 enddo
              enddo
              iflb (sna, snb) = 1
           enddo

        endif
     enddo
  enddo
  phi (:, :, :, :) = phi (:, :, :, :) / DBLE(nsymq)

  
  ! print *, "OUT PHI:"
  ! do na = 1, nat
  !    do nb = 1, nat
  !       print *, na, nb
  !       do jpol = 1, 3
  !          print *, phi(:, jpol, na, nb)
  !       end do
  !    end do
  ! end do
  
  return
end subroutine symdynph_gq_new
