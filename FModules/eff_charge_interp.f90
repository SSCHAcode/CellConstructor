subroutine rgd_blk (nr1,nr2,nr3,nat,dyn,q,tau,epsil,zeu,bg,omega,alat,loto_2d,sign)
   !-----------------------------------------------------------------------
   ! compute the rigid-ion (long-range) term for q
   ! The long-range term used here, to be added to or subtracted from the
   ! dynamical matrices, is exactly the same of the formula introduced in:
   ! X. Gonze et al, PRB 50. 13035 (1994) . Only the G-space term is
   ! implemented: the Ewald parameter alpha must be large enough to
   ! have negligible r-space contribution
   !
   use kinds, only: dp
   use constants, only: pi,tpi, fpi, e2
   implicit none
   integer ::  nr1, nr2, nr3    !  FFT grid
   integer ::  nat              ! number of atoms
   complex(DP) :: dyn(3,3,nat,nat) ! dynamical matrix
   real(DP) &
        q(3),           &! q-vector
        tau(3,nat),     &! atomic positions
        epsil(3,3),     &! dielectric constant tensor
        zeu(3,3,nat),   &! effective charges tensor
        at(3,3),        &! direct     lattice basis vectors
        bg(3,3),        &! reciprocal lattice basis vectors
        omega,          &! unit cell volume
        alat,           &! cell dimension units 
        sign             ! sign=+/-1.0 ==> add/subtract rigid-ion term
   logical :: loto_2d ! 2D LOTO correction 
   !
   ! local variables
   !
   real(DP):: geg, gp2, r                    !  <q+G| epsil | q+G>,  For 2d loto: gp2, r
   integer :: na,nb, i,j, m1, m2, m3
   integer :: nr1x, nr2x, nr3x
   real(DP) :: alph, fac,g1,g2,g3, facgd, arg, gmax, alat_new
   real(DP) :: zag(3),zbg(3),zcg(3), fnat(3), reff(2,2)
   real :: time1, time2
   complex(dp) :: facg
   !
   ! alph is the Ewald parameter, geg is an estimate of G^2
   ! such that the G-space sum is convergent for that alph
   ! very rough estimate: geg/4/alph > gmax = 14
   ! (exp (-14) = 10^-6)
   !
   call cpu_time(time1)
   gmax= 14.d0
   alph= 1.0d0
   geg = gmax*alph*4.0d0
 
    !print *, ""
    !print *, "[RGD_BLK] Q = ", q
    !print *, "[RGD_BLK] NAT:", nat
    !print *, "[RGD_BLK] OMEGA:", omega
    !print *, "[RGD_BLK] ZEU:"
    !do i = 1, nat
    !   do j = 1, 3
    !      print *, zeu(:, j, i)
    !   enddo
    !end do
    !print *, "[RGD_BLK] ALAT:", alat
    !print *, "[RGD_BLK] BG:"
    !do i = 1, 3
    !   print *, bg(:, i)
    !end do
    !print *, "[RGD_BLK] TAU:"
    !do i = 1, nat
    !   print *, tau(:, i)
    !end do

    !print *, "[RGD_BLK] EPSIL:"
    !do i = 1, 3
    !   print *, epsil(:, i)
    !end do
 
    !print *, "[RGD_BLK] DYN:"
    !do na = 1, nat
    !   do nb = 1, nat
    !      do i = 1, 3
    !         print *,  dyn(:, i, na, nb)
    !      end do
    !   end do
    !end do
   
   ! Silicon 
   alat_new = 10.0d0
   
 
   ! Estimate of nr1x,nr2x,nr3x generating all vectors up to G^2 < geg
   ! Only for dimensions where periodicity is present, e.g. if nr1=1
   ! and nr2=1, then the G-vectors run along nr3 only.
   ! (useful if system is in vacuum, e.g. 1D or 2D)
   !
   if (nr1 == 1) then
      nr1x=0
   else
      nr1x = int ( sqrt (geg) / &
                   (sqrt (bg (1, 1) **2 + bg (2, 1) **2 + bg (3, 1) **2) )) + 1
   endif
   if (nr2 == 1) then
      nr2x=0
   else
      nr2x = int ( sqrt (geg) / &
                   ( sqrt (bg (1, 2) **2 + bg (2, 2) **2 + bg (3, 2) **2) )) + 1
   endif
   if (nr3 == 1) then
      nr3x=0
   else
      nr3x = int ( sqrt (geg) / &
                   (sqrt (bg (1, 3) **2 + bg (2, 3) **2 + bg (3, 3) **2) )) + 1
   endif
   !
 
   !print *, "[RGD_BLK] integration grid:", nr1x, nr2x, nr3x
   if (abs(sign) /= 1.0_DP) &
        call errore ('rgd_blk',' wrong value for sign ',1)
   !
   IF (loto_2d) THEN 
      fac = sign*e2*fpi/omega*0.5d0*alat/bg(3,3)
      reff=0.0d0
      DO i=1,2
         DO j=1,2
            reff(i,j)=epsil(i,j)*0.5d0*tpi/bg(3,3) ! (eps)*c/2 in 2pi/a units
         ENDDO
      ENDDO
      DO i=1,2
         reff(i,i)=reff(i,i)-0.5d0*tpi/bg(3,3) ! (-1)*c/2 in 2pi/a units
      ENDDO 
   ELSE
     fac = sign*e2*fpi/omega
   ENDIF
   do m1 = -nr1x,nr1x
   do m2 = -nr2x,nr2x
   do m3 = -nr3x,nr3x
      !
      g1 = m1*bg(1,1) + m2*bg(1,2) + m3*bg(1,3)
      g2 = m1*bg(2,1) + m2*bg(2,2) + m3*bg(2,3)
      g3 = m1*bg(3,1) + m2*bg(3,2) + m3*bg(3,3)
      !
      IF (loto_2d) THEN 
         geg = g1**2 + g2**2 + g3**2
         r=0.0d0
         gp2=g1**2+g2**2
         IF (gp2>1.0d-8) THEN
            r=g1*reff(1,1)*g1+g1*reff(1,2)*g2+g2*reff(2,1)*g1+g2*reff(2,2)*g2
            r=r/gp2
         ENDIF
      ELSE
          geg = (g1*(epsil(1,1)*g1+epsil(1,2)*g2+epsil(1,3)*g3)+      &
             g2*(epsil(2,1)*g1+epsil(2,2)*g2+epsil(2,3)*g3)+      &
             g3*(epsil(3,1)*g1+epsil(3,2)*g2+epsil(3,3)*g3))
      ENDIF
      !
      if (geg > 0.0_DP .and. geg/alph/4.0_DP < gmax ) then
         !
         IF (loto_2d) THEN 
           facgd = fac*exp(-geg/alph/4.0d0)/SQRT(geg)/(1.0+r*SQRT(geg)) 
         ELSE
           facgd = fac*exp(-geg/alph/4.0d0)/geg
         ENDIF
         !
         do na = 1,nat
            zag(:)=g1*zeu(1,:,na)+g2*zeu(2,:,na)+g3*zeu(3,:,na)
            fnat(:) = 0.d0
            do nb = 1,nat
               arg = 2.d0*pi* (g1 * (tau(1,na)-tau(1,nb))+             &
                               g2 * (tau(2,na)-tau(2,nb))+             &
                               g3 * (tau(3,na)-tau(3,nb)))
               zcg(:) = g1*zeu(1,:,nb) + g2*zeu(2,:,nb) + g3*zeu(3,:,nb)
               fnat(:) = fnat(:) + zcg(:)*cos(arg)
            end do
            do j=1,3
               do i=1,3
                  dyn(i,j,na,na) = dyn(i,j,na,na) - facgd * &
                                   zag(i) * fnat(j)
               end do
            end do
         end do
      end if
      !
      g1 = g1 + q(1)
      g2 = g2 + q(2)
      g3 = g3 + q(3)
      !
      IF (loto_2d) THEN 
         geg = g1**2+g2**2+g3**2
         r=0.0d0
         gp2=g1**2+g2**2
         IF (gp2>1.0d-8) THEN
            r=g1*reff(1,1)*g1+g1*reff(1,2)*g2+g2*reff(2,1)*g1+g2*reff(2,2)*g2
            r=r/gp2
         ENDIF
      ELSE
      geg = (g1*(epsil(1,1)*g1+epsil(1,2)*g2+epsil(1,3)*g3)+      &
             g2*(epsil(2,1)*g1+epsil(2,2)*g2+epsil(2,3)*g3)+      &
             g3*(epsil(3,1)*g1+epsil(3,2)*g2+epsil(3,3)*g3))
      ENDIF
      !
      if (geg > 0.0_DP .and. geg/alph/4.0_DP < gmax ) then
         !
         IF (loto_2d) THEN 
           facgd = fac*exp(-geg/alph/4.0d0)/SQRT(geg)/(1.0+r*SQRT(geg))
         ELSE
           facgd = fac*exp(-geg/alph/4.0d0)/geg
         ENDIF
         !
         do nb = 1,nat
            zbg(:)=g1*zeu(1,:,nb)+g2*zeu(2,:,nb)+g3*zeu(3,:,nb)
            do na = 1,nat
               zag(:)=g1*zeu(1,:,na)+g2*zeu(2,:,na)+g3*zeu(3,:,na)
               arg = 2.d0*pi* (g1 * (tau(1,na)-tau(1,nb))+             &
                               g2 * (tau(2,na)-tau(2,nb))+             &
                               g3 * (tau(3,na)-tau(3,nb)))
               !
               facg = facgd * CMPLX(cos(arg),sin(arg),kind=DP)
               do j=1,3
                  do i=1,3
                     dyn(i,j,na,nb) = dyn(i,j,na,nb) + facg *      &
                                      zag(i) * zbg(j)
                  end do
               end do
            end do
         end do
      end if
   end do
   end do
 end do
 
!  call cpu_time(time2)
!  print *, "Elapsed time in rgd_blk: ", time2 - time1
 
 
!   print *, ""
!   print *, "[RGD_BLK] DYN FINAL:"
!   do na = 1, nat
!      do nb = 1, nat
!         do i = 1, 3
!            print *,  dyn(:, i, na, nb)
!         end do
!      end do
!   end do
 
!   print *, ""
!   print *, ""
!  print *, "----------------------------------------------------"
 
  
 
   !
   return
   !
 end subroutine rgd_blk
 

subroutine rgd_blk_diff_phase_conv (nr1,nr2,nr3,nat,dyn,q,tau,epsil,zeu,bg,omega,alat,loto_2d,sign)
   !-----------------------------------------------------------------------
   ! compute the rigid-ion (long-range) term for q
   ! The long-range term used here, to be added to or subtracted from the
   ! dynamical matrices, is exactly the same of the formula introduced in:
   ! X. Gonze et al, PRB 50. 13035 (1994) . Only the G-space term is
   ! implemented: the Ewald parameter alpha must be large enough to
   ! have negligible r-space contribution
   !
   use kinds, only: dp
   use constants, only: pi,tpi, fpi, e2
   implicit none
   integer ::  nr1, nr2, nr3    !  FFT grid
   integer ::  nat              ! number of atoms
   complex(DP) :: dyn(3,3,nat,nat) ! dynamical matrix
   real(DP) &
        q(3),           &! q-vector
        tau(3,nat),     &! atomic positions
        epsil(3,3),     &! dielectric constant tensor
        zeu(3,3,nat),   &! effective charges tensor
        at(3,3),        &! direct     lattice basis vectors
        bg(3,3),        &! reciprocal lattice basis vectors
        omega,          &! unit cell volume
        alat,           &! cell dimension units 
        sign             ! sign=+/-1.0 ==> add/subtract rigid-ion term
   logical :: loto_2d ! 2D LOTO correction 
   !
   ! local variables
   !
   real(DP):: geg, gp2, r                    !  <q+G| epsil | q+G>,  For 2d loto: gp2, r
   integer :: na,nb, i,j, m1, m2, m3
   integer :: nr1x, nr2x, nr3x
   real(DP) :: alph, fac,g1,g2,g3, facgd, arg, gmax, alat_new
   real(DP) :: zag(3),zbg(3),zcg(3), fnat(3), reff(2,2)
   real :: time1, time2
   complex(dp) :: facg
   !
   ! alph is the Ewald parameter, geg is an estimate of G^2
   ! such that the G-space sum is convergent for that alph
   ! very rough estimate: geg/4/alph > gmax = 14
   ! (exp (-14) = 10^-6)
   !
   call cpu_time(time1)
   gmax= 14.d0
   alph= 1.0d0
   geg = gmax*alph*4.0d0
 
    !print *, ""
    !print *, "[RGD_BLK] Q = ", q
    !print *, "[RGD_BLK] NAT:", nat
    !print *, "[RGD_BLK] OMEGA:", omega
    !print *, "[RGD_BLK] ZEU:"
    !do i = 1, nat
    !   do j = 1, 3
    !      print *, zeu(:, j, i)
    !   enddo
    !end do
    !print *, "[RGD_BLK] ALAT:", alat
    !print *, "[RGD_BLK] BG:"
    !do i = 1, 3
    !   print *, bg(:, i)
    !end do
    !print *, "[RGD_BLK] TAU:"
    !do i = 1, nat
    !   print *, tau(:, i)
    !end do

    !print *, "[RGD_BLK] EPSIL:"
    !do i = 1, 3
    !   print *, epsil(:, i)
    !end do
 
    !print *, "[RGD_BLK] DYN:"
    !do na = 1, nat
    !   do nb = 1, nat
    !      do i = 1, 3
    !         print *,  dyn(:, i, na, nb)
    !      end do
    !   end do
    !end do
   
   ! Silicon 
   alat_new = 10.0d0
   
 
   ! Estimate of nr1x,nr2x,nr3x generating all vectors up to G^2 < geg
   ! Only for dimensions where periodicity is present, e.g. if nr1=1
   ! and nr2=1, then the G-vectors run along nr3 only.
   ! (useful if system is in vacuum, e.g. 1D or 2D)
   !
   if (nr1 == 1) then
      nr1x=0
   else
      nr1x = int ( sqrt (geg) / &
                   (sqrt (bg (1, 1) **2 + bg (2, 1) **2 + bg (3, 1) **2) )) + 1
   endif
   if (nr2 == 1) then
      nr2x=0
   else
      nr2x = int ( sqrt (geg) / &
                   ( sqrt (bg (1, 2) **2 + bg (2, 2) **2 + bg (3, 2) **2) )) + 1
   endif
   if (nr3 == 1) then
      nr3x=0
   else
      nr3x = int ( sqrt (geg) / &
                   (sqrt (bg (1, 3) **2 + bg (2, 3) **2 + bg (3, 3) **2) )) + 1
   endif
   !
 
   !print *, "[RGD_BLK] integration grid:", nr1x, nr2x, nr3x
   if (abs(sign) /= 1.0_DP) &
        call errore ('rgd_blk',' wrong value for sign ',1)
   !
   IF (loto_2d) THEN 
      fac = sign*e2*fpi/omega*0.5d0*alat/bg(3,3)
      reff=0.0d0
      DO i=1,2
         DO j=1,2
            reff(i,j)=epsil(i,j)*0.5d0*tpi/bg(3,3) ! (eps)*c/2 in 2pi/a units
         ENDDO
      ENDDO
      DO i=1,2
         reff(i,i)=reff(i,i)-0.5d0*tpi/bg(3,3) ! (-1)*c/2 in 2pi/a units
      ENDDO 
   ELSE
     fac = sign*e2*fpi/omega
   ENDIF
   do m1 = -nr1x,nr1x
   do m2 = -nr2x,nr2x
   do m3 = -nr3x,nr3x
      !
      g1 = m1*bg(1,1) + m2*bg(1,2) + m3*bg(1,3)
      g2 = m1*bg(2,1) + m2*bg(2,2) + m3*bg(2,3)
      g3 = m1*bg(3,1) + m2*bg(3,2) + m3*bg(3,3)
      !
      IF (loto_2d) THEN 
         geg = g1**2 + g2**2 + g3**2
         r=0.0d0
         gp2=g1**2+g2**2
         IF (gp2>1.0d-8) THEN
            r=g1*reff(1,1)*g1+g1*reff(1,2)*g2+g2*reff(2,1)*g1+g2*reff(2,2)*g2
            r=r/gp2
         ENDIF
      ELSE
          geg = (g1*(epsil(1,1)*g1+epsil(1,2)*g2+epsil(1,3)*g3)+      &
             g2*(epsil(2,1)*g1+epsil(2,2)*g2+epsil(2,3)*g3)+      &
             g3*(epsil(3,1)*g1+epsil(3,2)*g2+epsil(3,3)*g3))
      ENDIF
      !
      if (geg > 0.0_DP .and. geg/alph/4.0_DP < gmax ) then
         !
         IF (loto_2d) THEN 
           facgd = fac*exp(-geg/alph/4.0d0)/SQRT(geg)/(1.0+r*SQRT(geg)) 
         ELSE
           facgd = fac*exp(-geg/alph/4.0d0)/geg
         ENDIF
         !
         do na = 1,nat
            zag(:)=g1*zeu(1,:,na)+g2*zeu(2,:,na)+g3*zeu(3,:,na)
            fnat(:) = 0.d0
            do nb = 1,nat
               arg = -2.d0*pi* (g1 * (tau(1,na)-tau(1,nb))+             &
                               g2 * (tau(2,na)-tau(2,nb))+             &
                               g3 * (tau(3,na)-tau(3,nb)))
               zcg(:) = g1*zeu(1,:,nb) + g2*zeu(2,:,nb) + g3*zeu(3,:,nb)
               fnat(:) = fnat(:) + zcg(:)*cos(arg)
            end do
            do j=1,3
               do i=1,3
                  dyn(i,j,na,na) = dyn(i,j,na,na) - facgd * &
                                   zag(i) * fnat(j)
               end do
            end do
         end do
      end if
      !
      g1 = g1 +q(1)
      g2 = g2 +q(2)
      g3 = g3 +q(3)
      !
      IF (loto_2d) THEN 
         geg = g1**2+g2**2+g3**2
         r=0.0d0
         gp2=g1**2+g2**2
         IF (gp2>1.0d-8) THEN
            r=g1*reff(1,1)*g1+g1*reff(1,2)*g2+g2*reff(2,1)*g1+g2*reff(2,2)*g2
            r=r/gp2
         ENDIF
      ELSE
      geg = (g1*(epsil(1,1)*g1+epsil(1,2)*g2+epsil(1,3)*g3)+      &
             g2*(epsil(2,1)*g1+epsil(2,2)*g2+epsil(2,3)*g3)+      &
             g3*(epsil(3,1)*g1+epsil(3,2)*g2+epsil(3,3)*g3))
      ENDIF
      !
      if (geg > 0.0_DP .and. geg/alph/4.0_DP < gmax ) then
         !
         IF (loto_2d) THEN 
           facgd = fac*exp(-geg/alph/4.0d0)/SQRT(geg)/(1.0+r*SQRT(geg))
         ELSE
           facgd = fac*exp(-geg/alph/4.0d0)/geg
         ENDIF
         !
         do nb = 1,nat
            zbg(:)=g1*zeu(1,:,nb)+g2*zeu(2,:,nb)+g3*zeu(3,:,nb)
            do na = 1,nat
               zag(:)=g1*zeu(1,:,na)+g2*zeu(2,:,na)+g3*zeu(3,:,na)
               arg = -2.d0*pi* ((g1 - q(1)) * (tau(1,na)-tau(1,nb))+             &
                               (g2 - q(2)) * (tau(2,na)-tau(2,nb))+             &
                               (g3 - q(3)) * (tau(3,na)-tau(3,nb)))
               !
               facg = facgd * CMPLX(cos(arg),sin(arg),kind=DP)
               do j=1,3
                  do i=1,3
                     dyn(i,j,na,nb) = dyn(i,j,na,nb) + facg *      &
                                      zag(i) * zbg(j)
                  end do
               end do
            end do
         end do
      end if
   end do
   end do
 end do
 
!  call cpu_time(time2)
!  print *, "Elapsed time in rgd_blk: ", time2 - time1
 
 
!   print *, ""
!   print *, "[RGD_BLK] DYN FINAL:"
!   do na = 1, nat
!      do nb = 1, nat
!         do i = 1, 3
!            print *,  dyn(:, i, na, nb)
!         end do
!      end do
!   end do
 
!   print *, ""
!   print *, ""
!  print *, "----------------------------------------------------"
 
  
 
   !
   return
   !
 end subroutine rgd_blk_diff_phase_conv

! subroutine rgd_blk (nr1,nr2,nr3,nat,dyn,q,tau,epsil,zeu,bg,omega,alat,loto_2d,sign)
!    !-----------------------------------------------------------------------
!    ! compute the rigid-ion (long-range) term for q
!    ! The long-range term used here, to be added to or subtracted from the
!    ! dynamical matrices, is exactly the same of the formula introduced in:
!    ! X. Gonze et al, PRB 50. 13035 (1994) . Only the G-space term is
!    ! implemented: the Ewald parameter alpha must be large enough to
!    ! have negligible r-space contribution
!    !
!    use kinds, only: dp
!    use constants, only: pi,tpi, fpi, e2
!    implicit none
!    integer ::  nr1, nr2, nr3    !  FFT grid
!    integer ::  nat              ! number of atoms
!    complex(DP) :: dyn(3,3,nat,nat) ! dynamical matrix
!    real(DP) &
!         q(3),           &! q-vector
!         tau(3,nat),     &! atomic positions
!         epsil(3,3),     &! dielectric constant tensor
!         zeu(3,3,nat),   &! effective charges tensor
!         at(3,3),        &! direct     lattice basis vectors
!         bg(3,3),        &! reciprocal lattice basis vectors
!         omega,          &! unit cell volume
!         alat,           &! cell dimension units (The dimension of the first unit cell vector in Bohr)
!         sign             ! sign=+/-1.0 ==> add/subtract rigid-ion term
!    logical :: loto_2d ! 2D LOTO correction 
!    !
!    ! local variables
!    !
!    real(DP):: geg, gp2, r                    !  <q+G| epsil | q+G>,  For 2d loto: gp2, r
!    integer :: na,nb, i,j, m1, m2, m3
!    integer :: nr1x, nr2x, nr3x
!    real(DP) :: alph, fac,g1,g2,g3, facgd, arg, gmax, alat_new
!    real(DP) :: zag(3),zbg(3),zcg(3), fnat(3), reff(2,2)
!    real :: time1, time2
!    complex(dp) :: facg
!    !
!    ! alph is the Ewald parameter, geg is an estimate of G^2
!    ! such that the G-space sum is convergent for that alph
!    ! very rough estimate: geg/4/alph > gmax = 14
!    ! (exp (-14) = 10^-6)
!    !
!    call cpu_time(time1)
!    gmax= 14.d0
!    alph= 1.0d0
!    geg = gmax*alph*4.0d0
 
!    print *, ""
!    print *, "[RGD_BLK] Q = ", q
!    print *, "[RGD_BLK] NAT:", nat
!    print *, "[RGD_BLK] OMEGA:", omega
!    print *, "[RGD_BLK] ZEU:"
!    do i = 1, nat
!       do j = 1, 3
!          print *, zeu(:, j, i)
!       enddo
!    end do
!    print *, "[RGD_BLK] ALAT:", alat
!    print *, "[RGD_BLK] BG:"
!    do i = 1, 3
!       print *, bg(:, i)
!    end do
!    print *, "[RGD_BLK] TAU:"
!    do i = 1, nat
!       print *, tau(:, i)
!    end do
!    print *, "[RGD_BLK] EPSIL:"
!    do i = 1, 3
!       print *, epsil(:, i)
!    end do
 
!    print *, "[RGD_BLK] DYN:"
!    do na = 1, nat
!       do nb = 1, nat
!          do i = 1, 3
!             print *,  dyn(:, i, na, nb)
!          end do
!       end do
!    end do

!    ! Silicon example
!    alat_new = 10.0d0
   
   
 
!    ! Estimate of nr1x,nr2x,nr3x generating all vectors up to G^2 < geg
!    ! Only for dimensions where periodicity is present, e.g. if nr1=1
!    ! and nr2=1, then the G-vectors run along nr3 only.
!    ! (useful if system is in vacuum, e.g. 1D or 2D)
!    !
!    if (nr1 == 1) then
!       nr1x=0
!    else
!       nr1x = int ( sqrt (geg) / &
!                    (alat_new * sqrt (bg (1, 1) **2 + bg (2, 1) **2 + bg (3, 1) **2) ) ) + 1
!    endif
!    if (nr2 == 1) then
!       nr2x=0
!    else
!       nr2x = int ( sqrt (geg)  / &
!                    (alat_new * sqrt (bg (1, 2) **2 + bg (2, 2) **2 + bg (3, 2) **2) )) + 1
!    endif
!    if (nr3 == 1) then
!       nr3x=0
!    else
!       nr3x = int ( sqrt (geg)  / &
!                    (alat_new * sqrt (bg (1, 3) **2 + bg (2, 3) **2 + bg (3, 3) **2) )) + 1
!    endif
!    !

!    print *, "[RGD_BLK] integration grid:", nr1x, nr2x, nr3x
!    call flush()


!    if (abs(sign) /= 1.0_DP) &
!         call errore ('rgd_blk',' wrong value for sign ',1)
!    !
!    IF (loto_2d) THEN 
!       fac = sign*e2*fpi/omega*0.5d0/(alat * bg(3,3))
!       reff=0.0d0
!       DO i=1,2
!          DO j=1,2
!             reff(i,j)=epsil(i,j)*0.5d0*tpi/(alat * bg(3,3)) ! (eps)*c/2 in 2pi/a units
!          ENDDO
!       ENDDO
!       DO i=1,2
!          reff(i,i)=reff(i,i)-0.5d0*tpi/(alat * bg(3,3)) ! (-1)*c/2 in 2pi/a units
!       ENDDO 
!    ELSE
!      fac = sign*e2*fpi/omega
!    ENDIF
!    do m1 = -nr1x,nr1x
!    do m2 = -nr2x,nr2x
!    do m3 = -nr3x,nr3x
!       !
!       g1 = m1*bg(1,1) + m2*bg(1,2) + m3*bg(1,3)
!       g2 = m1*bg(2,1) + m2*bg(2,2) + m3*bg(2,3)
!       g3 = m1*bg(3,1) + m2*bg(3,2) + m3*bg(3,3)
!       !
!       IF (loto_2d) THEN 
!          geg = g1**2 + g2**2 + g3**2
!          r=0.0d0
!          gp2=g1**2+g2**2
!          IF (gp2>1.0d-8) THEN
!             r=g1*reff(1,1)*g1+g1*reff(1,2)*g2+g2*reff(2,1)*g1+g2*reff(2,2)*g2
!             r=r/gp2
!          ENDIF
!       ELSE
!           geg = (g1*(epsil(1,1)*g1+epsil(1,2)*g2+epsil(1,3)*g3)+      &
!              g2*(epsil(2,1)*g1+epsil(2,2)*g2+epsil(2,3)*g3)+      &
!              g3*(epsil(3,1)*g1+epsil(3,2)*g2+epsil(3,3)*g3))
!       ENDIF
!       !
!       if (geg > 0.0_DP .and. geg /alph/4.0_DP < gmax ) then
!          !
!          IF (loto_2d) THEN 
!            facgd = fac*exp(-geg/alph/4.0d0)/SQRT(geg)/(1.0+r*SQRT(geg)) 
!          ELSE
!            facgd = fac*exp(-geg/alph/4.0d0)/geg
!          ENDIF
!          !
!          do na = 1,nat
!             zag(:)=g1*zeu(1,:,na)+g2*zeu(2,:,na)+g3*zeu(3,:,na)
!             fnat(:) = 0.d0
!             do nb = 1,nat
!                arg = 2.d0*pi* (g1 * (tau(1,na)-tau(1,nb))+             &
!                                g2 * (tau(2,na)-tau(2,nb))+             &
!                                g3 * (tau(3,na)-tau(3,nb)))
!                zcg(:) = g1*zeu(1,:,nb) + g2*zeu(2,:,nb) + g3*zeu(3,:,nb)
!                fnat(:) = fnat(:) + zcg(:)*cos(arg)
!             end do
!             do j=1,3
!                do i=1,3
!                   dyn(i,j,na,na) = dyn(i,j,na,na) - facgd * &
!                                    zag(i) * fnat(j)
!                end do
!             end do
!          end do
!       end if
!       !
!       g1 = g1 + q(1)
!       g2 = g2 + q(2)
!       g3 = g3 + q(3)
!       !
!       IF (loto_2d) THEN 
!          geg = g1**2+g2**2+g3**2
!          r=0.0d0
!          gp2=g1**2+g2**2
!          IF (gp2>1.0d-8) THEN
!             r=g1*reff(1,1)*g1+g1*reff(1,2)*g2+g2*reff(2,1)*g1+g2*reff(2,2)*g2
!             r=r/gp2
!          ENDIF
!       ELSE
!       geg = (g1*(epsil(1,1)*g1+epsil(1,2)*g2+epsil(1,3)*g3)+      &
!              g2*(epsil(2,1)*g1+epsil(2,2)*g2+epsil(2,3)*g3)+      &
!              g3*(epsil(3,1)*g1+epsil(3,2)*g2+epsil(3,3)*g3))
!       ENDIF
!       !
!       if (geg > 0.0_DP .and. geg/alph/4.0_DP < gmax ) then
!          !
!          IF (loto_2d) THEN 
!            facgd = fac*exp(-geg/alph/4.0d0)/SQRT(geg)/(1.0+r*SQRT(geg))
!          ELSE
!            facgd = fac*exp(-geg/alph/4.0d0)/geg
!          ENDIF
!          !
!          do nb = 1,nat
!             zbg(:)=g1*zeu(1,:,nb)+g2*zeu(2,:,nb)+g3*zeu(3,:,nb)
!             do na = 1,nat
!                zag(:)=g1*zeu(1,:,na)+g2*zeu(2,:,na)+g3*zeu(3,:,na)
!                arg = 2.d0*pi* (g1 * (tau(1,na)-tau(1,nb))+             &
!                                g2 * (tau(2,na)-tau(2,nb))+             &
!                                g3 * (tau(3,na)-tau(3,nb)))
!                !
!                facg = facgd * CMPLX(cos(arg),sin(arg),kind=DP)
!                do j=1,3
!                   do i=1,3
!                      dyn(i,j,na,nb) = dyn(i,j,na,nb) + facg *      &
!                                       zag(i) * zbg(j)
!                   end do
!                end do
!             end do
!          end do
!       end if
!    end do
!    end do
!  end do
 
!  call cpu_time(time2)
!  print *, "Elapsed time in rgd_blk: ", time2 - time1


!  print *, ""
!  print *, "[RGD_BLK] DYN FINAL:"
!  do na = 1, nat
!     do nb = 1, nat
!        do i = 1, 3
!           print *,  dyn(:, i, na, nb)
!        end do
!     end do
!  end do

!  print *, ""
!  print *, ""
!  print *, "----------------------------------------------------"

 
!    !
!    return
!    !
!  end subroutine rgd_blk
 


!-----------------------------------------------------------------------
subroutine nonanal(nat, nat_blk, itau_blk, epsil, q, zeu, omega, dyn )
  !-----------------------------------------------------------------------
  !     add the nonanalytical term with macroscopic electric fields
  !     See PRB 55, 10355 (1997) Eq (60)
  !
  use kinds, only: dp
  use constants, only: pi, fpi, e2
 implicit none
 integer, intent(in) :: nat, nat_blk, itau_blk(nat)
 !  nat: number of atoms in the cell (in the supercell in the case
 !       of a dyn.mat. constructed in the mass approximation)
 !  nat_blk: number of atoms in the original cell (the same as nat if
 !       we are not using the mass approximation to build a supercell)
 !  itau_blk(na): atom in the original cell corresponding to
 !                atom na in the supercell
 !
 complex(DP), intent(inout) :: dyn(3,3,nat,nat) ! dynamical matrix
 real(DP), intent(in) :: q(3),  &! polarization vector
      &       epsil(3,3),     &! dielectric constant tensor
      &       zeu(3,3,nat_blk),   &! effective charges tensor
      &       omega            ! unit cell volume
 !
 ! local variables
 !
 real(DP) zag(3),zbg(3),  &! eff. charges  times g-vector
      &       qeq              !  <q| epsil | q>
 integer na,nb,              &! counters on atoms
      &  na_blk,nb_blk,      &! as above for the original cell
      &  i,j                  ! counters on cartesian coordinates
 !
 qeq = (q(1)*(epsil(1,1)*q(1)+epsil(1,2)*q(2)+epsil(1,3)*q(3))+    &
        q(2)*(epsil(2,1)*q(1)+epsil(2,2)*q(2)+epsil(2,3)*q(3))+    &
        q(3)*(epsil(3,1)*q(1)+epsil(3,2)*q(2)+epsil(3,3)*q(3)))
 !
!print*, q(1), q(2), q(3)
 if (qeq < 1.d-8) then
    write(6,'(5x,"A direction for q was not specified:", &
      &          "TO-LO splitting will be absent")')
    return
 end if
 !
 do na = 1,nat
    na_blk = itau_blk(na)
    do nb = 1,nat
       nb_blk = itau_blk(nb)
       !
       do i=1,3
          !
          zag(i) = q(1)*zeu(1,i,na_blk) +  q(2)*zeu(2,i,na_blk) + &
                   q(3)*zeu(3,i,na_blk)
          zbg(i) = q(1)*zeu(1,i,nb_blk) +  q(2)*zeu(2,i,nb_blk) + &
                   q(3)*zeu(3,i,nb_blk)
       end do
       !
       do i = 1,3
          do j = 1,3
             dyn(i,j,na,nb) = dyn(i,j,na,nb)+ fpi*e2*zag(i)*zbg(j)/qeq/omega
!             print*, zag(i),zbg(j),qeq, fpi*e2*zag(i)*zbg(j)/qeq/omega
          end do
       end do
    end do
 end do
 !
 return
end subroutine nonanal


SUBROUTINE trasl( phid, phiq, nq, nr1, nr2, nr3, nat, m1, m2, m3 )
   !----------------------------------------------------------------------------
   !
   USE kinds, ONLY : DP
   !
   IMPLICIT NONE
   INTEGER, intent(in) ::  nr1, nr2, nr3, m1, m2, m3, nat, nq
   COMPLEX(DP), intent(in) :: phiq(3,3,nat,nat,48)
   COMPLEX(DP), intent(out) :: phid(nr1,nr2,nr3,3,3,nat,nat)
   !
   INTEGER :: j1,j2,  na1, na2
   !
   DO j1=1,3
      DO j2=1,3
         DO na1=1,nat
            DO na2=1,nat
               phid(m1,m2,m3,j1,j2,na1,na2) = &
                    0.5d0 * (      phiq(j1,j2,na1,na2,nq) +  &
                           CONJG(phiq(j2,j1,na2,na1,nq)))
            END DO
         END DO
      END DO
   END DO
   !
   RETURN
 END SUBROUTINE trasl
 
