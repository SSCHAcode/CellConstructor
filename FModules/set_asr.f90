!
! Copyright (C) 2001-2012 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------
subroutine set_asr ( asr, axis, nat, tau, dyn, zeu )
  !-----------------------------------------------------------------------
  !
  !  Impose ASR - refined version by Nicolas Mounet
  !
  implicit none
  character(len=10), intent(in) :: asr
  integer, intent(in) :: axis, nat
  double precision, intent(in) :: tau(3,nat)
  double precision, intent(inout) :: zeu(3,3,nat)
  double complex, intent(inout) :: dyn(3,3,nat,nat)
  !
  integer :: i,j,n,m,p,k,l,q,r,na, nb, na1, i1, j1
  double precision, allocatable:: dynr_new(:,:,:,:,:), zeu_new(:,:,:)
  double precision, allocatable :: u(:,:,:,:,:)
  ! These are the "vectors" associated with the sum rules
  !
  integer u_less(6*3*nat),n_less,i_less
  ! indices of the vectors u that are not independent to the preceding ones,
  ! n_less = number of such vectors, i_less = temporary parameter
  !
  integer, allocatable :: ind_v(:,:,:)
  double precision, allocatable :: v(:,:)
  ! These are the "vectors" associated with symmetry conditions, coded by
  ! indicating the positions (i.e. the four indices) of the non-zero elements
  ! (there should be only 2 of them) and the value of that element.
  ! We do so in order to use limit the amount of memory used.
  !
  double precision, allocatable :: w(:,:,:,:), x(:,:,:,:)
  double precision sum, scal, norm2
  ! temporary vectors and parameters
  !
  double precision, allocatable :: zeu_u(:,:,:,:)
  ! These are the "vectors" associated with the sum rules on effective charges
  !
  integer zeu_less(6*3),nzeu_less,izeu_less
  ! indices of the vectors zeu_u that are not independent to the preceding
  ! ones, nzeu_less = number of such vectors, izeu_less = temporary parameter
  !
  double precision, allocatable :: zeu_w(:,:,:), zeu_x(:,:,:)
  ! temporary vectors
  !
  ! Initialization
  ! n is the number of sum rules to be considered (if asr.ne.'simple')
  ! 'axis' is the rotation axis in the case of a 1D system (i.e. the rotation
  !  axis is (Ox) if axis='1', (Oy) if axis='2' and (Oz) if axis='3')
  !
  if ( (asr.ne.'simple') .and. (asr.ne.'crystal') .and. (asr.ne.'one-dim') &
                         .and.(asr.ne.'zero-dim')) then
     call errore('set_asr','invalid Acoustic Sum Rule:' // asr, 1)
  endif
  if(asr.eq.'crystal') n=3
  if(asr.eq.'one-dim') then
     write(6,'("asr rotation axis in 1D system= ",I4)') axis
     n=4
  endif
  if(asr.eq.'zero-dim') n=6
  !
  ! ASR on effective charges
  !
  if(asr.eq.'simple') then
     do i=1,3
        do j=1,3
           sum=0.0d0
           do na=1,nat
              sum = sum + zeu(i,j,na)
           end do
           do na=1,nat
              zeu(i,j,na) = zeu(i,j,na) - sum/nat
           end do
        end do
     end do
  else
     ! generating the vectors of the orthogonal of the subspace to project
     ! the effective charges matrix on
     !
     allocate ( zeu_new(3,3,nat) )
     allocate (zeu_u(6*3,3,3,nat) )
     zeu_u(:,:,:,:)=0.0d0
     do i=1,3
        do j=1,3
           do na=1,nat
              zeu_new(i,j,na)=zeu(i,j,na)
           enddo
        enddo
     enddo
     !
     p=0
     do i=1,3
        do j=1,3
           ! These are the 3*3 vectors associated with the
           ! translational acoustic sum rules
           p=p+1
           zeu_u(p,i,j,:)=1.0d0
           !
        enddo
     enddo
     !
     if (n.eq.4) then
        do i=1,3
           ! These are the 3 vectors associated with the
           ! single rotational sum rule (1D system)
           p=p+1
           do na=1,nat
              zeu_u(p,i,MOD(axis,3)+1,na)=-tau(MOD(axis+1,3)+1,na)
              zeu_u(p,i,MOD(axis+1,3)+1,na)=tau(MOD(axis,3)+1,na)
           enddo
           !
        enddo
     endif
     !
     if (n.eq.6) then
        do i=1,3
           do j=1,3
              ! These are the 3*3 vectors associated with the
              ! three rotational sum rules (0D system - typ. molecule)
              p=p+1
              do na=1,nat
                 zeu_u(p,i,MOD(j,3)+1,na)=-tau(MOD(j+1,3)+1,na)
                 zeu_u(p,i,MOD(j+1,3)+1,na)=tau(MOD(j,3)+1,na)
              enddo
              !
           enddo
        enddo
     endif
     !
     ! Gram-Schmidt orthonormalization of the set of vectors created.
     !
     allocate ( zeu_w(3,3,nat), zeu_x(3,3,nat) )
     nzeu_less=0
     do k=1,p
        zeu_w(:,:,:)=zeu_u(k,:,:,:)
        zeu_x(:,:,:)=zeu_u(k,:,:,:)
        do q=1,k-1
           r=1
           do izeu_less=1,nzeu_less
              if (zeu_less(izeu_less).eq.q) r=0
           enddo
           if (r.ne.0) then
              call sp_zeu(zeu_x,zeu_u(q,:,:,:),nat,scal)
              zeu_w(:,:,:) = zeu_w(:,:,:) - scal* zeu_u(q,:,:,:)
           endif
        enddo
        call sp_zeu(zeu_w,zeu_w,nat,norm2)
        if (norm2.gt.1.0d-16) then
           zeu_u(k,:,:,:) = zeu_w(:,:,:) / DSQRT(norm2)
        else
           nzeu_less=nzeu_less+1
           zeu_less(nzeu_less)=k
        endif
     enddo
     !
     !
     ! Projection of the effective charge "vector" on the orthogonal of the
     ! subspace of the vectors verifying the sum rules
     !
     zeu_w(:,:,:)=0.0d0
     do k=1,p
        r=1
        do izeu_less=1,nzeu_less
           if (zeu_less(izeu_less).eq.k) r=0
        enddo
        if (r.ne.0) then
           zeu_x(:,:,:)=zeu_u(k,:,:,:)
           call sp_zeu(zeu_x,zeu_new,nat,scal)
           zeu_w(:,:,:) = zeu_w(:,:,:) + scal*zeu_u(k,:,:,:)
        endif
     enddo
     !
     ! Final substraction of the former projection to the initial zeu, to get
     ! the new "projected" zeu
     !
     zeu_new(:,:,:)=zeu_new(:,:,:) - zeu_w(:,:,:)
     call sp_zeu(zeu_w,zeu_w,nat,norm2)
     !write(6,'(5x,"Acoustic Sum Rule: || Z*(ASR) - Z*(orig)|| = ",E15.6)') &
     !     SQRT(norm2)
     !
     ! Check projection
     !
     !write(6,'("Check projection of zeu")')
     !do k=1,p
     !  zeu_x(:,:,:)=zeu_u(k,:,:,:)
     !  call sp_zeu(zeu_x,zeu_new,nat,scal)
     !  if (DABS(scal).gt.1d-10) write(6,'("k= ",I8," zeu_new|zeu_u(k)= ",F15.10)') k,scal
     !enddo
     !
     do i=1,3
        do j=1,3
           do na=1,nat
              zeu(i,j,na)=zeu_new(i,j,na)
           enddo
        enddo
     enddo
     deallocate (zeu_w, zeu_x)
     deallocate (zeu_u)
     deallocate (zeu_new)
  endif
  !
  ! ASR on dynamical matrix
  !
  if(asr.eq.'simple') then
     do i=1,3
        do j=1,3
           do na=1,nat
              sum=0.0d0
              do nb=1,nat
                 if (na.ne.nb) sum=sum + DBLE (dyn(i,j,na,nb))
              end do
              !print *, "FILLING WITH:", sum, "BEFORE:", dyn(i,j, na,na)
              dyn(i,j,na,na) = DCMPLX(-sum, 0.d0)
           end do
        end do
     end do
     !
  else
     ! generating the vectors of the orthogonal of the subspace to project
     ! the dyn. matrix on
     !
     allocate (u(6*3*nat,3,3,nat,nat))
     allocate (dynr_new(2,3,3,nat,nat))
     u(:,:,:,:,:)=0.0d0
     do i=1,3
        do j=1,3
           do na=1,nat
              do nb=1,nat
                 dynr_new(1,i,j,na,nb) = DBLE (dyn(i,j,na,nb) )
                 dynr_new(2,i,j,na,nb) =AIMAG (dyn(i,j,na,nb) )
              enddo
           enddo
        enddo
     enddo
     !
     p=0
     do i=1,3
        do j=1,3
           do na=1,nat
              ! These are the 3*3*nat vectors associated with the
              ! translational acoustic sum rules
              p=p+1
              do nb=1,nat
                 u(p,i,j,na,nb)=1.0d0
              enddo
              !
           enddo
        enddo
     enddo
     !
     if (n.eq.4) then
        do i=1,3
           do na=1,nat
              ! These are the 3*nat vectors associated with the
              ! single rotational sum rule (1D system)
              p=p+1
              do nb=1,nat
                 u(p,i,axis,na,nb)=0.0d0
                 u(p,i,MOD(axis,3)+1,na,nb)=-tau(MOD(axis+1,3)+1,nb)
                 u(p,i,MOD(axis+1,3)+1,na,nb)=tau(MOD(axis,3)+1,nb)
              enddo
              !
           enddo
        enddo
     endif
     !
     if (n.eq.6) then
        do i=1,3
           do j=1,3
              do na=1,nat
                 ! These are the 3*3*nat vectors associated with the
                 ! three rotational sum rules (0D system - typ. molecule)
                 p=p+1
                 do nb=1,nat
                    u(p,i,j,na,nb)=0.0d0
                    u(p,i,MOD(j,3)+1,na,nb)=-tau(MOD(j+1,3)+1,nb)
                    u(p,i,MOD(j+1,3)+1,na,nb)=tau(MOD(j,3)+1,nb)
                 enddo
                 !
              enddo
           enddo
        enddo
     endif
     !
     allocate (ind_v(9*nat*nat,2,4))
     allocate (v(9*nat*nat,2))
     m=0
     do i=1,3
        do j=1,3
           do na=1,nat
              do nb=1,nat
                 ! These are the vectors associated with the symmetry constraints
                 q=1
                 l=1
                 do while((l.le.m).and.(q.ne.0))
                    if ((ind_v(l,1,1).eq.i).and.(ind_v(l,1,2).eq.j).and. &
                         (ind_v(l,1,3).eq.na).and.(ind_v(l,1,4).eq.nb)) q=0
                    if ((ind_v(l,2,1).eq.i).and.(ind_v(l,2,2).eq.j).and. &
                         (ind_v(l,2,3).eq.na).and.(ind_v(l,2,4).eq.nb)) q=0
                    l=l+1
                 enddo
                 if ((i.eq.j).and.(na.eq.nb)) q=0
                 if (q.ne.0) then
                    m=m+1
                    ind_v(m,1,1)=i
                    ind_v(m,1,2)=j
                    ind_v(m,1,3)=na
                    ind_v(m,1,4)=nb
                    v(m,1)=1.0d0/DSQRT(2.0d0)
                    ind_v(m,2,1)=j
                    ind_v(m,2,2)=i
                    ind_v(m,2,3)=nb
                    ind_v(m,2,4)=na
                    v(m,2)=-1.0d0/DSQRT(2.0d0)
                 endif
              enddo
           enddo
        enddo
     enddo
     !
     ! Gram-Schmidt orthonormalization of the set of vectors created.
     ! Note that the vectors corresponding to symmetry constraints are already
     ! orthonormalized by construction.
     !
     allocate ( w(3,3,nat,nat), x(3,3,nat,nat))
     n_less=0
     do k=1,p
        w(:,:,:,:)=u(k,:,:,:,:)
        x(:,:,:,:)=u(k,:,:,:,:)
        do l=1,m
           !
           call sp2(x,v(l,:),ind_v(l,:,:),nat,scal)
           do r=1,2
              i=ind_v(l,r,1)
              j=ind_v(l,r,2)
              na=ind_v(l,r,3)
              nb=ind_v(l,r,4)
              w(i,j,na,nb)=w(i,j,na,nb)-scal*v(l,r)
           enddo
        enddo
        if (k.le.(9*nat)) then
           na1=MOD(k,nat)
           if (na1.eq.0) na1=nat
           j1=MOD((k-na1)/nat,3)+1
           i1=MOD((((k-na1)/nat)-j1+1)/3,3)+1
        else
           q=k-9*nat
           if (n.eq.4) then
              na1=MOD(q,nat)
              if (na1.eq.0) na1=nat
              i1=MOD((q-na1)/nat,3)+1
           else
              na1=MOD(q,nat)
              if (na1.eq.0) na1=nat
              j1=MOD((q-na1)/nat,3)+1
              i1=MOD((((q-na1)/nat)-j1+1)/3,3)+1
           endif
        endif
        do q=1,k-1
           r=1
           do i_less=1,n_less
              if (u_less(i_less).eq.q) r=0
           enddo
           if (r.ne.0) then
              call sp3(x,u(q,:,:,:,:),i1,na1,nat,scal)
              w(:,:,:,:) = w(:,:,:,:) - scal* u(q,:,:,:,:)
           endif
        enddo
        call sp1(w,w,nat,norm2)
        if (norm2.gt.1.0d-16) then
           u(k,:,:,:,:) = w(:,:,:,:) / DSQRT(norm2)
        else
           n_less=n_less+1
           u_less(n_less)=k
        endif
     enddo
     !
     ! Projection of the dyn. "vector" on the orthogonal of the
     ! subspace of the vectors verifying the sum rules and symmetry contraints
     !
     w(:,:,:,:)=0.0d0
     do l=1,m
        call sp2(dynr_new(1,:,:,:,:),v(l,:),ind_v(l,:,:),nat,scal)
        do r=1,2
           i=ind_v(l,r,1)
           j=ind_v(l,r,2)
           na=ind_v(l,r,3)
           nb=ind_v(l,r,4)
           w(i,j,na,nb)=w(i,j,na,nb)+scal*v(l,r)
        enddo
     enddo
     do k=1,p
        r=1
        do i_less=1,n_less
           if (u_less(i_less).eq.k) r=0
        enddo
        if (r.ne.0) then
           x(:,:,:,:)=u(k,:,:,:,:)
           call sp1(x,dynr_new(1,:,:,:,:),nat,scal)
           w(:,:,:,:) = w(:,:,:,:) + scal* u(k,:,:,:,:)
        endif
     enddo
     !
     ! Final substraction of the former projection to the initial dyn,
     ! to get the new "projected" dyn
     !
     dynr_new(1,:,:,:,:)=dynr_new(1,:,:,:,:) - w(:,:,:,:)
     call sp1(w,w,nat,norm2)
     !write(6,'(5x,"Acoustic Sum Rule: ||dyn(ASR) - dyn(orig)||= ",E15.6)') &
     !     DSQRT(norm2)
     !
     ! Check projection
     !
     !write(6,'("Check projection")')
     !do l=1,m
     !  call sp2(dynr_new(1,:,:,:,:),v(l,:),ind_v(l,:,:),nat,scal)
     !  if (DABS(scal).gt.1d-10) write(6,'("l= ",I8," dyn|v(l)= ",F15.10)') l,scal
     !enddo
     !do k=1,p
     !  x(:,:,:,:)=u(k,:,:,:,:)
     !  call sp1(x,dynr_new(1,:,:,:,:),nat,scal)
     !  if (DABS(scal).gt.1d-10) write(6,'("k= ",I8," dyn|u(k)= ",F15.10)') k,scal
     !enddo
     !
     deallocate ( w, x )
     deallocate ( v )
     deallocate ( ind_v )
     deallocate ( u )
     !
     do i=1,3
        do j=1,3
           do na=1,nat
              do nb=1,nat
                 dyn (i,j,na,nb) = &
                      DCMPLX(dynr_new(1,i,j,na,nb), dynr_new(2,i,j,na,nb))
              enddo
           enddo
        enddo
     enddo
     deallocate ( dynr_new )
  endif
  !
  return
end subroutine set_asr
!
!
!----------------------------------------------------------------------
subroutine sp_zeu(zeu_u,zeu_v,nat,scal)
  !-----------------------------------------------------------------------
  !
  ! does the scalar product of two effective charges matrices zeu_u and zeu_v
  ! (considered as vectors in the R^(3*3*nat) space, and coded in the usual way)
  !
  implicit none
  integer i,j,na,nat
  double precision zeu_u(3,3,nat)
  double precision zeu_v(3,3,nat)
  double precision scal
  !
  !
  scal=0.0d0
  do i=1,3
    do j=1,3
      do na=1,nat
        scal=scal+zeu_u(i,j,na)*zeu_v(i,j,na)
      enddo
    enddo
  enddo
  !
  return
  !
end subroutine sp_zeu
!
!
!----------------------------------------------------------------------
subroutine sp1(u,v,nat,scal)
  !-----------------------------------------------------------------------
  !
  ! does the scalar product of two dyn. matrices u and v (considered as
  ! vectors in the R^(3*3*nat*nat) space, and coded in the usual way)
  !
  implicit none
  integer i,j,na,nb,nat
  double precision u(3,3,nat,nat)
  double precision v(3,3,nat,nat)
  double precision scal
  !
  !
  scal=0.0d0
  do i=1,3
    do j=1,3
      do na=1,nat
        do nb=1,nat
          scal=scal+u(i,j,na,nb)*v(i,j,na,nb)
        enddo
      enddo
    enddo
  enddo
  !
  return
  !
end subroutine sp1
!
!----------------------------------------------------------------------
subroutine sp2(u,v,ind_v,nat,scal)
  !-----------------------------------------------------------------------
  !
  ! does the scalar product of two dyn. matrices u and v (considered as
  ! vectors in the R^(3*3*nat*nat) space). u is coded in the usual way
  ! but v is coded as explained when defining the vectors corresponding to the
  ! symmetry constraints
  !
  implicit none
  integer i,nat
  double precision u(3,3,nat,nat)
  integer ind_v(2,4)
  double precision v(2)
  double precision scal
  !
  !
  scal=0.0d0
  do i=1,2
    scal=scal+u(ind_v(i,1),ind_v(i,2),ind_v(i,3),ind_v(i,4))*v(i)
  enddo
  !
  return
  !
end subroutine sp2
!
!----------------------------------------------------------------------
subroutine sp3(u,v,i,na,nat,scal)
  !-----------------------------------------------------------------------
  !
  ! like sp1, but in the particular case when u is one of the u(k)%vec
  ! defined in set_asr (before orthonormalization). In this case most of the
  ! terms are zero (the ones that are not are characterized by i and na), so
  ! that a lot of computer time can be saved (during Gram-Schmidt).
  !
  implicit none
  integer i,j,na,nb,nat
  double precision u(3,3,nat,nat)
  double precision v(3,3,nat,nat)
  double precision scal
  !
  !
  scal=0.0d0
  do j=1,3
    do nb=1,nat
      scal=scal+u(i,j,na,nb)*v(i,j,na,nb)
    enddo
  enddo
  !
  return
  !
end subroutine sp3
