module third_order_ASR


contains 

subroutine enlarge(phi_in,xR2_out,xR3_out,xR2_in,xR3_in,phi_out,nat,n_blocks_out,n_blocks_in)
implicit none
INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
integer, intent(in) :: nat,n_blocks_in,n_blocks_out
integer, intent(in) :: xR2_in(3,n_blocks_in),xR3_in(3,n_blocks_in)
integer, intent(in) :: xR2_out(3,n_blocks_out),xR3_out(3,n_blocks_out)
real(kind=DP), intent(in), dimension(3*nat,3*nat,3*nat,n_blocks_in) :: phi_in
!
real(kind=DP), intent(out), dimension(3*nat,3*nat,3*nat,n_blocks_out) :: phi_out
!
integer :: i_block,j_block,block_index(n_blocks_in)
! ===================
do i_block=1,n_blocks_in
 do j_block=1,n_blocks_out
  if ( SUM(ABS(xR2_in(:,i_block)-xR2_out(:,j_block))+ABS(xR3_in(:,i_block)-xR3_out(:,j_block))) < 1.0d-7) then
     block_index(i_block)=j_block
    cycle
  end if
 end do 
end do
!
phi_out=0.0_dp
do i_block=1,n_blocks_in
    phi_out(:,:,:,block_index(i_block))=phi_in(:,:,:,i_block)
end do
! ====================
end subroutine enlarge
!
subroutine impose_ASR(phi,xRlen,xR_list,totnum_Rdiff,xRdiff_list,xR2,xR3,phi_ASR,n_blocks,totnum_R,nat)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), intent(in), dimension(3*nat,3*nat,3*nat,n_blocks) :: phi 
    integer, intent(in) :: nat ! # atoms
    integer, intent(in) :: xR2(3,n_blocks),xR3(3,n_blocks), n_blocks ! R2 and R3 vectors in the blocks
    integer, intent(in) :: xR_list(3,totnum_R),totnum_R ! list different R vectors
    integer, intent(in) :: xRdiff_list(3,totnum_Rdiff),totnum_Rdiff ! list R2-R3 vectors
    integer, intent(in) :: xRlen(3)
    !
    real(kind=DP), intent(out), dimension(3*nat,3*nat,3*nat,n_blocks) :: phi_ASR     
    !
    real(kind=DP),  dimension(3*nat,3*nat,3*nat,n_blocks) :: Pi
    !
    !
    !
    !
    phi_ASR=phi
    !
    !
    call computeP3(nat,n_blocks,xR2,totnum_R,xR_list,phi_ASR,Pi) 
    
        phi_ASR=phi_ASR-Pi

    call computeP2(nat,n_blocks,xR3,totnum_R,xR_list,phi_ASR,Pi) 
    
        phi_ASR=phi_ASR-Pi    
    
    call computeP1(nat,n_blocks,xRlen,xR2,xR3,totnum_Rdiff,xRdiff_list,phi_ASR,Pi) 
    
        phi_ASR=phi_ASR-Pi        
!    
! TEST
! 
 call computeP3(nat,n_blocks,xR2,totnum_R,xR_list,phi_ASR,Pi)
 print*,"test1= ",SUM(ABS(Pi))
 call computeP2(nat,n_blocks,xR3,totnum_R,xR_list,phi_ASR,Pi) 
 print*,"test2= ",SUM(ABS(Pi))
 call computeP1(nat,n_blocks,xRlen,xR2,xR3,totnum_Rdiff,xRdiff_list,phi_ASR,Pi)  
 print*,"test3= ",SUM(ABS(Pi)) 
 
end subroutine impose_ASR
!=================================================================================
subroutine computeP3(nat,n_blocks,xR2,totnum_R2,xR2_list,phi,P3)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(out) :: P3
    !
    integer , intent(in)  :: nat, n_blocks,xR2(3,n_blocks),totnum_R2, xR2_list(3,totnum_R2)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,gamma,u,i_R2,i_block,index_blocks_R2(n_blocks,totnum_R2)
    integer               :: xx
    integer               :: num_blocks_R2(totnum_R2)
    integer               :: h
    real(kind=DP)         :: x
    real(kind=DP)         :: tol=1.0d-6
    
    !
    ! Find blocks with a fixed R2 value
    !                                                                               
    do i_R2=1,totnum_R2 ! loop on (different) R2 
        ! Look for blocks having xR2_list(:,i_R2) as R2
        xx=0
        do i_block=1, n_blocks  ! loop on blocks
            if (SUM(ABS(xR2(:,i_block)-xR2_list(:,i_R2))) < tol) then ! the block has this R2
                xx=xx+1
                index_blocks_R2(xx,i_R2)=i_block 
            end if 
        end do
        num_blocks_R2(i_R2)=xx ! total number of blocks with this R2
        !
    end do
!     print*,num_blocks_R2
    !
    !
    P3 = 0.0_dp 
    ! 
    do i_R2=1,totnum_R2 ! loop on different R2
    do jn1 = 1,3*nat
    do jn2 = 1,3*nat 
    do gamma = 1,3                                                            
        !    
        !
        x=0.0_dp                                                                        
        do h=1,num_blocks_R2(i_R2) ! loop on blocks with this  R2 fixed 
        do u = 1, nat
            jn3=gamma+(u-1)*3
            x=x+phi(jn3,jn2,jn1,index_blocks_R2(h,i_R2))
        end do
        end do
        ! 
        do h=1,num_blocks_R2(i_R2)             
        do u = 1, nat
            jn3=gamma+(u-1)*3 
            P3(jn3,jn2,jn1,index_blocks_R2(h,i_R2))=x/num_blocks_R2(i_R2)
        end do    
        end do
        !                  
        !                                                           
    end do
    end do
    end do
    end do
    !
    P3=P3/(nat)
    
end subroutine computeP3 
!=================================================================================
subroutine computeP2(nat,n_blocks,xR3,totnum_R3,xR3_list,phi,P2)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(out) :: P2
    !
    integer , intent(in)  :: nat, n_blocks,xR3(3,n_blocks),totnum_R3, xR3_list(3,totnum_R3)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,beta,t,i_R3,i_block,index_blocks_R3(n_blocks,totnum_R3)
    integer               :: xx
    integer               :: num_blocks_R3(totnum_R3)
    integer               :: h
    real(kind=DP)         :: x
    real(kind=DP)         :: tol=1.0d-6
    
    !
    ! Find blocks with a fixed R3 value
    !                                                                                                            
    do i_R3=1,totnum_R3 ! loop on (different) R3 
        ! Look for blocks having xR3_list(:,i_R3) as R3
        xx=0
        do i_block=1, n_blocks  ! loop on blocks
            if (SUM(ABS(xR3(:,i_block)-xR3_list(:,i_R3))) < tol) then ! the block has this R3
                xx=xx+1
                index_blocks_R3(xx,i_R3)=i_block 
            end if 
        end do
        num_blocks_R3(i_R3)=xx ! total number of blocks with this R3
        !
    end do
!     print*,num_blocks_R3
    !
    !
    P2 = 0.0_dp
    ! 
    do i_R3=1,totnum_R3 ! loop on different R3
    do jn1  = 1,3*nat
    do beta = 1,3                                                            
    do jn3  = 1,3*nat 
        !    
        !
        x=0.0_dp                                                                        
        do h=1,num_blocks_R3(i_R3) ! loop on blocks with this R3 fixed 
        do t = 1, nat
            jn2=beta+(t-1)*3
            x=x+phi(jn3,jn2,jn1,index_blocks_R3(h,i_R3))
        end do
        end do
        ! 
        do h=1,num_blocks_R3(i_R3)             
        do t = 1, nat
            jn2=beta+(t-1)*3 
            P2(jn3,jn2,jn1,index_blocks_R3(h,i_R3))=x/num_blocks_R3(i_R3)
        end do    
        end do
        !                  
        !                                                           
    end do
    end do
    end do
    end do
    !
    P2=P2/(nat)
    
end subroutine computeP2    
!=================================================================================
subroutine computeP23(nat,n_blocks,phi,P23)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(out) :: P23
    !
    integer , intent(in)  :: nat, n_blocks
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,beta,gamma,t,u
    integer               :: h
    real(kind=DP)         :: x
    

    !
    !
    P23 = 0.0_dp
    ! 
    do jn1  = 1,3*nat
    do beta  = 1,3                                                            
    do gamma = 1,3 
        !    
        !
        x=0.0_dp                                                                        
        do h=1,n_blocks ! 
        do t = 1, nat
        do u = 1, nat
            jn2=beta+(t-1)*3
            jn3=gamma+(u-1)*3
            x=x+phi(jn3,jn2,jn1,h)
        end do
        end do
        end do
        ! 
        do h=1,n_blocks
        do t = 1, nat
        do u = 1, nat
            jn2=beta+(t-1)*3
            jn3=gamma+(u-1)*3
            P23(jn3,jn2,jn1,h)=x/n_blocks
        end do    
        end do
        end do
        !                  
        !                                                           
    end do
    end do
    end do
    !
    P23=P23/(nat*nat)
    
end subroutine computeP23    
!=================================================================================
subroutine computeP1(nat,n_blocks,leng,xR2,xR3,totnum_Rdiff,xRdiff_list,phi,P1)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(out) :: P1
    !
    integer, intent(in)  :: nat, n_blocks, leng(3),xR2(3,n_blocks), xR3(3,n_blocks)
    integer, intent(in)  :: totnum_Rdiff, xRdiff_list(3,totnum_Rdiff)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,alpha,s,i_block,i_Rdiff
    integer :: num_blocks_Rdiff(totnum_Rdiff),index_blocks_Rdiff(n_blocks,totnum_Rdiff)  ! 
    integer :: xx,h,v(3)
    real(kind=DP)         :: x
    real(kind=DP) :: tol=1.0d-6
    
    
    
    
    do i_Rdiff=1,totnum_Rdiff ! loop on (different) Rdiff 
        ! Look for blocks having xR_list(:,i_R3) as R3
        xx=0
        do i_block=1, n_blocks  ! loop on blocks
            v=(xR2(:,i_block)-xR3(:,i_block))
            if (Geq(v,xRdiff_list(3,i_Rdiff),leng) ) then ! the block has this R3
                xx=xx+1
                index_blocks_Rdiff(xx,i_Rdiff)=i_block 
            end if 
        end do
        num_blocks_Rdiff(i_Rdiff)=xx ! total number of blocks with this R3
        !
    end do    
    !
    P1 = 0.0_dp
    !
    do i_Rdiff=1,totnum_Rdiff
    do alpha = 1,3
    do jn2 = 1,3*nat
    do jn3 = 1,3*nat
                        ! loop on blocks with this fixed R3-R2
                        x=0.0_dp
                        do h=1,num_blocks_Rdiff(i_Rdiff)
                        do s = 1, nat
                                jn1=alpha+(s-1)*3
                                x=x+phi(jn3,jn2,jn1,index_blocks_Rdiff(h,i_Rdiff))
                        end do
                        end do
                        ! 
                        do h=1,num_blocks_Rdiff(i_Rdiff)                     
                        do s = 1, nat
                                jn1=alpha+(s-1)*3 
                                P1(jn3,jn2,jn1,index_blocks_Rdiff(h,i_Rdiff))=x/num_blocks_Rdiff(i_Rdiff)
                        end do  
                        end do
                        !
    end do
    end do
    end do
    end do    
    !
    P1=P1/(nat)
    !
 end subroutine computeP1    
!=================================================================================    
function three_to_one_len(v,v_min,v_len)
   implicit none
   INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
   integer :: three_to_one_len
   integer, intent(in), dimension(3) :: v, v_min, v_len 
   !
   three_to_one_len=(v(1)-v_min(1))*v_len(2)*v_len(3)+(v(2)-v_min(2))*v_len(3)+(v(3)-v_min(3))+1
end function three_to_one_len   
!=================================================================================    
function three_to_one(v,v_min,v_max)
   implicit none
   INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
   integer :: three_to_one
   integer, intent(in), dimension(3) :: v, v_min, v_max 
   integer             :: v_len(3)
   !
   v_len=v_max-v_min+(/1,1,1/)
   three_to_one=(v(1)-v_min(1))*v_len(2)*v_len(3)+(v(2)-v_min(2))*v_len(3)+(v(3)-v_min(3))+1
end function three_to_one
!=================================================================================    
function one_to_three_len(J,v_min,v_len)
    implicit none
    integer, dimension(3) :: one_to_three_len
    integer, intent(in) :: J 
    integer, intent(in) :: v_min(3), v_len(3)
    !
    one_to_three_len(1)              =   (J-1) / (v_len(3) * v_len(2))                    + v_min(1) 
    one_to_three_len(2)              =   MOD( J-1 ,  (v_len(3) * v_len(2)) ) / v_len(3)   + v_min(2) 
    one_to_three_len(3)              =   MOD( J-1 ,   v_len(3) )                          + v_min(3) 
end function one_to_three_len
!=================================================================================    
function one_to_three(J,v_min,v_max)
    implicit none
    integer, dimension(3) :: one_to_three
    integer, intent(in) :: J 
    integer, intent(in) :: v_min(3), v_max(3)
    integer             :: v_len(3)
    !
    v_len=v_max-v_min+(/1,1,1/)
    one_to_three(1)              =   (J-1) / (v_len(3) * v_len(2))                    + v_min(1) 
    one_to_three(2)              =   MOD( J-1 ,  (v_len(3) * v_len(2)) ) / v_len(3)   + v_min(2) 
    one_to_three(3)              =   MOD( J-1 ,   v_len(3) )                          + v_min(3)   
end function one_to_three
!=================================================================================    
function min_el_wise_2(a,b)
    implicit none        
    integer, dimension(3)             :: min_el_wise_2
    integer, dimension(3), intent(in) :: a, b
    !
    min_el_wise_2=(/min(a(1),b(1)),min(a(2),b(2)),min(a(3),b(3))/)
    !
end function min_el_wise_2   
!=================================================================================                     
function max_el_wise_2(a,b)
    implicit none        
    integer, dimension(3)             :: max_el_wise_2
    integer, dimension(3), intent(in) :: a, b
    !
    max_el_wise_2=(/max(a(1),b(1)),max(a(2),b(2)),max(a(3),b(3))/)
    !
end function max_el_wise_2   
!=================================================================================                     
function min_el_wise_3(a,b,c)
    implicit none        
    integer, dimension(3)             :: min_el_wise_3
    integer, dimension(3), intent(in) :: a, b, c
    !
    min_el_wise_3=(/min(a(1),b(1),c(1)),min(a(2),b(2),c(2)),min(a(3),b(3),c(3))/)
    !
end function min_el_wise_3   
!=================================================================================                     
function max_el_wise_3(a,b,c)
    implicit none        
    integer, dimension(3)             :: max_el_wise_3
    integer, dimension(3), intent(in) :: a, b, c
    !
    max_el_wise_3=(/max(a(1),b(1),c(1)),max(a(2),b(2),c(2)),max(a(3),b(3),c(3))/)
    !
end function max_el_wise_3   
!=================================================================================                     
function cryst_to_cart(v,alat)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3)    :: cryst_to_cart
    integer, dimension(3), intent(in)          :: v
    real(kind=DP), dimension(3,3), intent(in) :: alat
    !
    integer :: i
    !  
    do i=1,3
        cryst_to_cart(i)=alat(1,i)*v(1)+alat(2,i)*v(2)+alat(3,i)*v(3)
    end do
    !   
end function cryst_to_cart
!==========================================================================================
function Geq(v1,v2,leng)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    logical    :: Geq
    integer, dimension(3), intent(in)          :: v1,v2,leng
    !
    integer :: w(3)
    !  
    w=v1-v2

    Geq=.false.
    if (mod(w(1),leng(1))==0 .and. mod(w(2),leng(2))==0 .and. mod(w(3),leng(3))==0 )then
    Geq =.true.
    end if
    !   
end function Geq
!==========================================================================================

end module third_order_ASR