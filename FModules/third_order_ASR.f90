module third_order_ASR


contains 

subroutine impose_ASR(phi,n_blocks,totnum_R,xR_list,xR2,xR3,phi_ASR,nat)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), intent(in), dimension(3*nat,3*nat,3*nat,n_blocks) :: phi 
    integer, intent(in) :: nat ! # atoms
    integer, intent(in) :: xR2(3,n_blocks),xR3(3,n_blocks), n_blocks ! R2 and R3 vectors in the blocks
    integer, intent(in) :: xR_list(3,totnum_R),totnum_R ! list different R vectors
    !
    real(kind=DP), intent(out), dimension(3*nat,3*nat,3*nat,n_blocks) :: phi_ASR     
    !
    real(kind=DP),  dimension(3*nat,3*nat,3*nat,n_blocks) :: Pgamma 
    !
    !
    phi_ASR=phi
    !
    !
    call computeP3(nat,n_blocks,xR2,totnum_R,xR_list,phi_ASR,Pgamma)

    phi_ASR=phi_ASR-Pgamma
    
    call computeP2(nat,n_blocks,xR3,totnum_R,xR_list,phi_ASR,Pgamma)
      
    phi_ASR=phi_ASR-Pgamma

    call computeP1(nat,n_blocks,xR2,xR3,phi_ASR,Pgamma)
    
    phi_ASR=phi_ASR-Pgamma
    !
    !
    
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
subroutine computeP1(nat,n_blocks,xR2,xR3,phi,P1)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(out) :: P1
    !
    integer , intent(in)  :: nat, n_blocks, xR2(3,n_blocks), xR3(3,n_blocks)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,alpha,s,i_block,j_block
    integer :: list(n_blocks,n_blocks),num(n_blocks)  ! list(:,i_block) is list of blocks
                             ! having R3-R2 equal to the R3-R2 of the i_block
                             ! the number of these blocks is num(i_blocks)
    integer :: xx,h
    real(kind=DP)         :: x
    real(kind=DP) :: tol=1.0d-6
    
      
    !
    ! Find blocks with a fixed (R3-R2) value
    !               
    do i_block=1,n_blocks ! select a xR3(:,i_block)-xR2(:,i_block) value
        !
        xx=0
        do j_block=1,n_blocks
            if  (SUM(ABS(xR3(:,j_block)-xR3(:,i_block)-xR2(:,j_block)+xR2(:,i_block))) < tol) then
             xx=xx+1
             list(xx,i_block)=j_block
            end if 
        end do
        num(i_block)=xx
    end do  
    !
    !
    P1 = 0.0_dp
    !
    do i_block=1,n_blocks ! loop on  R3-R2
    do alpha = 1,3
    do jn2 = 1,3*nat
    do jn3 = 1,3*nat
                        ! loop on blocks with this fixed R3-R2
                        x=0.0_dp
                        do h=1,num(i_block)
                        do s = 1, nat
                                jn1=alpha+(s-1)*3
                                x=x+phi(jn3,jn2,jn1,list(h,i_block))
                        end do
                        end do
                        ! 
!                         do h=1,num(i_block)                        
                        do s = 1, nat
                                jn1=alpha+(s-1)*3 
                                P1(jn3,jn2,jn1,i_block)=x/num(i_block)
                        end do    
!                         end do
                        !
    end do
    end do
    end do
    end do    
    !
    P1=P1/(nat)
    !
 end subroutine computeP1    
! =================================================================================
! ! ! ! !=================================================================================
! ! ! ! subroutine computeP2(nat,n_blocks,xR3,totnum_R3,xR3_list,phi,P2)
! ! ! !     implicit none
! ! ! !     INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
! ! ! !     real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(out) :: P2
! ! ! !     !
! ! ! !     integer , intent(in)  :: nat, n_sup_WS
! ! ! !     real(kind=DP), dimension(3*nat,3*nat,3*nat,n_blocks), intent(in) :: phi
! ! ! !     !
! ! ! !     integer               :: jn1,jn2,jn3,beta,t,i_R3,i_block,xx
! ! ! !     real(kind=DP)         :: x
! ! ! !     real(kind=DP) :: tol=1.0d-6
! ! ! !     
! ! ! !     !
! ! ! !     !                                                                               
! ! ! !     do i_R3=1,totnum_R3 ! loop sui vettori R3
! ! ! !         xx=0.0_dp
! ! ! !         do i_block=1, n_blocks  ! loop sui blocchi
! ! ! !             if (SUM(ABS(xR3(:,i_block)-xR3_list(:,i_R3))) < tol) then ! il blocco ha questo R3
! ! ! !                 xx=xx+1
! ! ! !                 index_blocks_R3(xx,i_R3)=i_block ! aggiung quest indic d blocco per qst R3
! ! ! !             end if 
! ! ! !         end do
! ! ! !         num_blocks_R3(i_R3)=xx ! numero totale di blocchi per i_R3_vec-th vettore R3
! ! ! !     end do
! ! ! !     !
! ! ! !     ! index_blocks_R3(1:num_blocks_R3(i_R3),i_R3) indici dei "num_blocks(i_R3)" blocchi
! ! ! !     ! che hanno come vettore R3, il vettore xR3_list(i_R3) 
! ! ! !     ! 
! ! ! !     P2 = 0.0_dp
! ! ! !     !
! ! ! !     do i_R3=1,totnum_R3 ! lista di vettori R3 diversi
! ! ! !         do jn1 = 1,3*nat
! ! ! !             do beta = 1,3
! ! ! !                 do jn3 = 1,3*nat                
! ! ! !                     ! 
! ! ! !                     !
! ! ! !                     x=0.0_dp                                                                        
! ! ! !                     do h=1,num_blocks_R3(i_R3) ! numero blocchi con questo R3,
! ! ! !                                                ! con indice  index_blocks_R3(h,i_R3_vec)
! ! ! !                     do t = 1, nat
! ! ! !                             jn2=beta+(t-1)*3
! ! ! !                             x=x+phi(jn3,jn2,jn1,index_blocks_R3(h,i_R3))
! ! ! !                     end do
! ! ! !                     end do
! ! ! !                     ! 
! ! ! !                     do h=1,num_blocks_R3(i_R3)             
! ! ! !                     do t = 1, nat
! ! ! !                             jn2=beta+(t-1)*3 
! ! ! !                             P2(jn3,jn2,jn1,index_blocks_R3(h,i_R3))=x/num_blocks_R3(i_R3)
! ! ! !                     end do    
! ! ! !                     end do
! ! ! !                     !                  
! ! ! !                     !                                                                          
! ! ! !                 end do
! ! ! !             end do
! ! ! !         end do
! ! ! !     end do
! ! ! !     !
! ! ! !     P2=P2/(nat)
! ! ! !     !
! ! ! ! end subroutine computeP2    
! ! ! ! !=================================================================================
! ! ! ! subroutine computeP1(nat,n_sup_WS,xR2,xR3,phi,P1)
! ! ! !     implicit none
! ! ! !     INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
! ! ! !     real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS*n_sup_WS), intent(out) :: P1
! ! ! !     !
! ! ! !     integer , intent(in)  :: nat, n_sup_WS, xR2(3,n_sup_WS*n_sup_WS), xR3(3,n_sup_WS*n_sup_WS)
! ! ! !     real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS*n_sup_WS), intent(in) :: phi
! ! ! !     !
! ! ! !     integer               :: jn1,jn2,jn3,alpha,s,i_block,&
! ! ! !                              j_block,list(n_sup_WS*n_sup_WS,n_sup_Ws*n_sup_WS),&
! ! ! !                              num(n_sup_WS*n_sup_WS),xx,h
! ! ! !     real(kind=DP)         :: x
! ! ! !     integer               :: xR(3,n_sup_WS),sec_minus_first(n_sup_WS,n_sup_WS),tot_weight(n_sup_WS)
! ! ! !     real(kind=DP) :: tol=1.0d-6
! ! ! !     
! ! ! !     
! ! ! !     do i_block=1,n_blocks
! ! ! !         !
! ! ! !         xx=0
! ! ! !         do j_block=1,n_blocks
! ! ! !             if  (SUM(ABS(xR3(:,j_block)-xR3(:,i_block)-xR2(:,j_block)+xR2(:,i_block))) < tol) then
! ! ! !              xx=xx+1
! ! ! !              list(xx,i_block)=j_block
! ! ! !             end if 
! ! ! !         end do
! ! ! !         num(i_block)=xx
! ! ! !     end do  
! ! ! !     !
! ! ! !     !
! ! ! !     P1 = 0.0_dp
! ! ! !     !
! ! ! !     do i_block=1,n_sup_WS*n_sup_WS
! ! ! !         do alpha = 1,3
! ! ! !             do jn2 = 1,3*nat
! ! ! !                 do jn3 = 1,3*nat
! ! ! !                         !
! ! ! !                         x=0.0_dp
! ! ! !                         do h=1,num(i_block)
! ! ! !                         do s = 1, nat
! ! ! !                                 jn1=alpha+(s-1)*3
! ! ! !                                 x=x+phi(jn3,jn2,jn1,list(h,i_block))
! ! ! !                         end do
! ! ! !                         end do
! ! ! !                         ! 
! ! ! ! !                         do h=1,num(i_block)                        
! ! ! !                         do s = 1, nat
! ! ! !                                 jn1=alpha+(s-1)*3 
! ! ! !                                 P1(jn3,jn2,jn1,i_block)=x/num(i_block)
! ! ! !                         end do    
! ! ! ! !                         end do
! ! ! !                         !
! ! ! !                 end do
! ! ! !             end do
! ! ! !         end do
! ! ! !     end do    
! ! ! !     !
! ! ! !     P1=P1/(nat)
! ! ! !     !
! ! ! !  end subroutine computeP1    
!=================================================================================    
! subroutine check_ASR(lat_min,lat_max,phi,printout,filename,nat,n_sup_WS,MAXabsASR)
!     implicit none
!     INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
!     !
!     integer, intent(in)  :: nat, n_sup_WS, lat_min(3), lat_max(3)
!     real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(in) :: phi
!     logical, intent(in)  :: printout
!     character(len=*)     :: filename
!     !    
!     real(kind=DP), intent(out)         :: MAXabsASR
!     !
!     integer               :: jn1,jn2,jn3,alpha,beta,gamma,s,t,u,H,I,J
!     integer               :: xR(3,n_sup_WS),sec_minus_first(n_sup_WS,n_sup_WS)
!     real(kind=DP)         :: ASR
!     real(kind=DP), dimension(3*nat,3*nat,3*nat,0:n_sup_WS,0:n_sup_WS) :: phi_masked
!     !
!     !
!     phi_masked=0.0_dp
!     phi_masked(:,:,:,1:n_sup_WS,1:n_sup_WS)=phi(:,:,:,1:n_sup_WS,1:n_sup_WS)
!     !
!     do I = 1, n_sup_WS 
!        xR(:,I)=one_to_three(I,lat_min,lat_max)
!        do H=1, n_sup_WS 
!          J=three_to_one(xR(:,I)-xR(:,H),lat_min,lat_max)
!          if ( J >= 1 .and. J<= n_sup_WS) then
!          sec_minus_first(H,I)=J
!          else
!          sec_minus_first(H,I)=0
!          end if
!        end do
!     end do 
!     !    
!     if (printout) open(unit=666,file=filename,status="REPLACE")
!     !
!     ! check third index
! !     if (printout) write(666,*) "# Third index"
!     do jn1=1,3*nat
!     do jn2=1,3*nat
!     do I=1,n_sup_WS
!     do gamma=1,3
!         !
!         ASR=0.0_dp
!         do J=1,n_sup_WS
!         do u=1,nat
!           jn3=gamma+(u-1)*3
!           ASR=ASR+phi(jn1,jn2,jn3,I,J)
!         end do
!         end do
!         !
! !         if (printout) write(666,*) jn1,jn2,gamma,I,ASR
!         MAXabsASR=MAX(abs(ASR),MAXabsASR)
!     end do
!     end do
!     end do
!     end do
!     ! check second index
! !     if (printout) write(666,*) "# Second index"    
!     do jn1=1,3*nat
!     do jn3=1,3*nat
!     do J=1,n_sup_WS
!     do beta=1,3
!         !
!         ASR=0.0_dp
!         do I=1,n_sup_WS
!         do t=1,nat
!           jn2=beta+(t-1)*3
!           ASR=ASR+phi(jn1,jn2,jn3,I,J)
!         end do
!         end do
!         !
! !         if (printout) write(666,*) jn1,beta,jn3,J,ASR
!         MAXabsASR=MAX(abs(ASR),MAXabsASR)
!     end do
!     end do
!     end do
!     end do    
!     ! check first index
!     if (printout) write(666,*) "# First index"    
!     do alpha=1,3
!     do jn2=1,3*nat
!     do jn3=1,3*nat
!     do I=1,n_sup_WS
!     do J=1,n_sup_WS
!         !
!         ASR=0.0_dp
!         do H=1,n_sup_WS 
!         do s=1,nat
!             jn1=alpha+(s-1)*3
!           ASR=ASR+phi_masked(jn1,jn2,jn3,sec_minus_first(H,I),sec_minus_first(H,J))
!         end do
!         end do        
!         !
!         if (printout) write(666,*) alpha,jn2,jn3,I,J,ASR
!         MAXabsASR=MAX(abs(ASR),MAXabsASR)        
!     end do
!     end do
!     end do
!     end do    
!     end do
!     !
!     !
!     if (printout) close(unit=666)    
!     !
! end subroutine check_ASR
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

end module third_order_ASR