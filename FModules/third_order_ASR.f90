module third_order_ASR


contains 

subroutine impose_ASR(phi,nat,n_sup_WS,lat_min,lat_max,phi_ASR)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), intent(in), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS) :: phi 
    integer, intent(in) :: nat, n_sup_WS, lat_min(3), lat_max(3)
    !
    real(kind=DP), intent(out), dimension(3*nat,3*nat,3*nat,n_sup_WS*n_sup_WS) :: phi_ASR     
    !
    real(kind=DP),  dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS) :: phi_ASR_tmp,Px,Py 
    integer :: counter, I, J
        
    phi_ASR_tmp=phi
    !
    !
    
!     call computeP3(nat,n_sup_WS,phi_ASR_tmp,Px)
!     
!     phi_ASR_tmp=phi_ASR_tmp-Px
!     
!     call computeP2(nat,n_sup_WS,phi_ASR_tmp,Px)
!      
!     phi_ASR_tmp=phi_ASR_tmp-Px

    call computeP1(nat,n_sup_WS,lat_min,lat_max,phi_ASR_tmp,Px)
    
    phi_ASR_tmp=phi_ASR_tmp-Px
  
    call computeP1(nat,n_sup_WS,lat_min,lat_max,phi_ASR_tmp,Px)
    
    print*,'test= ',SUM(ABS(Px))
    
    !
    !
!     phi_ASR=RESHAPE(phi_ASR_tmp,(/3*nat,3*nat,3*nat,n_sup_WS*n_sup_WS/))
    
    
    ! RESHAPE 
    
    counter=0
    
    do I=1,n_sup_WS
     do J=1,n_sup_WS
     
        counter=counter+1
     
        phi_ASR(:,:,:,counter)=phi_ASR_tmp(:,:,:,I,J)
     
     end do
    end do  
    
    
end subroutine impose_ASR    
!=================================================================================
subroutine computeP3(nat,n_sup_WS,phi,P3)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(out) :: P3
    !
    integer , intent(in)  :: nat, n_sup_WS
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,gamma,u,I,J
    real(kind=DP)         :: x
    
    P3 = 0.0_dp
    !                                                                           
    do jn1 = 1,3*nat                                                            
        do jn2 = 1,3*nat                                                        
            do gamma = 1,3                                                      
                do I =1, n_sup_WS                                               
                    !                                                           
                    x=0.0_dp                                                    
                    do J = 1, n_sup_WS                                          
                    do u = 1, nat                                               
                            jn3=gamma+(u-1)*3                                   
                            x=x+phi(jn1,jn2,jn3,I,J)                            
                    end do                                                      
                    end do                                                      
                    !                                                           
                    do u = 1, nat                                               
                            jn3=gamma+(u-1)*3                                   
                            P3(jn1,jn2,jn3,I,:)=x                               
                    end do                                                      
                    !                                                           
                end do
            end do
        end do
    end do
    !
    P3=P3/(nat*n_sup_WS)
    
end subroutine computeP3    
!=================================================================================
subroutine computeP2(nat,n_sup_WS,phi,P2)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(out) :: P2
    !
    integer , intent(in)  :: nat, n_sup_WS
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,beta,t,I,J
    real(kind=DP)         :: x
    
    P2 = 0.0_dp
    !
    do jn1 = 1,3*nat
        do beta = 1,3
            do jn3 = 1,3*nat
                do J =1, n_sup_WS
                    !
                    x=0.0_dp
                    do I = 1, n_sup_WS
                    do t = 1, nat
                            jn2=beta+(t-1)*3
                            x=x+phi(jn1,jn2,jn3,I,J)
                    end do
                    end do
                    !
                    do t = 1, nat
                            jn2=beta+(t-1)*3
                            P2(jn1,jn2,jn3,:,J)=x
                    end do                    
                    !
                end do
            end do
        end do
    end do
    !
    P2=P2/(nat*n_sup_WS)
    !
end subroutine computeP2    
!=================================================================================
subroutine computeP1(nat,n_sup_WS,lat_min,lat_max,phi,P1)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(out) :: P1
    !
    integer , intent(in)  :: nat, n_sup_WS, lat_min(3), lat_max(3)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(in) :: phi
    !
    integer               :: jn1,jn2,jn3,alpha,s,I,J,H
    real(kind=DP)         :: x
    integer               :: xR(3,n_sup_WS),sec_minus_first(n_sup_WS,n_sup_WS),tot_weight
    real(kind=DP), dimension(3*nat,3*nat,3*nat,0:n_sup_WS,0:n_sup_WS) :: phi_masked   
    real(kind=DP), dimension(3*nat,3*nat,3*nat,0:n_sup_WS,0:n_sup_WS) :: P1_masked
    
    
    
    phi_masked=0.0_dp
    phi_masked(:,:,:,1:n_sup_WS,1:n_sup_WS)=phi(:,:,:,1:n_sup_WS,1:n_sup_WS)
    !
    do I = 1, n_sup_WS 
       xR(:,I)=one_to_three(I,lat_min,lat_max)
       do H=1, n_sup_WS 
         J=three_to_one(xR(:,I)-xR(:,H),lat_min,lat_max)
         if ( J >= 1 .and. J <= n_sup_WS) then
         sec_minus_first(H,I)=J
         else
         sec_minus_first(H,I)=0
         end if
       end do
    end do 
    !
    P1_masked = 0.0_dp
    !
    do alpha = 1,3
        do jn2 = 1,3*nat
            do jn3 = 1,3*nat
                do I = 1, n_sup_WS
                    do J = 1, n_sup_WS
                        !
                        x=0.0_dp
                        do H = 1, n_sup_WS
                        do s = 1, nat
                                jn1=alpha+(s-1)*3
                                x=x+phi_masked(jn1,jn2,jn3,sec_minus_first(H,I),sec_minus_first(H,J))
                        end do
                        end do
                        ! 
                        do H = 1, n_sup_WS                        
                        do s = 1, nat
                                jn1=alpha+(s-1)*3 
                                P1_masked(jn1,jn2,jn3,sec_minus_first(H,I),sec_minus_first(H,J))=x
                        end do    
                        end do
                        !
                    end do
                end do
            end do
        end do
    end do    
    !
    P1(:,:,:,1:n_sup_WS,1:n_sup_WS)=P1_masked(:,:,:,1:n_sup_WS,1:n_sup_WS)
    P1=P1/(nat*n_sup_WS)
    !
end subroutine computeP1    
!=================================================================================    
subroutine check_ASR(lat_min,lat_max,phi,printout,filename,nat,n_sup_WS,MAXabsASR)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    !
    integer, intent(in)  :: nat, n_sup_WS, lat_min(3), lat_max(3)
    real(kind=DP), dimension(3*nat,3*nat,3*nat,n_sup_WS,n_sup_WS), intent(in) :: phi
    logical, intent(in)  :: printout
    character(len=*)     :: filename
    !    
    real(kind=DP), intent(out)         :: MAXabsASR
    !
    integer               :: jn1,jn2,jn3,alpha,beta,gamma,s,t,u,H,I,J
    integer               :: xR(3,n_sup_WS),sec_minus_first(n_sup_WS,n_sup_WS)
    real(kind=DP)         :: ASR
    real(kind=DP), dimension(3*nat,3*nat,3*nat,0:n_sup_WS,0:n_sup_WS) :: phi_masked
    !
    !
    phi_masked=0.0_dp
    phi_masked(:,:,:,1:n_sup_WS,1:n_sup_WS)=phi(:,:,:,1:n_sup_WS,1:n_sup_WS)
    !
    do I = 1, n_sup_WS 
       xR(:,I)=one_to_three(I,lat_min,lat_max)
       do H=1, n_sup_WS 
         J=three_to_one(xR(:,I)-xR(:,H),lat_min,lat_max)
         if ( J >= 1 .and. J<= n_sup_WS) then
         sec_minus_first(H,I)=J
         else
         sec_minus_first(H,I)=0
         end if
       end do
    end do 
    !    
    if (printout) open(unit=666,file=filename,status="REPLACE")
    !
    ! check third index
!     if (printout) write(666,*) "# Third index"
    do jn1=1,3*nat
    do jn2=1,3*nat
    do I=1,n_sup_WS
    do gamma=1,3
        !
        ASR=0.0_dp
        do J=1,n_sup_WS
        do u=1,nat
          jn3=gamma+(u-1)*3
          ASR=ASR+phi(jn1,jn2,jn3,I,J)
        end do
        end do
        !
!         if (printout) write(666,*) jn1,jn2,gamma,I,ASR
        MAXabsASR=MAX(abs(ASR),MAXabsASR)
    end do
    end do
    end do
    end do
    ! check second index
!     if (printout) write(666,*) "# Second index"    
    do jn1=1,3*nat
    do jn3=1,3*nat
    do J=1,n_sup_WS
    do beta=1,3
        !
        ASR=0.0_dp
        do I=1,n_sup_WS
        do t=1,nat
          jn2=beta+(t-1)*3
          ASR=ASR+phi(jn1,jn2,jn3,I,J)
        end do
        end do
        !
!         if (printout) write(666,*) jn1,beta,jn3,J,ASR
        MAXabsASR=MAX(abs(ASR),MAXabsASR)
    end do
    end do
    end do
    end do    
    ! check first index
    if (printout) write(666,*) "# First index"    
    do alpha=1,3
    do jn2=1,3*nat
    do jn3=1,3*nat
    do I=1,n_sup_WS
    do J=1,n_sup_WS
        !
        ASR=0.0_dp
        do H=1,n_sup_WS 
        do s=1,nat
            jn1=alpha+(s-1)*3
          ASR=ASR+phi_masked(jn1,jn2,jn3,sec_minus_first(H,I),sec_minus_first(H,J))
        end do
        end do        
        !
        if (printout) write(666,*) alpha,jn2,jn3,I,J,ASR
        MAXabsASR=MAX(abs(ASR),MAXabsASR)        
    end do
    end do
    end do
    end do    
    end do
    !
    !
    if (printout) close(unit=666)    
    !
end subroutine check_ASR
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