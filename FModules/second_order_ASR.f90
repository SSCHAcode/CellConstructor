module second_order_ASR



logical :: perm_initialized = .false.
integer, allocatable :: P(:)

contains 

! ============================== INITIALIZATIONS ============================================

subroutine initialize_perm(R2, n_blocks,SClat,PBC)
    implicit none
    integer, intent(in)           :: R2(3,n_blocks),n_blocks
    integer, intent(in)           :: SCLat(3)
    logical, intent(in)           :: PBC
    !
    integer :: x(3), y(3),i_block,j_block
    logical :: found
   
    write(*,*) " "
    write(*,*) "Initialize indices for permutation...", perm_initialized

       
    allocate(P(n_blocks))

    do i_block = 1, n_blocks
                
        x=R2(:,i_block)
        found=.false.
        
        do j_block = 1, n_blocks
        !    
            y=R2(:,j_block)
            
            if ( Geq( y, -x  ,SCLat, PBC) ) then
                P(i_block)=j_block
                found=.true.
                cycle
            end if
        !    
        end do
        
        if ( .not. found ) then
        
            write(*,*) " ERROR: new vector found during the permutation symmetry initialization "
            write(*,*) "        the execution stops here. "
            write(*,*) "        If the 2ndFC is centered, try to repeat the centering with higher Far "            
            write(*,*) "        If the 2ndFC is not centered, try to repeat the ASR imposition with PBC=True "            
            stop
            
        end if
    
    end do

    perm_initialized =.true.

    write(*,*) "Initialize indices for permutation...done"
    write(*,*) " "
    
end subroutine initialize_perm

subroutine clear_all()
    implicit none

    ! This subroutine deallocates the permutations
    if (perm_initialized) then
        perm_initialized = .false.
        deallocate(P)
    end if
end subroutine clear_all

!==========================================================================================

function Geq(v1,v2,Lat,PBC)
    implicit none
    logical    :: Geq
    integer, dimension(:), intent(in)           :: v1,v2
    integer, dimension(:), intent(in) :: Lat
    logical, intent(in) :: PBC
    !    
    if (PBC) then
     Geq= ( ALL (mod(v1-v2,Lat)==0) )
    else
     Geq= ( ALL (v1-v2 ==0) )
    end if
    !
    !   
end function Geq

!=============================== PERM SYM =========================================

subroutine impose_perm_sym(FC,R2,SClat,PBC,verbose,FCvar,FC_sym,nat,n_blocks)
    implicit none
    integer, parameter :: DP = selected_real_kind(14,200)
    real(kind=DP), intent(in)     :: FC(3*nat,3*nat,n_blocks)
    integer, intent(in)           :: nat, n_blocks
    integer, intent(in)           :: R2(3,n_blocks)
    integer, intent(in)           :: SClat(3)
    logical, intent(in)           :: PBC, verbose
    !
    real(kind=DP), intent(out)    :: FC_sym(3*nat,3*nat,n_blocks), FCvar
    !
    integer                       :: jn1, jn2, i_block,nat3
    !

    
    if ( .not. perm_initialized ) call initialize_perm(R2,n_blocks,SClat,PBC)
          
    nat3=3*nat  
      
    do i_block = 1,n_blocks

    do jn1 = 1, nat3
    do jn2 = 1, nat3
    
    FC_sym(jn1,jn2, i_block)=    FC(jn1,jn2, i_block)    &
                               + FC(jn2,jn1, P(i_block))                                 

    end do
    end do
                                    
    end do

    FC_sym=FC_sym/2.0_dp


    FCvar=SUM(ABS(FC-FC_sym))/ SUM(ABS(FC)) 

    if (verbose) then
    write(*, * ) ""
    write(*,  "(' FC variation due to permut. symm.= 'e20.6)") FCvar
    write(*, * ) ""
    end if
    !
end subroutine impose_perm_sym

!================================ ASR ON 2nd INDEX ============================================

subroutine impose_ASR_2nd(FC,pow,SClat,PBC,verbose, &
                          FCvar,sum2nd,FC_asr,nat,n_blocks)
    implicit none
    integer, parameter :: DP = selected_real_kind(14,200)
    real(kind=DP), intent(in) :: FC(3*nat,3*nat, n_blocks)
    real(kind=DP), intent(in)    :: pow
    integer, intent(in) :: nat,n_blocks
    integer, intent(in) :: SCLat(3)
    logical, intent(in) :: PBC,verbose
    !
    real(kind=DP), intent(out) :: FCvar,sum2nd,FC_asr(3*nat,3*nat, n_blocks)
    !
    integer :: nat3,jn1,j2,n2,i_block
    real(kind=DP) :: d1,num,den,ratio,invpow

    !    
    nat3=3*nat
    
    FC_asr=0.0_dp
    
    d1=0.0_dp
    sum2nd=0.0_dp
    !
    do jn1=1,nat3
    do j2 =1,3
            !
            num=0.0_dp
            den=0.0_dp
            !
            if ( pow > 0.01_dp ) then
                !
                !
                do i_block=1,n_blocks
                do n2=1,nat
                    num=num+ FC(jn1 , j2+3*(n2-1), i_block)
                    den=den+ ABS(FC(jn1 , j2+3*(n2-1), i_block))**pow     
                end do
                end do
                !
                if ( den > 0.0_dp ) then
                    ratio=num/den
                    !
                    do i_block=1,n_blocks
                    do n2=1,nat
                        FC_asr(jn1, j2+3*(n2-1), i_block)= &
                            FC(jn1, j2+3*(n2-1), i_block)- &
                            ratio*ABS(FC(jn1, j2+3*(n2-1), i_block))**pow
                    end do
                    end do
                    !  
                else ! no need to modify the FC
                    !
                    do i_block=1,n_blocks
                    do n2=1,nat
                        FC_asr(jn1, j2+3*(n2-1), i_block)= &
                            FC(jn1, j2+3*(n2-1), i_block)
                    end do
                    end do                
                    !           
                end if
                !
                !
            else
                !
                !
                do i_block=1,n_blocks
                do n2=1,nat
                    num=num+ FC(jn1 , j2+3*(n2-1), i_block)
                    den=den+1      
                end do
                end do            
                !
                ratio=num/den
                !
                do i_block=1,n_blocks
                do n2=1,nat
                    FC_asr(jn1, j2+3*(n2-1), i_block)= &
                        FC(jn1, j2+3*(n2-1), i_block)- ratio
                end do
                end do
                !  
            end if
            !            
            !
            d1 = d1 + den
            sum2nd = sum2nd + ABS(num) ! should be zero if the ASR were fulfilled
            !
            !
    end do
    end do
    !    
    FCvar=SUM(ABS(FC-FC_ASR))/ SUM(ABS(FC)) 
    sum2nd=sum2nd/SUM(ABS(FC))
    !
    if (verbose) then
    if ( pow > 0.01_dp ) then
    
        invpow=1.0_dp/pow
        
        write(*, * ) ""   
        write(*, "(' ASR imposition on 2nd index with pow= 'f5.3)") pow
        write(*, "(' Previous values: sum(|sum_2nd phi|)/sum(|phi|)=       'e20.6)" ) sum2nd
        write(*, "('                  sum(|phi|**pow)**(1/pow)/sum(|phi|)= 'e20.6)" ) d1**invpow/SUM(ABS(FC))
        write(*, "(' FC relative variation= 'e20.6)" ) FCvar    
        write(*, * ) "" 
    
    else

        write(*, * ) ""   
        write(*, "(' ASR imposition on 2nd index with pow= 0' )") 
        write(*, "(' Previous value: sum(|sum_2nd phi|)/sum(|phi|)= 'e20.6 )" ) sum2nd
        write(*, "(' FC relative variation= 'e20.6)" ) FCvar    
        write(*, * ) ""     
    
    
    end if
    end if
    !
    !
end subroutine impose_ASR_2nd


! ==================================  MAIN ==========================================


subroutine impose_ASR(FC,R2,pow,SClat,PBC,threshold,maxite,FC_out,verbose,nat,n_blocks)
implicit none
integer, parameter :: DP = selected_real_kind(14,200)
real(kind=DP), intent(in) :: FC(3*nat,3*nat,n_blocks)
real(kind=DP), intent(in) :: pow,threshold
integer, intent(in)       :: maxite, nat, n_blocks
integer, intent(in) :: R2(3,n_blocks)
integer, intent(in) :: SCLat(3)
logical, intent(in) :: PBC,verbose
!
real(kind=DP), intent(out) :: FC_out(3*nat,3*nat,n_blocks)
!
real(kind=DP)   :: FC_tmp(3*nat,3*nat,n_blocks)
integer         :: ite, contr, iter, ios
real(kind=DP)   :: FCvar, sum2nd
logical :: converged

if (verbose) then
write(*,*) " "
write(*, "(' Iterative ASR imposition with pow= 'f5.3)") pow
write(*,*) " "
end if

call clear_all()
call initialize_perm(R2, n_blocks,SClat,PBC)

 converged = .false.
 FC_tmp=FC

ite=1
if ( maxite == 0 ) then
 contr=-1
else
 contr=1
end if
iter=ite*contr

do while (iter < maxite)

    if (verbose) write(*,"(' Iter #' I5 '  ====')") ite
    if (verbose) write(*,*) ""
        call impose_ASR_2nd(FC_tmp,pow,SClat,PBC,.false.,FCvar,sum2nd,FC_out ,nat,n_blocks)
    if (verbose) write(*,"('         Sum on 2nd='  e20.6  '    Imp. ASR on 2nd:  => delta FC=' e20.6)") sum2nd,FCvar
        call impose_perm_sym(FC_out, R2, SClat,PBC,.false.,FCvar,FC_tmp,nat,n_blocks)
    if (verbose) write(*,"('                    '  20X    '    Imp. permut sym:  => delta FC=' e20.6)") FCvar
    if (verbose) write(*,*) ""

   !  check converg
   if ( sum2nd < threshold  .and. FCvar < threshold ) then
        write(*,*) " "
        write(*,"( ' * Convergence reached within threshold:' e20.6 )") threshold
        write(*,*) " "
        write(*,"( ' * Total FC relative variation:' e20.6 )") SUM(ABS(FC-FC_out))/ SUM(ABS(FC)) 
        converged = .True.
        EXIT
   end if
   
   !  check if there is STOP file
   OPEN(unit=100, file="STOP", status='OLD', iostat=ios)
   IF(ios==0) THEN
        CLOSE(100,status="DELETE")
        write(*,*) " File STOP found, the ASR execution terminates here"
        EXIT 
   ENDIF
   
   ite=ite+1
   iter=ite*contr
end do


if (.not. converged ) then
   write(*,*) " "
   write(*,"( ' Max number of iteration reached ('I6')' )") maxite
   write(*,"( ' Convergence not reached within threshold:' e20.6 )") threshold
   write(*,*) " "
end if   


end subroutine impose_ASR


end module second_order_ASR
