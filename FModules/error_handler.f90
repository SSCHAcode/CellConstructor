!
! Copyright (C) 2001-2007 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------------
SUBROUTINE errore( calling_routine, message, ierr )
  !----------------------------------------------------------------------------
  !
  ! ... This is a simple routine which writes an error message to output: 
  ! ... if ierr <= 0 it does nothing, 
  ! ... if ierr  > 0 it stops.
  !
  ! ...          **** Important note for parallel execution ***
  !
  ! ... in parallel execution unit 6 is written only by the first node;
  ! ... all other nodes have unit 6 redirected to nothing (/dev/null).
  ! ... As a consequence an error not occurring on the first node
  ! ... will be invisible. For T3E and ORIGIN machines, this problem
  ! ... is solved by writing an error message to unit * instead of 6.
  ! ... Whenever possible (IBM SP machines), we write to the standard
  ! ... error, unit 0 (the message will appear in the error files 
  ! ... produced by loadleveler).
  !
  !
  IMPLICIT NONE
  !
  CHARACTER(LEN=*), INTENT(IN) :: calling_routine, message
    ! the name of the calling calling_routine
    ! the output message
  INTEGER,          INTENT(IN) :: ierr
    ! the error flag
  INTEGER :: crashunit
  INTEGER, EXTERNAL :: find_free_unit
  CHARACTER(LEN=6) :: cerr
  !
  !
  IF ( ierr <= 0 ) RETURN
  !
  ! ... the error message is written un the "*" unit
  !
  WRITE( cerr, FMT = '(I6)' ) ierr
  WRITE( UNIT = *, FMT = '(/,1X,78("%"))' )
  WRITE( UNIT = *, FMT = '(5X,"Error in routine ",A," (",A,"):")' ) &
        TRIM(calling_routine), TRIM(ADJUSTL(cerr))
  WRITE( UNIT = *, FMT = '(5X,A)' ) TRIM(message)
  WRITE( UNIT = *, FMT = '(1X,78("%"),/)' )
  !
#if defined (__MPI) && defined (__AIX)
  !
  ! ... in the case of ibm machines it is also written on the "0" unit
  ! ... which is automatically connected to stderr
  !
  WRITE( UNIT = 0, FMT = '(/,1X,78("%"))')
  WRITE( UNIT = 0, FMT = '(5X,"Error in routine ",A," (",A,"):")' ) &
        TRIM(calling_routine), TRIM(ADJUSTL(cerr))
  WRITE( UNIT = 0, FMT = '(5X,A)' ) TRIM(message)
  WRITE( UNIT = 0, FMT = '(1X,78("%"),/)' )
  !
#endif
  !
  WRITE( *, '("     stopping ...")' )
  !
  CALL flush_unit( 6 )
  !
#ifdef __PTRACE
#ifdef __INTEL
  call tracebackqq(user_exit_code=-1)
#else
    WRITE( UNIT = 0, FMT = '(5X,A)' ) "Printing strace..."
    CALL ptrace()
#endif
#endif 
!
#if defined (__MPI)
  !
  !  .. write the message to a file and close it before exiting
  !  .. this will prevent loss of information on systems that
  !  .. do not flush the open streams
  !  .. added by C.C.
  !
  crashunit = find_free_unit () 
  OPEN( UNIT = crashunit, FILE = crash_file, &
        POSITION = 'APPEND', STATUS = 'UNKNOWN' )
  !      
  WRITE( UNIT = crashunit, FMT = '(/,1X,78("%"))' )
  WRITE( UNIT = crashunit, FMT = '(5X,"task #",I10)' ) mpime
  WRITE( UNIT = crashunit, &
         FMT = '(5X,"from ",A," : error #",I10)' ) calling_routine, ierr
  WRITE( UNIT = crashunit, FMT = '(5X,A)' ) message
  WRITE( UNIT = crashunit, FMT = '(1X,78("%"),/)' )
  !
  CLOSE( UNIT = crashunit )
  !
  ! ... try to exit in a smooth way
  !
  CALL mp_abort ( 1, world_comm )
  !
#endif
  !
  STOP 1
  !
  RETURN
  !
END SUBROUTINE errore
!
!----------------------------------------------------------------------
SUBROUTINE infomsg( routine, message )
  !----------------------------------------------------------------------
  !
  ! ... This is a simple routine which writes an info message 
  ! ... from a given routine to output. 
  !
  USE io_global,  ONLY : stdout, ionode
  !
  IMPLICIT NONE
  !
  CHARACTER (LEN=*) :: routine, message
  ! the name of the calling routine
  ! the output message
  !
  IF ( ionode ) THEN
     !
     WRITE( stdout , '(5X,"Message from routine ",A,":")' ) routine
     WRITE( stdout , '(5X,A)' ) message
     !
  END IF
  !
  RETURN
  !
END SUBROUTINE infomsg
!
