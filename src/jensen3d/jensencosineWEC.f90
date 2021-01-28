

subroutine WindFrame(nTurbines, wind_direction, turbineX, turbineY, turbineXw, turbineYw)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: wind_direction
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY

    ! out
    real(dp), dimension(nTurbines), intent(out) :: turbineXw, turbineYw

    ! local
    real(dp) :: windDirectionDeg, windDirectionRad
    real(dp), parameter :: pi = 3.141592653589793_dp, tol = 0.000001_dp

    windDirectionDeg = 270. - wind_direction
    if (windDirectionDeg < 0.) then
        windDirectionDeg = windDirectionDeg + 360.
    end if
    windDirectionRad = pi*windDirectionDeg/180.0

    turbineXw = turbineX*cos(-windDirectionRad)-turbineY*sin(-windDirectionRad)
    turbineYw = turbineX*sin(-windDirectionRad)+turbineY*cos(-windDirectionRad)

end subroutine WindFrame


subroutine get_cosine_factor_original(nTurbines, X, Y, R0, bound_angle, relaxationFactor,f_theta)

    implicit none
    integer, parameter :: dp = kind(0.d0)
    integer, intent(in) :: nTurbines

    real(dp), intent(in) :: bound_angle, R0, relaxationFactor
    real(dp), dimension(nTurbines), intent(in) :: X, Y

    real(dp), parameter :: pi = 3.141592653589793_dp

    real(dp), dimension(nTurbines,nTurbines), intent(out) :: f_theta

    real(dp) :: q, gamma, theta, z
    integer :: i, j

    q = pi/(bound_angle*pi/180.0)

    gamma = pi/2.0 - (bound_angle*pi/180.0)

    do i = 1, nTurbines
        do j = 1, nTurbines
            if (X(i) < X(j)) then
                z = (relaxationFactor * R0 * sin(gamma))/sin((bound_angle*pi/180.0))
                theta = atan((Y(j) - Y(i)) / (X(j) - X(i) + z))
                if (-(bound_angle*pi/180.0) < theta .and. theta < (bound_angle*pi/180.0)) then
                    f_theta(i,j) = (1. + cos(q*theta))/2.
                else
                    f_theta(i,j) = 0.
                end if
            else
                f_theta(i,j) = 0.
            end if
        end do
    end do

end subroutine get_cosine_factor_original

! calculate axial induction from Ct
subroutine ct_to_axial_ind_func(CT, axial_induction)
    
    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: CT

    ! out
    real(dp), intent(out) :: axial_induction

    ! initialize axial induction to zero
    axial_induction = 0.0_dp

    ! calculate axial induction
    if (CT > 0.96) then  ! Glauert condition
        axial_induction = 0.143_dp + sqrt(0.0203_dp-0.6427_dp*(0.889_dp - CT))
    else
        axial_induction = 0.5_dp*(1.0_dp-sqrt(1.0_dp-CT))
    end if
    
end subroutine ct_to_axial_ind_func

subroutine linear_interpolation(nPoints, x, y, xval, yval)

    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nPoints
    real(dp), dimension(nPoints), intent(in) :: x, y
    real(dp), intent(in) :: xval
    
    ! local
    integer :: idx
    real(dp) :: x0, x1, y0, y1
    
    ! out
    real(dp), intent(out) :: yval
    
    if (xval < x(1)) then
        yval = y(1)
    else if (xval > x(nPoints)) then
        yval = y(nPoints)
    
    else
        idx = 1
    
        do while ((xval > x(idx)) .and. (idx <= nPoints))
            idx = idx + 1
        end do
    
        idx = idx - 1
        
        x0 = x(idx)
        x1 = x((idx + 1))
        y0 = y(idx)
        y1 = y((idx + 1))
    
        yval = (xval-x0)*(y1-y0)/(x1-x0) + y0

    end if
    
end subroutine linear_interpolation

subroutine JensenWake(nTurbines, nCtPoints, turbineXw, turbineYw, turb_diam, alpha, &
&    bound_angle, ct_curve_ct, ct_curve_wind_speed, use_ct_curve, relaxationFactor, & 
&    windSpeed, wtVelocity)
    ! dependent (output) params: loss

    ! independent (input) params: turbineXw, turbineYw, turb_diam

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nctpoints
    real(dp), intent(in) :: turb_diam, relaxationFactor, bound_angle, alpha, windSpeed
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw
    real(dp), dimension(nCtPoints), intent(in) :: ct_curve_ct, ct_curve_wind_speed
    logical, intent(in) :: use_ct_curve

    ! out
    real(dp), dimension(nTurbines), intent(out) :: wtVelocity

    ! local
    real(dp) :: loss, a
    real(dp) :: r0, x, y, r
    real(dp), dimension(nTurbines) :: loss_array, Ct_local
    real(dp), dimension(nTurbines,nTurbines) :: f_theta
    real(dp), parameter :: pi = 3.141592653589793_dp
    integer :: i, j

    intrinsic sum, sqrt

    r0 = turb_diam/2.0_dp

    wtVelocity(:) = windSpeed

    call get_cosine_factor_original(nTurbines, turbineXw, turbineYw, r0, bound_angle, relaxationFactor, f_theta)

    !print *, "using Jensen Fortran"
    do i = 1, nturbines
        do j = 1, nTurbines
            
            x = turbineXw(i) - turbineXw(j)
            y = turbineYw(i) - turbineYw(j)
            if (x > 0.) then
                r = alpha*x + r0
                if (use_ct_curve) then
                    call ct_to_axial_ind_func(ct_local(j), a)
                else
                    a = 1.0_dp/3.0_dp
                end if
                loss_array(j) = 2.0_dp*a*((r0/(r0 + alpha*x))**2)*f_theta(j,i)
            else
                loss_array(j) = 0.0_dp
            end if
        end do
        loss = sqrt(sum(loss_array**2))
        wtVelocity(i) = (1.0 - loss) * windSpeed

        ! update thrust coefficient for turbI
        call linear_interpolation(nCtPoints, ct_curve_wind_speed, ct_curve_ct, wtVelocity(i), Ct_local(i))
    end do

end subroutine JensenWake


! subroutine DirPower(nTurbines, turbineX, turbineY, wind_dir_deg, wind_speed, turb_diam, &
!                     &turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor, pwrDir)

!     implicit none

!     ! define precision to be the standard for a double precision ! on local system
!     integer, parameter :: dp = kind(0.d0)

!     ! in
!     integer, intent(in) :: nTurbines
!     real(dp), intent(in) :: wind_dir_deg, wind_speed, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor
!     real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY

!     ! out
!     real(dp), intent(out) :: pwrDir

!     ! local
!     real(dp), dimension(nTurbines) :: wind_speed_eff, turb_pwr
!     real(dp), dimension(nTurbines) :: turbineXw, turbineYw, loss
!     integer :: n

!     call WindFrame(nTurbines, wind_dir_deg, turbineX, turbineY, turbineXw, turbineYw)
!     call JensenWake(nTurbines, turbineXw, turbineYw, turb_diam, relaxationFactor, loss)

!     wind_speed_eff = wind_speed*(1.-loss)


!     do n = 1, nTurbines
!         if (turb_ci <= wind_speed_eff(n) .and. wind_speed_eff(n) < rated_ws) then
!             turb_pwr(n) = rated_pwr * ((wind_speed_eff(n)-turb_ci)/(rated_ws-turb_ci))**3
!         else if (rated_ws <= wind_speed_eff(n) .and. wind_speed_eff(n) < turb_co) then
!             turb_pwr(n) = rated_pwr
!         else
!             turb_pwr(n) = 0.0
!         end if
!     end do

!     pwrDir = sum(turb_pwr)


! end subroutine DirPower



! subroutine calcAEP(nTurbines, nDirections, turbineX, turbineY, wind_freq, wind_speed, wind_dir,&
!             &turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor, AEP)

!     implicit none

!     ! define precision to be the standard for a double precision ! on local system
!     integer, parameter :: dp = kind(0.d0)

!     ! in
!     integer, intent(in) :: nTurbines, nDirections
!     real(dp), intent(in) :: turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor
!     real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY
!     real(dp), dimension(nDirections), intent(in) :: wind_freq, wind_speed, wind_dir

!     ! out
!     real(dp), intent(out) :: AEP

!     ! local
!     real(dp), dimension(nDirections) :: pwr_produced
!     real(dp) :: hrs_per_year, pwrDir
!     integer :: i


!     do i = 1, nDirections
!         call DirPower(nTurbines, turbineX, turbineY, wind_dir(i), wind_speed(i), turb_diam, &
!                             &turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor, pwrDir)
!         pwr_produced(i) = pwrDir
!     end do

!     hrs_per_year = 365.*24.
!     AEP = hrs_per_year * (sum(wind_freq * pwr_produced))
!     AEP = AEP/1000000.0

! end subroutine calcAEP
