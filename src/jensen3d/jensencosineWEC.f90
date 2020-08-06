

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


subroutine JensenWake(nTurbines, turbineXw, turbineYw, turb_diam, alpha, bound_angle, a, relaxationFactor, loss)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: turb_diam, relaxationFactor, a, bound_angle, alpha
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw

    ! out
    real(dp), dimension(nTurbines), intent(out) :: loss

    ! local
    real(dp) :: r0, x, y, r
    real(dp), dimension(nTurbines) :: loss_array
    real(dp), dimension(nTurbines,nTurbines) :: f_theta
    real(dp), parameter :: pi = 3.141592653589793_dp
    integer :: i, j

    intrinsic sum, sqrt

    r0 = turb_diam/2.0_dp

    call get_cosine_factor_original(nTurbines, turbineXw, turbineYw, r0, bound_angle, relaxationFactor, f_theta)
    !print *, "using Jensen Fortran"
    do i = 1, nTurbines

        do j = 1, nTurbines
            x = turbineXw(i) - turbineXw(j)
            y = turbineYw(i) - turbineYw(j)
            if (x > 0.) then
                r = alpha*x + r0
                ! loss_array(j) = 2.0_dp*a*(r0*f_theta(j,i)/(r0 + alpha*x))**2
                loss_array(j) = 2.0_dp*a*((r0/(r0 + alpha*x))**2)*f_theta(j,i)
            else
                loss_array(j) = 0.0_dp
            end if
        end do
        loss(i) = sqrt(sum(loss_array**2))
    end do

end subroutine JensenWake


subroutine DirPower(nTurbines, turbineX, turbineY, wind_dir_deg, wind_speed, turb_diam, &
                    &turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor, pwrDir)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: wind_dir_deg, wind_speed, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY

    ! out
    real(dp), intent(out) :: pwrDir

    ! local
    real(dp), dimension(nTurbines) :: wind_speed_eff, turb_pwr
    real(dp), dimension(nTurbines) :: turbineXw, turbineYw, loss
    integer :: n

    call WindFrame(nTurbines, wind_dir_deg, turbineX, turbineY, turbineXw, turbineYw)
    call JensenWake(nTurbines, turbineXw, turbineYw, turb_diam, relaxationFactor, loss)

    wind_speed_eff = wind_speed*(1.-loss)


    do n = 1, nTurbines
        if (turb_ci <= wind_speed_eff(n) .and. wind_speed_eff(n) < rated_ws) then
            turb_pwr(n) = rated_pwr * ((wind_speed_eff(n)-turb_ci)/(rated_ws-turb_ci))**3
        else if (rated_ws <= wind_speed_eff(n) .and. wind_speed_eff(n) < turb_co) then
            turb_pwr(n) = rated_pwr
        else
            turb_pwr(n) = 0.0
        end if
    end do

    pwrDir = sum(turb_pwr)


end subroutine DirPower



subroutine calcAEP(nTurbines, nDirections, turbineX, turbineY, wind_freq, wind_speed, wind_dir,&
            &turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor, AEP)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nDirections
    real(dp), intent(in) :: turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY
    real(dp), dimension(nDirections), intent(in) :: wind_freq, wind_speed, wind_dir

    ! out
    real(dp), intent(out) :: AEP

    ! local
    real(dp), dimension(nDirections) :: pwr_produced
    real(dp) :: hrs_per_year, pwrDir
    integer :: i


    do i = 1, nDirections
        call DirPower(nTurbines, turbineX, turbineY, wind_dir(i), wind_speed(i), turb_diam, &
                            &turb_ci, turb_co, rated_ws, rated_pwr, relaxationFactor, pwrDir)
        pwr_produced(i) = pwrDir
    end do

    hrs_per_year = 365.*24.
    AEP = hrs_per_year * (sum(wind_freq * pwr_produced))
    AEP = AEP/1000000.0

end subroutine calcAEP
