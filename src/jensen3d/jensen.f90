subroutine jensen(nTurbs, turbineX, turbineY, rotorDiameter,&
      &alpha, wind_speed, axialInduction, wtVelocity)

    implicit NONE

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    !in
    integer, intent(in) :: nTurbs
    real(dp), dimension(nTurbs), intent(in) :: turbineX, &
      &turbineY, rotorDiameter, axialInduction
    real(dp), intent(in) :: alpha, wind_speed

    !out
    real(dp), dimension(nTurbs), intent(out) :: wtVelocity

    !local
    integer :: i, j
    real(dp) :: pi, dx, dy, dz, d, big_R, A, overlap_area, totalLoss
    real(dp), dimension(nTurbs) :: little_r, loss
    real(dp), dimension(nTurbs, nTurbs) :: overlap_fraction

    little_r = 0.5_dp*rotorDiameter(:)
    pi = 3.1415926_dp

    !make identity matrix
    overlap_fraction = 0.
    do i = 1, nTurbs
        overlap_fraction(i,i) = 1.
    enddo

    do i = 1, nTurbs
        do j = 1, nTurbs
            dx = turbineX(i)-turbineX(j)
            dy = abs(turbineY(i)-turbineY(j))
            big_R = little_r(j)+dx*alpha
            A = little_r(i)**2_dp*pi
            overlap_area = 0.0_dp
            if (dx <= 0_dp) then
                overlap_fraction(i,j) = 0.0_dp
            else
                if (dy <= big_R-little_r(i)) then
                    if (A <= pi*big_R**2_dp) then
                        overlap_fraction(i,j) = 1.0_dp
                    else
                        overlap_fraction(i,j) = pi*big_R**2_dp/A
                    endif
                else if (dy >= big_R+little_r(i)) then
                    overlap_fraction(i,j) = 0.0_dp

                else
                    overlap_area = little_r(i)**2_dp*acos((dy**2_dp+little_r(i)**2_dp-big_R**2_dp)/(2_dp*dy*little_r(i)))+&
                                   &big_R**2_dp*acos((dy**2_dp+big_R**2_dp-little_r(i)**2_dp)/(2_dp*dy*big_R))-&
                                   &0.5_dp*sqrt((-1_dp*dy+little_r(i)+big_R)*(dy+little_r(i)-big_R)*(dy-little_r(i)+big_R)*&
                                   &(dy+little_r(i)+big_R))
                    overlap_fraction(i,j) = overlap_area/A
                endif
            endif
        enddo
    enddo

    do i = 1, nTurbs
        loss(:) = 0.0_dp
        do j = 1, nTurbs
            dx = turbineX(i)-turbineX(j)
            if (dx > 0) then
                loss(j) = overlap_fraction(i,j)*2.0_dp*axialInduction(j)*(little_r(j)/(little_r(j)+alpha*dx))**2_dp
                loss(j) = loss(j)**2_dp
            endif
        enddo
        totalLoss = sqrt(sum(loss))
        wtVelocity(i) = (1_dp-totalLoss)*wind_speed
    enddo
end subroutine
