# This example simulates a two-dimensional oceanic bottom boundary layer
# in a domain that's tilted with respect to gravity. 
# bottom boundary conditions are no-slip, no flux and there is no initial noise added
# constant along-slope (y-direction) velocity, and constant density stratification.
#
# This example illustrates
#
#   * changing the direction of gravitational acceleration in the buoyancy model;
#   * changing the axis of rotation for Coriolis forces.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, NCDatasets, CairoMakie"
# ```
#
# ## The domain
#
# We create a grid with finer resolution near the bottom,

using Oceananigans
using Oceananigans.Units

##################
# set up grid
##################

Lz        = 2000meters            # total depth        (m)
Nz        = 150                   # number of cells
dz_bottom = 0.5meters             # uniform cell size  (m)
n_const   = 100                   # number of uniform cells


# add yuchen's comments back
function compute_stretched_faces(Lz, Nz, dz_bottom, n_const)
    n_str = Nz - n_const
    z_str = Lz - dz_bottom * n_const

    r_lo, r_hi = 1.0 + eps(), 1.05

    while dz_bottom * (r_hi^n_str - 1) / (r_hi - 1) ≤ z_str
        r_hi = 1.0 + 2 * (r_hi - 1)
    end

    for _ in 1:60
        r_mid = 0.5 * (r_lo + r_hi)
        summed = dz_bottom * (r_mid^n_str - 1) / (r_mid - 1)
        if summed < z_str
            r_lo = r_mid
        else
            r_hi = r_mid
        end
    end

    r = 0.5 * (r_lo + r_hi)
    dz_stretch = dz_bottom .* (r .^ (0:n_str-1))
    dz = vcat(fill(dz_bottom, n_const), dz_stretch)
    return [0 ; cumsum(dz)]
end

z_faces = compute_stretched_faces(Lz, Nz, dz_bottom, n_const)
Lx, Nx = 200meters, 32 
grid = RectilinearGrid( topology = (Periodic, Flat, Bounded),
                       size      = (Nx, Nz),
                       x         = (0, Lx),
                       z         = z_faces)

#GPU(),

###############################
# plot size of z-grid cells to check resolution
###############################
# comment out when running model
# using CairoMakie

# dz = vec(collect(zspacings(grid, Center())))
# z = collect(znodes(grid, Center()))  # This should already be 1D

# scatterlines(dz, z;
#              axis = (xlabel = "Vertical spacing (m)",
#                      ylabel = "Depth (m)"))

# current_figure() #hide

###############################
# Tilting the domain and boundary condtions
###############################

# We use a domain that's tilted with respect to gravity by

rad =  5e-3
θ = rad*180/pi # degrees

# so that ``x`` is the along-slope direction, ``z`` is the across-slope direction that
# is perpendicular to the bottom, and the unit vector anti-aligned with gravity is

ẑ = (sind(θ), 0, cosd(θ))

# Changing the vertical direction impacts both the `gravity_unit_vector`
# for `BuoyancyForce` as well as the `rotation_axis` for Coriolis forces,

buoyancy = BuoyancyForce(BuoyancyTracer(), gravity_unit_vector = .-ẑ)
coriolis = ConstantCartesianCoriolis(f = 5.5e-5, rotation_axis = ẑ)
# PC2023 used a coriolis parameter of 5.5*10^-5 - redo that simulation with this one?

# where above we used a constant Coriolis parameter ``f = 10^{-4} \, \rm{s}^{-1}``.
# The tilting also affects the kind of density stratified flows we can model.
# In particular, a constant density stratification in the tilted
# coordinate system

@inline constant_stratification(x, z, t, p) = p.N² * (x * p.ẑ[1] + z * p.ẑ[3])

# is _not_ periodic in ``x``. Thus we cannot explicitly model a constant stratification
# on an ``x``-periodic grid such as the one used here. Instead, we simulate periodic
# _perturbations_ away from the constant density stratification by imposing
# a constant stratification as a `BackgroundField`,

N² = 1e-5 # s⁻² # background vertical buoyancy gradient
B∞_field = BackgroundField(constant_stratification, parameters=(; ẑ, N² = N²))

# We choose to impose a bottom boundary condition of zero *total* diffusive buoyancy
# flux across the seafloor,
# ```math
# ∂_z B = ∂_z b + N^{2} \cos{\theta} = 0.
# ```
# This shows that to impose a no-flux boundary condition on the total buoyancy field ``B``, we must apply a boundary condition to the perturbation buoyancy ``b``,
# ```math
# ∂_z b = - N^{2} \cos{\theta}.
# ```

∂z_b_bottom = - N² * cosd(θ)
negative_background_diffusive_flux = GradientBoundaryCondition(∂z_b_bottom)
b_bcs = FieldBoundaryConditions(bottom = negative_background_diffusive_flux)




# Let us also create `BackgroundField` for the along-slope interior velocity:
V∞ = 0.1 # m s⁻¹ # imposed constant along slope velocity
V∞_field = BackgroundField(V∞)

# ## Create the `NonhydrostaticModel`
#
# We are now ready to create the model. We create a `NonhydrostaticModel` with a
# fifth-order `UpwindBiased` advection scheme and a constant viscosity and diffusivity.
# Here we use a smallish value of ``10^{-4} \, \rm{m}^2\, \rm{s}^{-1}``.


ν = 1e-4
κ = 1e-4

closure = ScalarDiffusivity(; ν, κ) # check if this is what we want and what it actually means

# no slip bottom boundary conditions
no_slip_bc = ValueBoundaryCondition(0.0)
no_slip_field_bcs = FieldBoundaryConditions(no_slip_bc);
u_bcs = no_slip_field_bcs 
v_bcs = no_slip_field_bcs

###############################
# Model setup and run
###############################

model = NonhydrostaticModel(; grid, buoyancy, coriolis, closure,
                            advection = UpwindBiased(order=5), # don't know what this means 'UpwindBiased'
                            tracers = :b,
                            boundary_conditions = (u=u_bcs, v=v_bcs, b=b_bcs), # add no slip bc's
                            background_fields = (; b=B∞_field, v=V∞_field)) # add constant velocisty and background stratification


noise(x, z) = 0* randn() * exp(-(10z)^2 / grid.Lz^2) # multiplying by zero for no noise
set!(model, u=noise, w=noise)

# ## Create and run a simulation
#
# We are now ready to create the simulation. We begin by setting the initial time step
# conservatively, based on the smallest grid size of our domain and either an advective
# or diffusive time scaling, depending on which is shorter.

# had issues with delta t size, currently at 1.25 seconds, at 2.5 seconds after a day there was nan and it stopped :\
Δt₀ = 0.25 * minimum([minimum_zspacing(grid) / V∞, minimum_zspacing(grid)^2/κ]) 
# run simulation which stops after 'stop_time'
simulation = Simulation(model, Δt = Δt₀, stop_time = 30days) 

# We use a `TimeStepWizard` to adapt our time-step,
# wizard fucks things up at the moment

# wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

# diagnostics 
# simulation.diagnostics[:max_w] = Maximum(abs, model.velocities.w)
# simulation.diagnostics[:max_v] = Maximum(abs, model.velocities.v)
# simulation.diagnostics[:KE] = Average((u, v ) -> 0.5 * (u^2 + v^2),
#                                        model.velocities.u, model.velocities.v, model.velocities.w)


# and also we add another callback to print a progress message,
using Printf

start_time = time_ns() # so we can print the total elapsed wall time

progress_message(sim) =
    @printf("Iteration: %04d, time: %s, Δt: %s, max|w|: %.1e m s⁻¹, wall time: %s\n",
            iteration(sim), prettytime(time(sim)),
            prettytime(sim.Δt), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(200))

###############################
# Save model outputs
###############################

# We add outputs to our model using the `NetCDFWriter`, which needs `NCDatasets` to be loaded:

u, v, w = model.velocities
b = model.tracers.b
B∞ = model.background_fields.tracers.b

B = b + B∞ 
V = v + V∞
ωy = ∂z(u) - ∂x(w)

outputs = (; u, V, w, B, ωy)

using NCDatasets

simulation.output_writers[:fields] = NetCDFWriter(model, outputs;
                                                  filename = joinpath(@__DIR__, "tilted_bottom_boundary_layer.nc"),
                                                  schedule = TimeInterval(20minutes),
                                                  overwrite_existing = true)

# Now we just run it!

run!(simulation)

# ## Visualize the results
#
# First we load the required package to load NetCDF output files and define the coordinates for
# plotting using existing objects:

using NCDatasets, CairoMakie

xb, yb, zb = nodes(B)
xω, yω, zω = nodes(ωy)
xv, yv, zv = nodes(V)
xu, yu, zu = nodes(u)

# Read in the simulation's `output_writer` for the two-dimensional fields and then create an
# animation showing the ``y``-component of vorticity.

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

fig = Figure(size = (800, 600))

axis_kwargs = (xlabel = "Across-slope distance (m)",
               ylabel = "Slope-normal\ndistance (m)",
               limits = ((0, Lx), (0, Lz)),
               )

ax_ω = Axis(fig[2, 1]; title = "Along-slope vorticity", axis_kwargs...)
# ax_v = Axis(fig[3, 1]; title = "Along-slope velocity (v)", axis_kwargs...)
ax_u = Axis(fig[3, 1]; title = "cross-slope velocity (u)", axis_kwargs...)

n = Observable(1)

ωy = @lift ds["ωy"][:, :, $n]
B = @lift ds["B"][:, :, $n]
hm_ω = heatmap!(ax_ω, xω, zω, ωy, colorrange = (-0.015, +0.015), colormap = :balance)
Colorbar(fig[2, 2], hm_ω; label = "s⁻¹")
ct_b = contour!(ax_ω, xb, zb, B, levels=-1e-3:0.5e-4:1e-3, color=:black)

u = @lift ds["u"][:, :, $n]
u_max = @lift maximum(abs, ds["u"][:, :, $n])

hm_v = heatmap!(ax_u, xu, zu, u, colorrange = (-V∞, +V∞), colormap = :balance)
Colorbar(fig[3, 2], hm_v; label = "m s⁻¹")
ct_b = contour!(ax_u, xb, zb, B, levels=-1e-3:0.5e-4:1e-3, color=:black)

times = collect(ds["time"])
title = @lift "t = " * string(prettytime(times[$n]))
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

current_figure() #hide
fig

# Finally, we record a movie.

frames = 1:length(times)

record(fig, "tilted_bottom_boundary_layer.mp4", frames, framerate=12) do i
    n[] = i
end
nothing #hide

# ![](tilted_bottom_boundary_layer.mp4)

# Don't forget to close the NetCDF file!

close(ds)
