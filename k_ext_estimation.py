#Covariance analysis - Love number estimation.
# Load required standard modules
#%%
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
import scipy as sc
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.numerical_simulation.estimation_setup import parameter as gfv
from tudatpy.kernel.astro import gravitation
from tudatpy.io import save2txt


def get_gravity_ganymede():

    mu_ganymede = spice.get_body_properties("Ganymede", "GM", 1)[0]*10**9 #same as Sam's: 9877.5555788329
    radius_ganymede = spice.get_body_properties("Ganymede", "RADII", 3)[0]*10**3 #is: 2631.2 km (spice) was: 2634 km (SAM)
    cosine_coef = np.zeros((31, 31))
    sine_coef = np.zeros((31, 31))

    cosine_coef[0, 0] = 1.0
    cosine_coef[2, 0] = -127.8e-6 / gravitation.legendre_normalization_factor(2, 0)
    cosine_coef[2, 2] = 38.3e-6 / gravitation.legendre_normalization_factor(2, 2)
    
    return environment_setup.gravity_field.spherical_harmonic(mu_ganymede, radius_ganymede, cosine_coef, sine_coef, "IAU_Ganymede")

def getKaulaConstraint(kaula_constraint_multiplier, degree):
    return kaula_constraint_multiplier / degree ** 2

def apply_kaula_constraint_a_priori(kaula_constraint_multiplier, max_deg_gravity, indices_cosine_coef, indices_sine_coef, inv_apriori):

    index_cosine_coef = indices_cosine_coef[0]
    index_sine_coef = indices_sine_coef[0]

    for deg in range(2, max_deg_gravity + 1):
        kaula_constraint =getKaulaConstraint(kaula_constraint_multiplier, deg)
        for order in range(deg + 1):
            inv_apriori[index_cosine_coef, index_cosine_coef] = kaula_constraint ** -2
            index_cosine_coef += 1
        for order in range(1, deg + 1):
            inv_apriori[index_sine_coef, index_sine_coef] = kaula_constraint ** -2
            index_sine_coef += 1

# Load spice kernels
path = os.getcwd()
print(path)
kernels = [path+'/kernel_juice.bsp', path+'/kernel_noe.bsp']
spice.load_standard_kernels(kernels)

# Input simulation time
number_of_days = 5
start_gco = DateTime( 2035, 5, 21, 15, 0, 0.0 ).epoch( )  # Start of GCO500
end_gco = start_gco + number_of_days * constants.JULIAN_DAY # circular orbital phase

# Create default body settings
bodies_to_create = ['Ganymede', 'Jupiter', 'Sun', 'Saturn', 'Europa', 'Io', 'Callisto','Earth']

# Create default body settings for bodies_to_create
global_frame_origin = "Ganymede"
global_frame_orientation = 'J2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Compute rotation rate for Ganymede
mu_jupiter = spice.get_body_properties("Jupiter", "GM", 1)[0] * 1.0e9
initial_state_ganymede = spice.get_body_cartesian_state_at_epoch("Ganymede", "Jupiter", "J2000", "None", start_gco)
keplerian_state_ganymede = element_conversion.cartesian_to_keplerian(initial_state_ganymede, mu_jupiter)
rotation_rate_ganymede = np.sqrt(mu_jupiter/keplerian_state_ganymede[0]**3)
# Set rotation model settings
initial_orientation_ganymede = spice.compute_rotation_matrix_between_frames("J2000", "IAU_Ganymede", start_gco)


# Finally, we are ready to get Ganymede's rotation and gravity field model, adding it the body_settings

# Get Rotation settings
body_settings.get("Ganymede").rotation_model_settings = environment_setup.rotation_model.simple(
    "J2000", "IAU_Ganymede", initial_orientation_ganymede, start_gco, rotation_rate_ganymede)
# Get Gravity field settings
body_settings.get("Ganymede").gravity_field_settings = get_gravity_ganymede()

#Atmosphere Ganymede
density_scale_height = 43300 # (Plainaki 2015)
surface_density = 7.5e-12 # (Plainaki 2015) and calculated from rho_H20 at surface = 2.5e14 m^-3

body_settings.get("Ganymede").atmosphere_settings = environment_setup.atmosphere.exponential(
     density_scale_height, surface_density)

#Use extended Love numbers
tide_raising_bodies = ["Jupiter"] 
love_numbers: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
# Adding k Love numbers for forcing (2, 0) and LV(1,1) of 1%

love_numbers[(2, 0)] = {(2, 0): 3.07297e-01,(2, 2): 2.28408e-07}
"""    (2, -2): 2.28408e-07,
    (2, 0): 3.07297e-01,
    (2, 2): 2.28408e-07,
    (3, -3): 2.27317e-10,
    (3, -1): 7.83920e-05,
    (3, 1): -7.83920e-05,
    (3, 3): -2.27317e-10,
    (4, -2): 3.25230e-08,
    (4, 0): -4.11388e-08,
    (4, 2): 3.25230e-08,
    (5, -3): 1.75113e-11,
    (5, -1): -2.43185e-11,
    (5, 1): 2.43185e-11,
    (5, 3): -1.75113e-11
}"""

gravity_field_variation_list = list()
#Implementing extended Love numbers for time-varying gravity field
#gravity_field_variation_list.append(environment_setup.gravity_field_variation.solid_body_tide("Jupiter",0.0,2))
gravity_field_variation_list.append(environment_setup.gravity_field_variation.mode_coupled_solid_body_tide(tide_raising_bodies,love_numbers))

#Implementing standard k2 Love number for time-varying gravity field
#gravity_field_variation_list.append(environment_setup.gravity_field_variation.solid_body_tide("Jupiter", 0.3, 2))

body_settings.get( "Ganymede" ).gravity_field_variation_settings = gravity_field_variation_list

body_settings.add_empty_settings( 'JUICE' )
body_settings.get('JUICE').constant_mass = 2500 #Rounded up dry mass of 2420kg from https://www.esa.int/Science_Exploration/Space_Science/Juice/Juice_spacecraft_specs to include leftover fuel

# Create empty multi-arc ephemeris for JUICE
empty_ephemeris_dict = dict()
juice_ephemeris = environment_setup.ephemeris.tabulated(
    empty_ephemeris_dict,
    global_frame_origin,
    global_frame_orientation)
juice_ephemeris.make_multi_arc_ephemeris = True
body_settings.get("JUICE").ephemeris_settings = juice_ephemeris

bodies = environment_setup.create_system_of_bodies(body_settings)

# Create aerodynamic coefficient interface settings, and add to vehicle
reference_area = 100.0
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area, [drag_coefficient, 0, 0]
)
environment_setup.add_aerodynamic_coefficient_interface(
    bodies, "JUICE", aero_coefficient_settings)


# Create radiation pressure settings, and add to vehicle
reference_area_radiation = 100.0
radiation_pressure_coefficient = 1.2
occulting_bodies = {"Sun": ["Ganymede"]}
juice_srp_settings = environment_setup.radiation_pressure.cannonball_radiation_target(reference_area_radiation, radiation_pressure_coefficient, occulting_bodies)
environment_setup.add_radiation_pressure_target_model(bodies, "JUICE", juice_srp_settings)

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.
acceleration_settings_on_vehicle = dict(
    
    Ganymede=[propagation_setup.acceleration.spherical_harmonic_gravity(30, 30), #only C_20 and C_22 populated rest is empty for estimation of gravity field.
              propagation_setup.acceleration.aerodynamic(),
              propagation_setup.acceleration.empirical()
              ],
    
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Jupiter=[
        propagation_setup.acceleration.spherical_harmonic_gravity(4, 0)
    ],
    Saturn=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Europa=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Io=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Callisto=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations dictionary.
acceleration_settings = {'JUICE': acceleration_settings_on_vehicle}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define initial state.
system_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='JUICE',
    observer_body_name='Ganymede',
    reference_frame_name='J2000',
    aberration_corrections='NONE',
    ephemeris_time = start_gco )

print("Verification of spacecraft altitude:")
print(np.linalg.norm(system_initial_state[0:3])/1e3-2631.2," km above Ganymede's surface")

# Define degree/order combinations for which to save acceleration contributions
spherical_harmonic_terms_ganymede = [ (0,0), (2,0), (2,2) ]
spherical_harmonic_terms_jupiter = [ (0,0), (2,0), (4,0) ]

dependent_variables_to_save = [
    propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'),
    
    propagation_setup.dependent_variable.latitude("JUICE", "Ganymede"),
    propagation_setup.dependent_variable.longitude("JUICE", "Ganymede"),
    
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "JUICE", "Ganymede"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.aerodynamic_type, "JUICE", "Ganymede"
    ),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm( "JUICE", "Ganymede", spherical_harmonic_terms_ganymede),
    
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "JUICE", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.radiation_pressure_type, "JUICE", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "JUICE", "Saturn"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "JUICE", "Io"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "JUICE", "Europa"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "JUICE", "Callisto"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "JUICE", "Jupiter"
    ),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm( "JUICE", "Jupiter", spherical_harmonic_terms_jupiter),
    propagation_setup.dependent_variable.total_gravity_field_variation_acceleration("JUICE","Ganymede")
    ]

# Create propagation arcs during GCO (one day long)
arc_duration = 1.0 * constants.JULIAN_DAY

arc_start_times = []
arc_end_times = []
arc_start = start_gco
while arc_start+arc_duration <= end_gco:
    arc_start_times.append(arc_start)
    arc_end_times.append(arc_start+arc_duration+3600.0) #1h overlap
    arc_start += arc_duration

# Extract total number of (propagation) arcs during GCO
nb_arcs = len(arc_start_times)
print(f'Total number of arcs for GCO500: {nb_arcs}')

# Initial states per arc, extracted from JUICE's SPICE ephemeris (JUICE's SPICE ID is -28) at the start of each propagation arc.
initial_states = []
for i in range(nb_arcs):
    initial_states.append(spice.get_body_cartesian_state_at_epoch("-28", "Ganymede", "J2000", "None", arc_start_times[i]))

# Create numerical integrator settings.
time_step = 100.0
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(time_step, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78)

# Define arc-wise propagator settings
propagator_settings_list = []
for i in range(nb_arcs):
    propagator_settings_list.append(propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate, initial_states[i], arc_start_times[i], integrator_settings, propagation_setup.propagator.time_termination(arc_end_times[i]),
        propagation_setup.propagator.cowell, dependent_variables_to_save))

# Combine propagator settings list into multi-arc propagator settings
propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

# Propagate dynamics and retrieve simulation results - 1st Arc
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
simulation_results = simulator.propagation_results.single_arc_results

# Extract numerical solution for states and dependent variables
state_history = simulation_results[0].state_history
dependent_variables = simulation_results[0].dependent_variable_history

state = np.vstack(list(state_history.values()))
depvars = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - start_gco / constants.JULIAN_DAY for t in time ]

fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(111, projection='3d')

# draw sphere
R = 2631.2 #km
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = R*np.cos(u)*np.sin(v)
y = R*np.sin(u)*np.sin(v)
z = R*np.cos(v)
ax.plot_wireframe(x, y, z, color="r")
ax.plot(state[:,0]/1e3,state[:,1]/1e3,state[:,2]/1e3,color="blue",label='JUICE GCO500')
ax.set_aspect("equal")

# Plot JUICE ground track 
fig, ax2 = plt.subplots()
ganymede_map = path+'/ganymede_map.jpg'
ax2.imshow(plt.imread(ganymede_map), extent = [0, 360, -90, 90])
# Resolve 2pi ambiguity longitude
longitude = depvars[:,7]
latitude = depvars[:,6]
for k in range(len(longitude)):
    if longitude[k] < 0:
        longitude[k] = longitude[k] + 2.0 * np.pi
ax2.plot(longitude*180/np.pi, latitude*180.0/np.pi, '.', markersize=3, color='red')

ax2.set_xlabel('Longitude [deg]')
ax2.set_ylabel('Latitude [deg]')
ax2.set_xticks(np.arange(0, 361, 40))
ax2.set_yticks(np.arange(-90, 91, 30))
ax2.set_title('JUICE Ground Track')

# Define ground stations on Earth
station_names = ["Malargue"]
station_coordinates = {station_names[0]: [1550.0, np.deg2rad(-35.0), np.deg2rad(-69.0)]}

for station in station_names:
    environment_setup.add_ground_station(
        bodies.get_body("Earth"), station, station_coordinates[station], element_conversion.geodetic_position_type)

# Define link ends for two-way Doppler and range observables, for each ESTRACK station
link_ends = []
for station in station_names:
    link_ends_per_station = dict()
    link_ends_per_station[observation.transmitter] = observation.body_reference_point_link_end_id("Earth", station)
    link_ends_per_station[observation.receiver] = observation.body_reference_point_link_end_id("Earth", station)
    link_ends_per_station[observation.reflector1] = observation.body_origin_link_end_id("JUICE")
    link_ends.append(link_ends_per_station)

# Tracking arcs - equal to one day propagation arcs
tracking_arc_duration = arc_duration
tracking_arcs_start = []
tracking_arcs_end = []
for arc_start in arc_start_times:
    tracking_arc_start = arc_start
    if( arc_start == arc_start_times[ 0 ] ):
        tracking_arcs_start.append(tracking_arc_start+3600.0)
    else:
        tracking_arcs_start.append(tracking_arc_start)
    tracking_arcs_end.append(tracking_arc_start + tracking_arc_duration)

# Create observation settings for each link ends and observable
# Define light-time calculations settings
light_time_correction_settings = observation.first_order_relativistic_light_time_correction(["Sun"])
# Define range biases settings
biases = []
for i in range(nb_arcs):
    biases.append(np.array([0.0]))
range_bias_settings = observation.arcwise_absolute_bias(tracking_arcs_start, biases, observation.receiver)

# Define observation settings list
observation_settings_list = []
for link_end in link_ends:
    link_definition = observation.LinkDefinition(link_end)
    observation_settings_list.append(observation.n_way_doppler_averaged(link_definition, [light_time_correction_settings]))
    observation_settings_list.append(observation.two_way_range(link_definition, [light_time_correction_settings], range_bias_settings))

# Define observation simulation times for both Doppler and range observables
doppler_cadence = 60.0
range_cadence = 300.0

observation_times_doppler = []
observation_times_range = []
for i in range(nb_arcs):
    # Doppler observables
    time = tracking_arcs_start[i]
    while time <= tracking_arcs_end[i]:
        observation_times_doppler.append(time)
        time += doppler_cadence
    # Range observables
    time = tracking_arcs_start[i]
    while time <= tracking_arcs_end[i]:
        observation_times_range.append(time)
        time += range_cadence

observation_times_per_type = dict()
observation_times_per_type[observation.n_way_averaged_doppler_type] = observation_times_doppler
observation_times_per_type[observation.n_way_range_type] = observation_times_range

# Define observation settings for both observables, and all link ends (i.e., all ESTRACK stations)
observation_simulation_settings = []
for link_end in link_ends:
    # Doppler
    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.n_way_averaged_doppler_type, observation.LinkDefinition(link_end), observation_times_per_type[observation.n_way_averaged_doppler_type]))
    # Range
    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.n_way_range_type, observation.LinkDefinition(link_end), observation_times_per_type[observation.n_way_range_type]))


# Create viability settings which define when an observation is feasible
viability_settings = []

# For all tracking stations (for now only Malargue), check if elevation is sufficient
for station in station_names:
    viability_settings.append(observation.elevation_angle_viability(["Earth", station], np.deg2rad(15.0)))
# Check whether Ganymede or Jupiter are occulting the signal
viability_settings.append(observation.body_occultation_viability(["JUICE", ""], "Ganymede"))
viability_settings.append(observation.body_occultation_viability(["JUICE", ""], "Jupiter"))
viability_settings.append(observation.body_occultation_viability(["JUICE", ""], "Io"))
viability_settings.append(observation.body_occultation_viability(["JUICE", ""], "Europa"))
viability_settings.append(observation.body_occultation_viability(["JUICE", ""], "Callisto"))

# # Check if Sun avoidance angle is sufficiently large
viability_settings.append(observation.body_avoidance_viability(["JUICE", ""], "Sun", np.deg2rad(5.0)))

# Apply viability checks to all simulated observations
observation.add_viability_check_to_all(observation_simulation_settings, viability_settings)


# Define parameters to estimate
# Add arc-wise initial states of the JUICE spacecraft wrt Ganymede
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies, arc_start_times)

# Add Ganymede's gravitational parameter
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Ganymede"))

# Add Ganymede's gravity field spherical harmonics coefficients
max_deg_ganymede_gravity = 15
parameter_settings.append(estimation_setup.parameter.spherical_harmonics_c_coefficients("Ganymede", 2, 0, max_deg_ganymede_gravity, max_deg_ganymede_gravity))
parameter_settings.append(estimation_setup.parameter.spherical_harmonics_s_coefficients("Ganymede", 2, 1, max_deg_ganymede_gravity, max_deg_ganymede_gravity))

# Add Ganymede's rotational parameters
parameter_settings.append(estimation_setup.parameter.constant_rotation_rate("Ganymede"))
parameter_settings.append(estimation_setup.parameter.rotation_pole_position("Ganymede"))

# When propagating the dynamics of the spacecraft during each arc,  we might want to take into account for possible errors in the accelerometer calibration of the spacecraft. 
# These are modelled by introducing an **empirical acceleration** components along each spatial dimension. 

# Add arc-wise empirical accelerations acting on the JUICE spacecraft
acc_components = {estimation_setup.parameter.radial_empirical_acceleration_component: [estimation_setup.parameter.constant_empirical],
                  estimation_setup.parameter.along_track_empirical_acceleration_component: [estimation_setup.parameter.constant_empirical],
                  estimation_setup.parameter.across_track_empirical_acceleration_component: [estimation_setup.parameter.constant_empirical]}
parameter_settings.append(estimation_setup.parameter.arcwise_empirical_accelerations("JUICE", "Ganymede", acc_components, arc_start_times))

# Add ground stations' positions
for station in station_names:
    parameter_settings.append(estimation_setup.parameter.ground_station_position("Earth", station))

#Estimation of Love numbers
estimated_love_numbers: dict[tuple[int, int], list[tuple[int, int]]] = {}
#Option 1: Only k2 Love number

estimated_love_numbers[(2, 0)] = [(2, 0),(2, 2)] #forcing D/O : response D/O
#parameter_settings.append(estimation_setup.parameter.order_invariant_k_love_number("Ganymede",2,tide_raising_bodies))
#Option 2: Extended ove numbers

#estimated_love_numbers[(2, 0)] = [(2, -2),(2,0),(2,2),(3,-3),(3,-1),(3,1),(3,3),(4,-2),(4,0),(4,2),(5,-3),(5,-1),(5,1),(5,3)]
parameter_settings.append(estimation_setup.parameter.mode_coupled_k_love_numbers("Ganymede",estimated_love_numbers,tide_raising_bodies))

# Create parameters to estimate object
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings) 
estimation_setup.print_parameter_names(parameters_to_estimate)
nb_parameters = len(parameters_to_estimate.parameter_vector)
print(f'Total number of parameters to estimate: {nb_parameters}')

# Create the estimator
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings_list, propagator_settings)

# Simulate all observations
simulated_observations = estimation.simulate_observations(observation_simulation_settings, estimator.observation_simulators, bodies)
print("observation vector size: ",simulated_observations.observation_vector_size)

sorted_observations = simulated_observations.sorted_observation_sets
doppler_obs_times_malargue_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times if t <= end_gco]
plt.figure()
d = 1
for i in doppler_obs_times_malargue_first_arc:
    if i < d*24:
        plt.scatter(d, i-24*(d-1),s=1,c='black')
    else:
        d += 1
        plt.scatter(d,i-24*(d-1),s=1,c='black')

#plt.scatter(doppler_obs_times_malargue_first_arc, 3.0 * np.ones((len(doppler_obs_times_malargue_first_arc),1 )))
plt.xlabel('Days')
plt.ylabel('Observation times [h]')
plt.xticks(np.arange(1,nb_arcs+1,1))
plt.ylim([0, 24])
plt.title('Viable Observations')
plt.grid()

"""
# Define a priori covariance matrix
inv_apriori = np.zeros((nb_parameters, nb_parameters))

# Set a priori constraints for JUICE state(s)
a_priori_position = 5.0e3 # initial a priori uncertainty on JUICE's position (m)
a_priori_velocity = 0.5 # initial a priori uncertainty on JUICE's velocity (m/s)
indices_states = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.arc_wise_initial_body_state_type, ("JUICE", "")))[0]
for i in range(indices_states[1]//6): # for loop to create the inverse apriori covariance matrix (with 1/a_priori_position^2 and 1/a_priori_velocity^2) on the diagonal
    for j in range(3):
        inv_apriori[indices_states[0]+i*6+j, indices_states[0]+i*6+j] = a_priori_position**-2  # a priori position
        inv_apriori[indices_states[0]+i*6+j+3, indices_states[0]+i*6+j+3] = a_priori_velocity**-2  # a priori velocity

# Set a priori constraint for Ganymede's gravitational parameter
a_priori_mu = 0.03e9
indices_mu = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.gravitational_parameter_type, ("Ganymede", "")))[0]
for i in range(indices_mu[1]):
    inv_apriori[indices_mu[0]+i, indices_mu[0]+i] = a_priori_mu**-2

# Set a priori constraint for Ganymede's gravity field coefficients
nb_cosine_coef = (max_deg_ganymede_gravity+1) * (max_deg_ganymede_gravity+2) // 2 - 3  # the minus 3 accounts for degrees 0 and 1 coefficients which are not estimated
indices_cosine_coef = (nb_arcs*6+1, nb_cosine_coef)
# indices_cosine_coef = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.spherical_harmonics_cosine_coefficient_block_type, ("Ganymede", "")))[0]
indices_sine_coef = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.spherical_harmonics_sine_coefficient_block_type, ("Ganymede", "")))[0]

# Apply Kaula's constraint to Ganymede's gravity field a priori
kaula_constraint_multiplier = 1.0e-5
apply_kaula_constraint_a_priori(kaula_constraint_multiplier, max_deg_ganymede_gravity, indices_cosine_coef, indices_sine_coef, inv_apriori)

# Overwrite Kaula's rule with existing uncertainties for C20 and C22
apriori_C20 = 2.9e-6
apriori_C22 = 0.87e-6
inv_apriori[indices_cosine_coef[0], indices_cosine_coef[0]] = apriori_C20**-2
inv_apriori[indices_cosine_coef[0]+2, indices_cosine_coef[0]+2] = apriori_C22**-2

# Set tight constraint for C21, S21, and S22
inv_apriori[indices_cosine_coef[0]+1, indices_cosine_coef[0]+1] = 1.0e-12**-2
inv_apriori[indices_sine_coef[0], indices_sine_coef[0]] = 1.0e-12**-2
inv_apriori[indices_sine_coef[0]+1, indices_sine_coef[0]+1] = 1.0e-12**-2

# Set a priori constraints for empirical accelerations acting on JUICE
a_priori_emp_acc = 1.0e-8
indices_emp_acc = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.arc_wise_empirical_acceleration_coefficients_type, ("JUICE", "Ganymede")))[0]
for i in range(indices_emp_acc[1]):
    inv_apriori[indices_emp_acc[0] + i, indices_emp_acc[0] + i] = a_priori_emp_acc ** -2

# Set a priori constraints for ground station positions
a_priori_station = 0.03
for station in station_names:
    indices_station_pos = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.ground_station_position_type, ("Earth", station)))[0]
    for i in range(indices_station_pos[1]):
        inv_apriori[indices_station_pos[0] + i, indices_station_pos[0] + i] = a_priori_station ** -2

# Retrieve full vector of a priori constraints
apriori_constraints = np.reciprocal(np.sqrt(np.diagonal(inv_apriori)))

print('A priori constraints')
print(apriori_constraints)
"""

# Define covariance input settings
covariance_input = estimation.CovarianceAnalysisInput(simulated_observations)#,inv_apriori, consider_parameters_covariance)
covariance_input.define_covariance_settings(reintegrate_variational_equations=False, save_design_matrix=True)

# Apply weights to simulated observations
doppler_noise = 12.0e-6
range_noise = 0.2
weights_per_observable = {observation.n_way_averaged_doppler_type: doppler_noise ** -2,
                          observation.n_way_range_type: range_noise ** -2}
covariance_input.set_constant_weight_per_observable(weights_per_observable)

# Perform the covariance analysis
covariance_output = estimator.compute_covariance(covariance_input)

# Retrieve covariance results
correlations = covariance_output.correlations
covariance = covariance_output.covariance
formal_errors = covariance_output.formal_errors
partials = covariance_output.weighted_design_matrix

# Print the covariance matrix
print("Correlations", correlations, "\n")
print("Covariance", covariance, "\n")
print("Formal_errors", formal_errors, "\n")
print("Partials", partials, "\n")

# Plot weighted partials
plt.figure(figsize=(9, 6))
plt.imshow(np.log10(np.abs(partials)), aspect='auto', interpolation='none')
cb = plt.colorbar()
cb.set_label('log10(weighted partials)')
plt.title("Weighted partials")
plt.ylabel("Index - Observation")
plt.xlabel("Index - Estimated Parameter")
plt.tight_layout()

np.savetxt('H_matrix.txt',partials)
