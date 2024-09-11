#Covariance analysis - Love number estimation.
# Load required standard modules

import numpy as np
from matplotlib import pyplot as plt
import os, sys
import scipy as sc
# Load required tudatpy modules
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

# Load spice kernels
spice.load_standard_kernels()
current_directory = os.getcwd()
spice.load_kernel( current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp" );

A = 7
B = 1
C = 0

simulation_start_epoch = 35.4 * constants.JULIAN_YEAR + A * 7.0 * constants.JULIAN_DAY + B * constants.JULIAN_DAY + C * constants.JULIAN_DAY / 24.0
simulation_end_epoch = simulation_start_epoch + 30.0 * constants.JULIAN_DAY / 24.0 #344h from assignment

# Create default body settings
bodies_to_create = ['Ganymede', 'Jupiter', 'Sun', 'Saturn', 'Europa', 'Io', 'Callisto','Earth']

# Create default body settings for bodies_to_create
global_frame_origin = "Ganymede"
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

density_scale_height = 40000
surface_density = 2e-9

body_settings.get("Ganymede").atmosphere_settings = environment_setup.atmosphere.exponential(
     density_scale_height, surface_density)

#Gravity field variations from tides exerted by Jupiter on Ganymede.
tide_raising_bodies = ["Jupiter"] #list[str]

# Initialize extended love_numbers dictionary
love_numbers: dict[tuple[int, int], dict[tuple[int, int], float]] = {}

# love_numbers[(2, 0)] = {(2, 0): 0.32} #forcing D/O : response D/O : value of Love number

# Adding k Love numbers for forcing (2, 0) and LV(1,1) of 1%
love_numbers[(2, 0)] = {
    (0, 0): 3.99191e-24,
    (1, -1): -9.01374e-29,
    (1, 0): 0.00000e+00,
    (1, 1): 1.66960e-28,
    (2, -2): 2.28408e-07,
    (2, -1): 0.00000e+00,
    (2, 0): 3.07297e-01,
    (2, 1): 0.00000e+00,
    (2, 2): 2.28408e-07,
    (3, -3): 2.27317e-10,
    (3, -2): 0.00000e+00,
    (3, -1): 7.83920e-05,
    (3, 0): 0.00000e+00,
    (3, 1): -7.83920e-05,
    (3, 2): 0.00000e+00,
    (3, 3): -2.27317e-10,
    (4, -3): 0.00000e+00,
    (4, -2): 3.25230e-08,
    (4, -1): 0.00000e+00,
    (4, 0): -4.11388e-08,
    (4, 1): 0.00000e+00,
    (4, 2): 3.25230e-08,
    (4, 3): 0.00000e+00,
    (5, -3): 1.75113e-11,
    (5, -1): -2.43185e-11,
    (5, 1): 2.43185e-11,
    (5, 3): -1.75113e-11
}
#forcing D/O : response D/O : value of Love number

#print(love_numbers[(2, 0)][(2, 0)])
gravity_field_variation_list = list()
gravity_field_variation_list.append( environment_setup.gravity_field_variation.mode_coupled_solid_body_tide(tide_raising_bodies, love_numbers)) 

body_settings.get( "Ganymede" ).gravity_field_variation_settings = gravity_field_variation_list

bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################

# Create vehicle object
bodies.create_empty_body( 'JUICE' )
bodies.get('JUICE').mass = 2000

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
occulting_bodies = ["Ganymede"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)
environment_setup.add_radiation_pressure_interface(
    bodies, "JUICE", radiation_pressure_settings)

###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.
acceleration_settings_on_vehicle = dict(
    
    Ganymede=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
              propagation_setup.acceleration.aerodynamic()
              ],
    
    Sun=[
        propagation_setup.acceleration.cannonball_radiation_pressure(),
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
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time = simulation_start_epoch )

# Define required outputs

# Define degree/order combinations for which to save acceleration contributions
spherical_harmonic_terms_ganymede = [ (0,0), (2,0), (2,2) ]
spherical_harmonic_terms_jupiter = [ (0,0), (2,0), (4,0) ]

dependent_variables_to_save = [
    propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'),
    
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
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "JUICE", "Sun"
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
    propagation_setup.dependent_variable.relative_position("JUICE", "Sun"),
    propagation_setup.dependent_variable.relative_position("JUICE", "Ganymede"),
    propagation_setup.dependent_variable.total_gravity_field_variation_acceleration("JUICE","Ganymede")
    ]

# Create numerical integrator settings.
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    fixed_step_size
)

# Create propagation settings.
termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables = dependent_variables_to_save
)

propagator_settings.print_settings.print_initial_and_final_conditions = True


# Define the position of the ground station on Earth
station_altitude = 0.0
delft_latitude = np.deg2rad(52.00667)
delft_longitude = np.deg2rad(4.35556)

# Add the ground station to the environment
environment_setup.add_ground_station(
    bodies.get_body("Earth"),
    "TrackingStation",
    [station_altitude, delft_latitude, delft_longitude],
    element_conversion.geodetic_position_type)

# Define the uplink link ends for one-way observable
link_ends = dict()
link_ends[observation.transmitter] = observation.body_reference_point_link_end_id("Earth", "TrackingStation")
link_ends[observation.receiver] = observation.body_origin_link_end_id("JUICE")

# Create observation settings for each link/observable
link_definition = observation.LinkDefinition(link_ends)
observation_settings_list = [observation.one_way_doppler_instantaneous(link_definition)]

# Define observation simulation times for each link 
observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 60.0)
observation_simulation_settings = observation.tabulated_simulation_settings(
    observation.one_way_instantaneous_doppler_type,
    link_definition,
    observation_times
)

noise_level = 0.003E-6
observation.add_gaussian_noise_to_observable(
    [observation_simulation_settings],
    noise_level,
    observation.one_way_instantaneous_doppler_type
)

# Create viability settings
viability_setting = observation.elevation_angle_viability(["Earth", "TrackingStation"], np.deg2rad(15))
observation.add_viability_check_to_all(
    [observation_simulation_settings],
    [viability_setting]
)

# Setup parameters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
estimated_love_numbers: dict[tuple[int, int], list[tuple[int, int]]] = {}
#estimated_love_numbers[(2, 0)] = [(2, 0)] #forcing D/O : response D/O
estimated_love_numbers[(2, 0)] = [(0, 0),(1, -1),(1, 0),(1, 1),(2, -2),(2,-1),(2,0),(2,1),(2,2),(3,-3),(3,-2),(3,-1),(3,0),(3,1),(3,2),(3,3),(4,-3),(4,-2),(4,-1),
                                  (4,0),(4,1),(4,2),(4,3),(5,-3),(5,-1),(5,1),(5,3)]

parameter_settings.append(estimation_setup.parameter.mode_coupled_k_love_numbers("Ganymede",estimated_love_numbers,tide_raising_bodies))

# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
    
# Create the estimator
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings_list,
    propagator_settings)

# Simulate required observations
simulated_observations = estimation.simulate_observations(
    [observation_simulation_settings],
    estimator.observation_simulators,
    bodies)

# Create input object for covariance analysis
covariance_input = estimation.CovarianceAnalysisInput(simulated_observations)

# Set methodological options
covariance_input.define_covariance_settings(reintegrate_variational_equations=False, save_design_matrix=True)

# Define weighting of the observations in the inversion
weights_per_observable = {estimation_setup.observation.one_way_instantaneous_doppler_type: noise_level ** -2}
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


