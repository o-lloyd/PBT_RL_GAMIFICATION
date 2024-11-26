import numpy as np
import Modules.beam_functions as fn
import matplotlib.pyplot as plt
from Modules.dose_interpolater import beam_interpolation 

def create_midpoint_array_numpy(rows, columns):

    """
    Generates a 3D array of midpoints for every voxel
    """
    # Generate grid of indices
    x_indices, y_indices = np.meshgrid(np.arange(columns), np.arange(rows))

    # Adjust indices to get midpoints
    x_midpoints =  x_indices + 0.5
    y_midpoints =  y_indices + 0.5

    # Combine into a single array of midpoint coordinates
    midpoints = np.dstack((y_midpoints, x_midpoints))

    return midpoints

def calculate_distances_and_intercepts(midpoint_array, m, c, d_x, d_y, grid_x, grid_y):

    """
    Calculates the distance of every voxel from the beam line perperndicularily and the coordinates of the intercept

    Does not currently account for changes in dx of the voxels. Should be implemented usign some trigernometry
    """
    

    x_coords = midpoint_array[:, :, 1]
    y_coords = midpoint_array[:, :, 0]

    distances = np.abs(m  * x_coords  -  y_coords  + c) / np.sqrt(m**2 + 1)

    distances[distances > 35] = np.nan #Want to remove values at an irrelevant distance
    
    # Calculate intercept points
    x_i = ((m * y_coords + x_coords - m*c) / (m**2 + 1))
    y_i = ((m**2 * y_coords + m*x_coords + c) / (m**2 + 1))
    
    # Round down the intercept points
    x_i = np.floor(x_i).astype(int) -1
    y_i = np.floor(y_i).astype(int) -1

    # Identify indices where x_i or y_i falls outside the range 0-511
    out_of_bounds = (x_i < 0) | (x_i > grid_x -1) | (y_i < 0) | (y_i > grid_y-1)

    # Set coordinates to (0, 0) for out-of-bounds indices
    x_i[out_of_bounds] = 0
    y_i[out_of_bounds] = 0

    cord_intercept_array = np.stack((x_i, y_i), axis=-1)
    
    return distances, cord_intercept_array

def gaussian(x, c):
    """
    Gaussian function.
    Parameters:
    - x: array of x values.
    - a: amplitude of the Gaussian.
    - b: mean (center) of the Gaussian.
    - c: standard deviation (spread) of the Gaussian.
    """
    return  np.exp(-(x)**2 / (2 * c**2))

def spread_on_dose(dist_array, cord_int_array, dose_map):
    
    row_indices, col_indices = cord_int_array[..., 0], cord_int_array[..., 1]

    guass_array = np.squeeze(gaussian(dist_array, 4.0))

    selected_doses = dose_map[col_indices, row_indices] 

    dose_spread = np.array(selected_doses) * np.array(guass_array)


    return selected_doses, guass_array, dose_spread



def cord_check(coords, spot_x, spot_y, grid_x, grid_y):

    beam_bool = True

    peak_distance_cord = np.where((coords == [spot_x-1,spot_y-1]).all(axis=1))[0]

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x-2,spot_y-1]).all(axis=1))[0]

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x-1,spot_y-2]).all(axis=1))[0]

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x-1,spot_y-2]).all(axis=1))[0] 

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x-2,spot_y-2]).all(axis=1))[0]   

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x-3,spot_y-2]).all(axis=1))[0]   

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x+1,spot_y]).all(axis=1))[0]  

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x,spot_y+1]).all(axis=1))[0]   

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x+1,spot_y+1]).all(axis=1))[0]   

    if peak_distance_cord.size ==0 :
        peak_distance_cord = np.where((coords == [spot_x+1,spot_y+2]).all(axis=1))[0] 

    if peak_distance_cord.size ==0 :
        print(f'Cords are {spot_x}, {spot_y}')

    if peak_distance_cord.size == 0:
        beam_bool = False


        return np.zeros((grid_y,grid_x)), beam_bool
    
    
    return peak_distance_cord, beam_bool

def ProtonBeamV4(spot_x, spot_y, angle, grid_x, grid_y, d_x, d_y, density_array, energy_data, phantom=False):

    spot_x = np.floor(spot_x)
    spot_y = np.floor(spot_y)

    beam_bool = True

    old_angle = angle

    # Convert angle to radians
    angle = angle / 180 * np.pi

    # Beam's straight line function parameters
    m = np.tan(np.pi / 2 - angle)
    c = spot_y - m * spot_x

    # Beam coordinates on edges of grid
    point_1 = [0, 0]
    point_2 = [0, 0]

    points = []

    if angle == np.pi / 2 or angle == 3/2 * np.pi:
        point_1 = [0, fn.straight_line_function(0, m, c)]
        point_2 = [grid_x, fn.straight_line_function(grid_x, m, c)]
    elif angle == 0 or angle == np.pi:
        point_1 = [fn.inverse_straight_line_function(0, m, c, spot_x), 0]
        point_2 = [fn.inverse_straight_line_function(grid_y, m, c, spot_x), grid_y]
    else:
        
        if 0 <= fn.inverse_straight_line_function(0, m, c, spot_x) <= grid_x:
            points.append([fn.inverse_straight_line_function(0, m, c, spot_x), 0])
        if 0 <= fn.inverse_straight_line_function(grid_y, m, c, spot_x) <= grid_x:
            points.append([fn.inverse_straight_line_function(grid_y, m, c, spot_x), grid_y])
        if 0 <= fn.straight_line_function(0, m, c) <= grid_y:
            points.append([0, fn.straight_line_function(0, m, c)])
        if 0 <= fn.straight_line_function(grid_x, m, c) <= grid_y:
            points.append([grid_x, fn.straight_line_function(grid_x, m, c)])
        
        if len(points) == 2:
            point_1 = points[0]
            point_2 = points[1]

        elif len(points) == 3: 
            point_1 = points[1]
            point_2 = points[2]

    if point_1 == [0, 0]:
        print("No point 1")
    if point_2 == [0, 0]:
        print("No point 2")

    # print(points)

    # Left, right, up, down x and y values, i.e. ignoring beam direction
    x_left = min(point_1[0], point_2[0])
    x_right = max(point_1[0], point_2[0])

    y_down = min(point_1[1], point_2[1])
    y_up = max(point_1[1], point_2[1])

    # x and y values of voxel boundaries
    x_voxel_boundaries = np.arange(np.ceil(x_left), np.floor(x_right), d_x)
    y_voxel_boundaries = np.arange(np.ceil(y_down), np.floor(y_up), d_y)

    # Where the beam crosses over a voxel boundary
    x_cross = np.array([point_1[0]])
    x_cross = np.append(x_cross, fn.inverse_straight_line_function(y_voxel_boundaries, m, c, point_1[1]))
    x_cross = np.append(x_cross, x_voxel_boundaries)

    y_cross = np.array([point_1[1]])
    y_cross = np.append(y_cross, y_voxel_boundaries)
    y_cross = np.append(y_cross, fn.straight_line_function(x_voxel_boundaries, m, c))

    beam_cross_coords = np.vstack((x_cross, y_cross))
    beam_cross_coords = np.transpose(beam_cross_coords)

    # Order beam cross coordinates along beam direction depending on angle
    if angle != 0 and angle != np.pi:
        beam_cross_coords = beam_cross_coords[beam_cross_coords[:, 0].argsort()]
        if angle < np.pi:
            beam_cross_coords = beam_cross_coords[::-1]
    else:
        beam_cross_coords = beam_cross_coords[beam_cross_coords[:, 1].argsort()]
        if angle == 0:
            beam_cross_coords = beam_cross_coords[::-1]

    distances = fn.calc_distance(beam_cross_coords[:-1], beam_cross_coords[1:])

    coords = np.floor(beam_cross_coords).astype(int)[:-1] - 1
    
    sp_distances = density_array[coords[:, 1], coords[:, 0]] * distances

    
    peak_distance_cord, beam_bool = cord_check(coords, spot_x, spot_y, grid_x, grid_y)

    if not beam_bool:
        print(spot_x, spot_y, old_angle)
        return np.zeros((grid_y,grid_x)), beam_bool
    

    peak_distance_cord = peak_distance_cord[0]
    
    sp_distance_culm = np.cumsum(sp_distances)
    peak_distance = sp_distance_culm[peak_distance_cord]

    dose_interpolation, energy_bool, energy_val = beam_interpolation(energy_data, peak_distance)

    if energy_bool == True:

        dose_deposition = dose_interpolation(sp_distance_culm)

        dose_map = np.zeros((grid_y, grid_x))

        dose_map[coords[:,1], coords[:,0]] = dose_deposition

        shifted_up = np.zeros_like(dose_map)
        shifted_down = np.zeros_like(dose_map)
        shifted_left = np.zeros_like(dose_map)
        shifted_right = np.zeros_like(dose_map)

        # Shift values in each direction
        shifted_up[:-1, :] = dose_map[1:, :]  # Up
        shifted_down[1:, :] = dose_map[:-1, :]  # Down
        shifted_left[:, :-1] = dose_map[:, 1:]  # Left
        shifted_right[:, 1:] = dose_map[:, :-1]  # Right


        shifted_stacked = np.stack([shifted_up, shifted_down, shifted_left, shifted_right])
        max_shifted = np.max(shifted_stacked, axis=0)

        # Identify positions where original array is 0
        mask_zeros = dose_map == 0

        # Initialize a copy of the original array to accumulate the results
        resulting_array = np.copy(dose_map)

        # Update the original array with the maximum of the shifted values, but only where the original value is 0
        resulting_array[mask_zeros] += max_shifted[mask_zeros]

        dose_map = resulting_array

        midpoint_array = create_midpoint_array_numpy(grid_y, grid_x)
        distances_array, cord_intercept_array = calculate_distances_and_intercepts(midpoint_array, m, c, d_x, d_y, grid_x, grid_y)

        dose_spread_map, guass_array, final_spread = spread_on_dose(distances_array, cord_intercept_array, dose_map)

        final_spread[np.isnan(final_spread)] = 0

        final_spread = np.where(final_spread > 0.1, final_spread,0)
        
        if phantom:
            return final_spread, beam_bool, dose_deposition, sp_distance_culm, dose_interpolation, energy_val
            
        else:
            return final_spread, beam_bool
        
    
    else: 
        if not phantom:
            beam_bool = False
            return np.zeros((grid_y,grid_x)), beam_bool
        else:
            beam_bool=False
            return np.zeros((grid_y,grid_x)), beam_bool, [],[]


