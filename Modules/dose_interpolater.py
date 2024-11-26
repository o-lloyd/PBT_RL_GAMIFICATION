import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def select_closest_beam_energy(energy_data, beam_distance):

    # print(beam_distance)

    arr = (energy_data['peak distances'][:,0] - beam_distance)

    # print(arr)

    lower_energy_index = np.where(arr < 0, arr, -np.inf).argmax()
    higher_energy_index = np.where(arr > 0, arr, np.inf).argmin()

    #print(lower_energy_index)

    lower_energy = energy_data['peak distances'][lower_energy_index,1]
    higher_energy = energy_data['peak distances'][higher_energy_index,1]

    # print(f' Lower energy: {lower_energy}')
    # print(f' Higher energy: {higher_energy}')

    return [lower_energy, higher_energy]

# peaks = select_closest_beam_energy(energy_data, 140)
# print(peaks)
def ln_function(x, a, b, c):
        return a * np.log(x + b) + c

def linear_beam_peak_fit(energy_data):

    x = energy_data['peak distances'][:,0]
    y = energy_data['peak distances'][:,1]


    # Fit the function to the data
    popt, pcov = curve_fit(ln_function, x, y)

    y_fit = ln_function(x, *popt)

    
    chi_square = np.sum((y - y_fit) ** 2 / y_fit)

    A =0

    if A ==1:
    # Plot the data and the fit
        plt.scatter(x, y, label='Data')
        plt.plot(x, ln_function(x, *popt), color='red', label=f'Fit:  {popt[0]:.2f} * ln(x + {popt[1]:.2f}) {popt[2]:.2f}')
        plt.xlabel('Position of Bragg Peak WEPL (mm)')
        plt.ylabel('Proton Beam Energy (MeV)')
        plt.title(f'Fit For Beam Energy from Peak Location ($\chi^2$ = {chi_square:.3f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('LogFitForBeamPeak.png',dpi=400)
        plt.show()

    return popt

# beam_data_path = r"C:\Users\hayde\Documents\MPhys\Sem2\ProtonBeamV2\ClinicalBeamData.pkl"
# with open(beam_data_path, 'rb') as file:
#     beam_data = pickle.load(file)

# popt = linear_beam_peak_fit(beam_data)
# print(popt)


def beam_interpolation(energy_data, beam_distance):

    energy_bool = True

    INTERP = 0

    if INTERP ==1:
        interpolated_dose_energy = interp1d(
            energy_data['peak distances'][:,0], 
            energy_data['peak distances'][:,1], 
            kind='linear', 
            fill_value="extrapolate", 
            bounds_error=False
        )

    #popt = linear_beam_peak_fit(energy_data)
    popt = [ 213.69713872,   221.13565533, -1117.10647303]

    
    energy = ln_function(beam_distance, *popt)

    #energy = interpolated_dose_energy(beam_distance)

    # print(f'Energy approx: {energy}')
    # print(f'Beam Distance: {beam_distance}')



    energies = [key for key in energy_data.keys() if key.replace('.', '').isdigit()]
    if energy in energies:
        distances = energy_data[str(energy)]['dx']
        doses = energy_data[str(energy)]['dose']
        return interp1d(distances, doses, kind='linear', fill_value="extrapolate")
    else:
        # Extrapolate using the closest energies available
        # energy = float(energy)
        # lower_energy = max([e for e in energies if float(e) < energy])
        # higher_energy = min([e for e in energies if float(e) > energy])


        closest_energies = select_closest_beam_energy(energy_data, beam_distance)

        if closest_energies[0] == closest_energies[1]:
            energy_bool=False

        # print(f'Closest energies {closest_energies}')

        lower_energy = str(closest_energies[0])
        higher_energy = str(closest_energies[1])
        
        if closest_energies[0] == closest_energies[1]:
            low_multiplier = 1
        else:

            low_multiplier = (energy - closest_energies[0])/(closest_energies[1]-closest_energies[0])
        high_multiplier = 1 - low_multiplier

        lower_peak_index = np.argmax(energy_data[lower_energy]['dose'])
        higher_peak_index = np.argmax(energy_data[higher_energy]['dose'])

        lower_peak_val = energy_data[lower_energy]['dx'][lower_peak_index]
        higher_peak_val = energy_data[higher_energy]['dx'][higher_peak_index]

        # print(f'Lower Peak distance {lower_peak_val}')
        # print(f'Higher Peak distance {higher_peak_val}')

        lower_peak_distance_diff = beam_distance - lower_peak_val
        higher_peak_distance_diff = beam_distance - higher_peak_val

        # print(lower_peak_distance_diff)
        # print(higher_peak_distance_diff)

        # print('\n------------------\n')

        adjusted_lower_energies = energy_data[lower_energy]['dx'] + lower_peak_distance_diff
        adjusted_higher_energies = energy_data[higher_energy]['dx'] + higher_peak_distance_diff



        lower_interpolation = interp1d(adjusted_lower_energies, energy_data[lower_energy]['dose'], kind='linear', fill_value="extrapolate")
        higher_interpolation = interp1d(adjusted_higher_energies, energy_data[higher_energy]['dose'], kind='linear', fill_value="extrapolate")


        return lambda x: (high_multiplier * higher_interpolation(x) + low_multiplier * lower_interpolation(x)), energy_bool, energy   # Interpolate between the two energies
    






