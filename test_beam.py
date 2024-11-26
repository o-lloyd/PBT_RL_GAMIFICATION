import pickle
import numpy as np
from ProtonBeam import ProtonBeamV4
import matplotlib.pyplot as plt
import os

class beam_test:
    def __init__(self, uid, dicom_data_path, beam_data_path, slice_index):

        with open(beam_data_path, 'rb') as file:
            self.energy_data = pickle.load(file)

        with open(dicom_data_path, 'rb') as file:
            self.dicom_data = pickle.load(file)

        self.density_map = self.dicom_data[uid]['Density'][int(slice_index)-1]

        self.grid_x = self.density_map.shape[1]
        self.grid_y = self.density_map.shape[0]


    def single_beam(self, cord_x, cord_y, angle):

        beam_map, bool = ProtonBeamV4(cord_x, cord_y, angle,self.grid_x,self.grid_y,1.1,1.1,self.density_map,self.energy_data)

        if not bool:
            print("Invalid Coordinates")
            return 1

        beam_map[beam_map < 0.1] = np.nan

        plt.imshow(self.density_map, cmap='bone')
        im = plt.imshow(beam_map, cmap='jet', alpha=0.5)
        cbar = plt.colorbar(im)
        cbar.set_label(label="Dose (Gy)", size=15)
        plt.show()

        return 0

    def phantom(self, cord_x, cord_y, angle):

        beam_map, bool, dose_vals, dist_vals, dose_interpolation, energy_val = ProtonBeamV4(cord_x, cord_y, angle,self.grid_x,self.grid_y,1.1,1.1,self.density_map,self.energy_data, phantom=True)

        if not bool:
            print("Invalid Coordiantes")
            return 1

        dist_vals=dist_vals[dose_vals > 0]
        dose_vals=dose_vals[dose_vals > 0]
        
        plt.figure(figsize=(12, 6)) 

        plt.scatter(dist_vals, dose_vals, color='hotpink', s=25, marker="x", label="True Dose")

        plt.plot(np.linspace(0,dist_vals[-1], 500), dose_interpolation(np.linspace(0,dist_vals[-1], 500)), alpha=0.7,color="black", label="Interpolation")

        plt.title(f'Proton beam dose values against water equivalent distance ({energy_val:.0f} MeV)', fontsize=16)
        plt.xlabel('Water Equivalent Distance', fontsize=14)
        plt.ylabel('Dose (Gy)', fontsize=14)
        plt.legend()

        plt.grid(True) 
       
        plt.show()




def example():
    x_c = 250
    y_c = 250
    angle = 100
    slice_index = 47
    
    dicom_path = r"pickles/example_dicom_processed_data.pkl"
    beam_path = r"pickles/beam_data.pkl"

    # Check if the DICOM file exists
    if not os.path.exists(dicom_path):
        print(f"Error: The file {dicom_path} does not exist. Please first run the data processing script to create this file.")
        return  # Exit the function if the file does not exist

    test_slice = beam_test("1.3.6.1.4.1.22213.2.26556", dicom_path, beam_path, slice_index)
    test_slice.phantom(x_c, y_c, angle)
    test_slice.single_beam(x_c, y_c, angle)

example()