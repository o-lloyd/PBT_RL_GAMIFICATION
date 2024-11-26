import pydicom as dcm
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from structures_to_display import STRUCTURES_TO_DISPLAY , PRIORITIES
import cv2
from pydicom.valuerep import IS 
import pickle
from scipy.interpolate import interp1d

base_path = r"data"

dicom_data = {}

ARRAY_SIZE_X , ARRAY_SIZE_Y = 512, 512

def correct_ROI_name(name):

    A = False
    for struct in STRUCTURES_TO_DISPLAY.keys():
        if name in STRUCTURES_TO_DISPLAY[struct]:
            name = struct
            A = True
            break
    if not A:
        print(f'{name} not in Stucture list')
            

    return name

def STRING(ID):
    new_string = str(int(ID))

    return new_string

def extract_slice(s):
    # Split the string using dot as a separator
    parts = s.split('.')
    # Return the last element of the list
    return parts[-1]

def study_id_num(scan):

#For now using Study date as is the only attribute that all files have and is only consistent between patients 

    id_num = f"{scan.StudyInstanceUID}"

    return id_num

def file_data_type(scan):

    media_storage = scan.Modality

    if media_storage == "CT":
        file_data_type = "CT"
    elif media_storage == "RTSTRUCT":
        file_data_type = "Struct"

    else:
        #print(media_storage)
        return 0

    return file_data_type

def ct_slice_num(scan):

    slice_num = scan.InstanceNumber

    return slice_num


def generate_dcm_path(base_path, pattern="**/*.dcm"):
    # Generate the full pattern
    full_pattern = os.path.join(base_path, pattern)

    # Use glob to get all matching files recursively
    dcm_file_paths = glob.glob(full_pattern, recursive=True)

    return dcm_file_paths


def additional_info(scan, dicom_data):

    id_num = study_id_num(scan)

    additional_data_fields = ["Pixel Spacing", "Slice Thickness", "Image Position", "Rows", "Columns", "SOPInstanceUID", "InstanceNumber"]
    additional_data_values = [scan.PixelSpacing, scan.SliceThickness, scan.ImagePositionPatient, scan.Rows, scan.Columns, [], []]

    for data_field, data_value in zip(additional_data_fields, additional_data_values):
        if data_field not in dicom_data[id_num]:
            dicom_data[id_num][data_field] = data_value

    dicom_data[id_num]["SOPInstanceUID"].append(scan.SOPInstanceUID)
    dicom_data[id_num]["InstanceNumber"].append(int(scan.InstanceNumber))

    return dicom_data

def target_dose(name):

    if name == 'PTV56':
        target = 56.0
    elif name == 'PTV70':
        target = 70.0
    else:
        target = 0.0
    
    return target

def max_dose(name):

    AVOID = ['Brain Stem', 'Spinal Cord', 'Oral Cavity', 'Avoid']

    if name == 'PTV56':
        max_val = 71.0
    elif name == 'PTV70':
        max_val = 95.0
    elif name in AVOID:
        max_val = 5.0
    elif name == 'External':
        max_val = 20.0
    else:
        max_val = 20.0

    return max_val


hu = np.array([-1025, -616, -50, 144, 234, 281, 449, 718, 1307, 2600, 5000, 5100, 10000 ])
spower = np.array([0.1, 0.384, 0.995, 1.103, 1.153, 1.177, 1.265, 1.400, 1.700, 2.378, 3.635, 5.100, 10])

hu_to_sp_interpolator = interp1d(hu, spower)

def HutoSP(x):

    y = hu_to_sp_interpolator(x)

    return y


def ctScanList(scan, dicom_data):

    data_type = file_data_type(scan)

    id_num = study_id_num(scan)
    i = 1

    if data_type == "CT":

        slice_num = ct_slice_num(scan)  
        
        if id_num not in dicom_data:
            dicom_data[id_num] = {}

        if "CT" not in dicom_data[id_num]:
            dicom_data[id_num]["CT"] = [] 

        if "Density" not in dicom_data[id_num]:
            dicom_data[id_num]["Density"] = [] 

        if "Struct" not in dicom_data[id_num]:
            dicom_data[id_num]["Struct"] = {}

        if "Slice Ref" not in dicom_data[id_num]:
            dicom_data[id_num]["Slice Ref"] = []

        dicom_data = additional_info(scan, dicom_data)

        array2D = scan.pixel_array

        z_cord = scan.ImagePositionPatient[2]

        # Some data files are already converted to HU so if the initial corner which should be air is below
        # -200 then it must be already converted, if not it can be adjusted
        if array2D[0,0] > -200:

            array2D = (array2D * float(scan.RescaleSlope))

            array2D = (array2D + float(scan.RescaleIntercept))


        arrayDens2D = hu_to_sp_interpolator(array2D) 

        #arrayDens2D = np.transpose(arrayDens2D)
        #print(arrayDens2D.shape)

        # Append the 2D array to the list of slices for the corresponding ID
        dicom_data[id_num]["CT"].append((slice_num, array2D))
        dicom_data[id_num]["Density"].append((slice_num, arrayDens2D))
        dicom_data[id_num]["Slice Ref"].append((slice_num, z_cord))

        # If you want to keep the slices sorted by slice_num, you can do so
        # This is useful when you want to stack them in the correct order
        dicom_data[id_num]["CT"].sort(key=lambda x: x[0])
        dicom_data[id_num]["Density"].sort(key=lambda x: x[0])


    return dicom_data

def ctScanStacker(dicom_data):
        
    for id_num in dicom_data:
        arrays = [tpl[1] for tpl in dicom_data[id_num]["CT"]]
        arrays2 = [tpl[1] for tpl in dicom_data[id_num]["Density"]]
        # Stack the arrays to create a 3D array
        dicom_data[id_num]["CT"] = np.stack(arrays, axis=0)
        dicom_data[id_num]["Density"] = np.stack(arrays2, axis=0)

    return dicom_data

def CT_Scan_arrays(dicom_data, ct_file_paths):


    for path in ct_file_paths:
        scan = dcm.dcmread(path)

        if hasattr(scan, 'InstanceNumber'):
            dicom_data = ctScanList(scan, dicom_data)
        
    dicom_data = ctScanStacker(dicom_data)

    for id_num in dicom_data:


        dicom_data[id_num]["Slice Ref"] = np.array(dicom_data[id_num]["Slice Ref"])
    

    return dicom_data

def structure_collection(dicom_data, ct_file_paths):

    for path in ct_file_paths:
        ds_struct = dcm.dcmread(path)

        data_type = file_data_type(ds_struct)

        #print(data_type)

        if data_type == "Struct":  
            id_num = study_id_num(ds_struct) 
            print(id_num)
            

            # roi_names = []


            for roi in ds_struct.StructureSetROISequence:

                # if roi.ROIName not in roi_names:
                #     roi_names.append(roi.ROIName)
                
                print(roi.ROIName)
                print(id_num)

                # if roi.ROIName in STRUCTURES_TO_DISPLAY:
                if any(roi.ROIName in name_list for name_list in STRUCTURES_TO_DISPLAY.values()):

                    name = correct_ROI_name(roi.ROIName)
                    
                    #print(roi)
                    contour = {}    

                    contour['name'] = name
                    contour['number'] = roi.ROINumber

                    #print(name)

                    dicom_data[id_num]["Struct"][name] = {}
                    dicom_data[id_num]["Struct"][name]["ROI Number"] = roi.ROINumber

                    for contour_data in ds_struct.ROIContourSequence:
                        if contour_data.ReferencedROINumber == roi.ROINumber:
                            # Assuming the contour data for the current CT slice is in a ContourSequence
                            # And each contour in the sequence has a ContourData attribute
                            # And the CT slice SOP Instance UID is in a ContourImageSequence within each contour
                            C = 0


                            image_position = dicom_data[id_num]["Image Position"]
                            pixel_spacing = dicom_data[id_num]["Pixel Spacing"]
                            slice_thickness = dicom_data[id_num]["Slice Thickness"]
                            rows = dicom_data[id_num]["Rows"]
                            columns = dicom_data[id_num]["Columns"] 
                            z_coord = dicom_data[id_num]["Image Position"][2]
                            


                            #print(type(dicom_data[id_num]["SOPInstanceUID"]))
                            
                            
                            #for UID in dicom_data[id_num]["SOPInstanceUID"]:

                            for contour in contour_data.ContourSequence:
                                indicies_array = contour.ContourData

                                cord = np.where(indicies_array[2] == dicom_data[id_num]["Slice Ref"][:,1])
                                slice_index = str(int(dicom_data[id_num]["Slice Ref"][cord,0][0][0]))


                                if slice_index not in dicom_data[id_num]["Struct"][name]:
                                    dicom_data[id_num]["Struct"][name][slice_index] = {}
                                    

                                num_of_roi_cords = len(indicies_array[0::3])

                                spacing_array_x = np.full(num_of_roi_cords, pixel_spacing[0])
                                spacing_array_y = np.full(num_of_roi_cords, pixel_spacing[1])

                                array_adjustment_x = np.full(num_of_roi_cords, rows)
                                array_adjustment_y = np.full(num_of_roi_cords, columns)

                                x_roi_cord_positions =  ((indicies_array[0::3] - np.full(len(indicies_array[0::3]) ,image_position[0] ))/ spacing_array_x) #+ (array_adjustment_x / 2)
                                y_roi_cord_positions =  ((indicies_array[1::3] - np.full(len(indicies_array[1::3]) ,image_position[1] ))/ spacing_array_y) #+ (array_adjustment_x / 2)

                                pixel_coordinates = list(zip(x_roi_cord_positions, y_roi_cord_positions))

                                # print(pixel_coordinates)
                                coordinates_array = np.array(pixel_coordinates, dtype=np.int32)
                                coordinates_array_cv = coordinates_array.reshape((-1, 1, 2))
                                binary_contour_array_1 = np.zeros((rows, columns), dtype=np.uint8)
                                binary_contour_array_2 = np.zeros((rows, columns), dtype=np.uint8)
                                binary_contour_array_max = np.zeros((rows, columns), dtype=np.uint8)

                                target = target_dose(name) 
                                #print(target) 
                                max_dose_val = max_dose(name)

                                cv2.polylines(binary_contour_array_1, [coordinates_array_cv], isClosed=True, color=1, thickness=1)

                                cv2.fillPoly(binary_contour_array_2, [coordinates_array_cv], target)

                                cv2.fillPoly(binary_contour_array_max, [coordinates_array_cv], max_dose_val)

                                dicom_data[id_num]["Struct"][name][slice_index]["Binary Mask array"] = binary_contour_array_1

                                dicom_data[id_num]["Struct"][name][slice_index]["Binary Fill array"] = binary_contour_array_2

                                dicom_data[id_num]["Struct"][name][slice_index]["Binary MAX Fill array"] = binary_contour_array_max
                            
    return dicom_data
                            
                            
def stack_all_binary_arrays(dicom_data):
    # Loop over all ids
    for id in dicom_data.keys():
        # Loop over all rois for each id
        for roi in dicom_data[id]["Struct"].keys():
            # Get the list of slice numbers (num) for the current id and roi
            

            def is_string_integer(k):
                # Check if k is a string
                if isinstance(k, str):
                    # Try converting the string to an integer
                    try:
                        int(k)
                        return True
                    except ValueError:
                        return False
                # Check if k is an integer
                elif isinstance(k, int):
                    return False
                # If k is neither string nor integer, return False
                else:
                    return False

            rows = dicom_data[id]["Rows"]
            columns = dicom_data[id]["Columns"]
            z_indexes = len(dicom_data[id]["SOPInstanceUID"])

            z_indexes = len(dicom_data[id]["CT"][:,0,0])

            slice_index_new = sorted([k for k in dicom_data[id]["Struct"][roi].keys() if is_string_integer(k)], key=lambda x: int(x))

            NEW_binary_fill_3d = np.zeros((z_indexes, rows, columns))
            NEW_binary_outline_3d = np.zeros((z_indexes, rows, columns))
            NEW_binary_max_fill_3d = np.zeros((z_indexes, rows, columns))

            print(f'{roi}  {id}')
            print(slice_index_new)
            
            for num in slice_index_new:

                slice_data = dicom_data[id]["Struct"][roi][str(num)]

                if isinstance(slice_data, dict) and "Binary Fill array" and "Binary Mask array"  in slice_data:

                    try:
                        NEW_binary_fill_3d[int(num)-1,:,:] = slice_data["Binary Fill array"]
                        NEW_binary_max_fill_3d[int(num)-1,:,:] = slice_data["Binary MAX Fill array"]
                        NEW_binary_outline_3d[int(num)-1,:,:] = slice_data["Binary Mask array"]
                    except:
                        print('exception')


            dicom_data[id]["Struct"][roi]["NEW BINARY FILL"] = NEW_binary_fill_3d
            dicom_data[id]["Struct"][roi]["NEW MAX BINARY FILL"] = NEW_binary_max_fill_3d
            dicom_data[id]["Struct"][roi]["NEW BINARY MASK"] = NEW_binary_outline_3d

                
    return dicom_data

def merge_target_roi_arrays(dicom_data, priorities=PRIORITIES):
    for id in dicom_data.keys():
        # Determine the shape of one of the ROI arrays
        sample_roi_name = next(iter(priorities))

        shape = dicom_data[id]["Struct"][sample_roi_name]["NEW BINARY FILL"].shape

        # Initialize the target array with zeros
        target_array = np.zeros(shape)
        max_array = np.zeros(shape)

        # Sort the ROI names based on their priorities
        sorted_rois = sorted(priorities, key=priorities.get, reverse=False)
        #print(sorted_rois)

        # Iterate over each sorted ROI
        for roi in sorted_rois:

            if roi in dicom_data[id]['Struct'].keys():

                # Access the ROI array from dicom_data
                roi_array = dicom_data[id]["Struct"][roi]["NEW BINARY FILL"]
                roi_max_array = dicom_data[id]["Struct"][roi]["NEW MAX BINARY FILL"]

                # Update the target array with the ROI values
                target_array = np.where(roi_array != 0, roi_array, target_array)
                max_array = np.where(roi_max_array != 0, roi_max_array, max_array)
                

        dicom_data[id]['Target Array 3D'] = target_array
        dicom_data[id]['Max Array 3D'] = max_array

    return dicom_data

def del_useless(dicom_data):

    key_to_keep = ['Max Array 3D', 'Target Array 3D', 'Density', 'Pixel Spacing']

    ids = dicom_data.keys()

    for  id in ids:

        key_to_delete = [key for key in dicom_data[id].keys() if key not in key_to_keep]

        for key in key_to_delete:
            
            del dicom_data[id][key]

    return dicom_data


dicom_data = {}

def __main__(dicom_data):

    ct_file_paths = generate_dcm_path(base_path, pattern="**/*.dcm")
    #print(ct_file_paths)

    dicom_data = CT_Scan_arrays(dicom_data, ct_file_paths)

    dicom_data = structure_collection(dicom_data, ct_file_paths)

    dicom_data = stack_all_binary_arrays(dicom_data)

    dicom_data = merge_target_roi_arrays(dicom_data)

    dicom_data = del_useless(dicom_data)

    with open(r'pickles/example_dicom_processed_data.pkl', 'wb') as file:
        pickle.dump(dicom_data, file)

    return 0
# #print(dicom_data.keys())
# for id in dicom_data: 

#     plt.imshow(dicom_data[id]["CT"][30], cmap=plt.cm.bone)
#     x_coords, y_coords = zip(*dicom_data[id]["Struct"]["GTV"]['30']["Coordinate array"])
#     plt.plot(x_coords, y_coords, label="ROI")
#     #print(dicom_data[id]["Struct"]["GTV"])
#    # print(dicom_data[id]["Pixel Spacing"])
#    # print(dicom_data[id]["Slice Thickness"])
#     #print(dicom_data[id]["Struct"]["GTV"]['30']["Coordinate array"])

#     #print(list(dicom_data[id]["Struct"]["GTV"].keys()))

#    # print(dicom_data[id]["SOPInstanceUID"])
#    # print(dicom_data[id]["Struct"]["avoid"])
#     plt.legend()
#     #plt.show()

#     plt.imshow(dicom_data[id]["Struct"]["GTV"]['30']["Binary Mask array"], cmap='gray')
#     #plt.show()


__main__(dicom_data)
