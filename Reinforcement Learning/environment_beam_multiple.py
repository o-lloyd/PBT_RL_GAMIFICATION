import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces # Action and Observation spaces
import modeules.functions as fn
import modeules.ProtonBeamV4 as ProtonBeamV4
from matplotlib.colors import LinearSegmentedColormap

class Beam_Env(gym.Env):
    def __init__(self, target_dose_array, max_dose_array, contours, voxel_size, density_array, energy_data, skin_fill, max_step_count, binary_index):

        self.INDEX = (slice(50, 350), slice(150,350))
    
        self.target_dose_array = target_dose_array
        self.max_dose_array = max_dose_array
        self.max_index = np.where(self.max_dose_array > 1)
        self.target_index = np.where(self.target_dose_array > 1)
        self.non_tumour_index = np.where((self.target_dose_array < 5) & (self.max_dose_array > 1))
        self.normalised_array = np.full(self.target_dose_array.shape,-2.0, dtype=np.float64)
        self.normalised_array[self.max_index] = 0
        self.reward_sum = 0
        # self.normalised_template = np.stack((self.target_dose_array,self.max_dose_array, self.normalised_array),axis=0)
        self.binary_info = binary_index

        self.normalised_template = np.stack((self.normalised_array, self.binary_info),axis=0)
        

        self.normalised_array = np.stack((self.normalised_array), axis=0)

        self.array_shape = target_dose_array.shape
        self.grid_x = self.array_shape[1]
        self.grid_y = self.array_shape[0]
        self.skin_fill = skin_fill
        self.range_y = 294
        self.range_x = 194
        self.x_offset = 3
        self.y_offset = 3
        self.angle_offset = 15
        self.coordinate_action_modifier = 3
        self.angle_action_modifier = 6
        # self.max_x = max(self.skin_fill[1])
        # self.max_y = max(self.skin_fill[0])
        # self.min_x = min(self.skin_fill[1])
        # self.min_y = min(self.skin_fill[0])
        self.voxel_dx = voxel_size[1]
        self.voxel_dy = voxel_size[0]
        self.step_count = 0
        self.max_step_count = max_step_count
        self.contours = contours
        self.density_array = density_array
        self.energy_data = energy_data
        self.dose_maps = []
        self.number_of_beams = len(self.dose_maps)
        self.dose_array = np.zeros(self.array_shape, dtype=np.float32)


        # self.normalised_array = fn.make_normalised_arrays(self.normalised_array,self.target_dose_array,self.max_dose_array,self.dose_array,self.target_index,self.max_index)
        self.normalised_array = fn.make_normalised_array(self.normalised_array,self.target_dose_array,self.max_dose_array,self.dose_array,self.target_index,self.max_index)

        # self.dose_norm_array = fn.make_normalised_array(self.target_dose_array, self.max_dose_array, self.dose_array, self.skin_fill)
        self.beam_coords = []
        self.add_beam(self.range_x / 2 + self.x_offset, self.range_y / 2 + self.y_offset, 3)




        # min_x = 0.4  # Minimum x coordinate
        # max_x = np.floor(self.range_x/self.coordinate_action_modifier) # Maximum x coordinate
        # min_y = 0.4# Minimum y coordinate
        # max_y = np.floor(self.range_y/self.coordinate_action_modifier) # Maximum y coordinate
        # min_angle = 0.4  # Minimum angle
        # max_angle = 5  # Maximum angle

        # # Define the bounds for the action space
        # low = np.array([min_x, min_y, min_angle])
        # self.high_multiplier = np.array([max_x, max_y, max_angle])
        # high = np.array([0.6,0.6,0.6])

        # # Define the continuous action space
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([np.floor(self.range_x/self.coordinate_action_modifier), np.floor(self.range_y/self.coordinate_action_modifier), self.angle_action_modifier]) # Action, which dose, x/y values of spot, beam angle
        
        
        
        
        
        
        
        #self.observation_space = spaces.Box(low=-1, high=150, shape=(self.dose_array.shape), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2, high=100, shape=(2, 300, 200), dtype=np.float32)
        #self.observation_space = spaces.Box(low=-2, high=10, shape=(2,512,512), dtype=np.float32)
        self.truncated = False
        self.done = False

    
    
    def step(self, action):
        """
        Performs an action in the environment. This can be removing, moving, or adding a dose, on any of the available doses. If the
        action is to move a dose, then a direction van be specified as well. For each action a reward is calculated based on dose
        values. Returns an observation (the environment's state), the reward, and whether the episode is finished: either by the tumour
        having enough dose, or the step count reaching its maximum value.
        """

        if self.done == True:
            raise ValueError("Episode has already ended. Call reset() to start a new episode.")

        reward = 0
        # print('\n --')
        # print(f'prioir act{action}')

        # action *= self.high_multiplier
        
        action_x, action_y, action_angle = action

        # action_x *= self.range_x
        # action_y *= self.range_y
        # action_angle *= 360

        # action_x = int(action_x)
        # action_y = int(action_y)

        

        # action_x = int(action_x)
        # action_y = int(action_y)

        action_x *= self.coordinate_action_modifier
        action_y *= self.coordinate_action_modifier
        action_x += self.x_offset
        action_y += self.y_offset
        action_angle *= 360 / self.angle_action_modifier
        action_angle += self.angle_offset

        # print(f'post act{action_x, action_y, action_angle}')
        # print('--- \n')

        #beam_bool = self.add_beam(action_x, action_y, action_angle)
        beam_bool = self.add_sobp(action_x, action_y, action_angle)

        if not beam_bool:
            reward -= 20
        else:

            y_indices, x_indices = self.skin_fill

            # y_index_tumour, x_index_tumour = self.target_index

            # Create a mask where both x and y coordinates match the action coordinates
            mask = (x_indices == action_x) & (y_indices == action_y)

            # mask_target = (x_index_tumour == action_x) & (y_index_tumour == action_y)

            # Check if any pair matches the action indices
            if not np.any(mask):
                reward -= 1
            else:
                
                # if not np.any(mask_target):
                #     reward -= 0.5

                # reward_1 = 0
                # reward_2 = 0

                #reward += np.sum(fn.reward_function(self.dose_array[self.max_index], self.target_dose_array[self.max_index], self.max_dose_array[self.max_index]))

                reward += fn.tumour_reward(self.dose_array[self.target_index],self.target_dose_array[self.target_index],self.max_dose_array[self.target_index]) / (2*len(self.target_index[0]))

                reward += fn.non_tumour_reward(self.dose_array[self.non_tumour_index], self.max_dose_array[self.non_tumour_index]) / (2*len(self.non_tumour_index[0]))

                # target_under_index = np.where((self.dose_array[self.target_index] < self.target_dose_array[self.target_index]))

                # target_under_max_= np.where((self.dose_array[self.target_index] < self.max_dose_array[self.target_index]) & (self.dose_array[self.target_index] > self.target_dose_array[self.target_index]))
                # reward_1 += len(target_under_max_[0])

                # reward_1 += np.sum(self.dose_array[self.target_index][target_under_index] / self.target_dose_array[self.target_index][target_under_index])
                # reward_1 /= len(self.target_index[0])

                # #reward += np.sum(np.where((self.dose_array[self.skin_fill] < self.max_dose_array[self.skin_fill]) & (self.dose_array[self.skin_fill] > self.target_dose_array[self.skin_fill]) , 1, 0))

                # reward_2 -= np.sum(np.where((self.dose_array[self.skin_fill] > self.max_dose_array[self.skin_fill]) & (self.max_dose_array[self.skin_fill] > 1), (self.dose_array[self.skin_fill] / self.max_dose_array[self.skin_fill]) - 0.5, 0))

                #reward /= len(self.skin_fill[0])

                # reward += reward_1 + reward_2

                
        # End if tumour has recieved enough dose or if maximum step count is exceeded
        if (self.dose_array >= self.target_dose_array).all() and (self.dose_array <= self.max_dose_array).all():
            self.done = True
            print("Tumour dosed")
            self.render(False, True, "completed_environment.png")
        else:
            self.done = fn.is_episode_complete(self, self.max_step_count)
        
        self.step_count += 1
        
        observation = self.update_observation_space()

        # if self.step_count in [20,40,60,80,100]:
        #     for contour in self.contours:
        #         plt.contour(contour[0], levels=[0.5], colors=contour[1], zorder=4, linewidths=2, linestyles=contour[2])
        #         plt.plot(0, 0, color=contour[1], linestyle=contour[2], label=contour[3])
        #         plt.imshow(contour[0], zorder=3, alpha=0)
                

        #     doseArray = self.dose_array
        #     legend_pos = (1.35,1.15)
        #     doseArray[doseArray<0.1] = np.nan
        #     im = plt.imshow(doseArray, cmap='jet', zorder=6, alpha=0.4)
        #     cbar = plt.colorbar(im)
        #     cbar.set_label(label="Dose (Gy)", size=15)
        #     legend_pos = (1.44,1.15)
        #     plt.imshow(self.density_array, cmap='bone', zorder=1)
        #     plt.legend(bbox_to_anchor=legend_pos, ncol=4, fontsize=14)
        #     plt.show()
        
        self.reward_sum += reward

        return observation, reward, self.done, self.truncated, {}
    

    def old_reward():
        """ OlD REWARD

            if not beam_bool:
                reward -= 1
            else:

                y_indices, x_indices = self.skin_fill

                # Create a mask where both x and y coordinates match the action coordinates
                mask = (x_indices == action_x) & (y_indices == action_y)

                # Check if any pair matches the action indices
                if not np.any(mask):
                    reward -= 1
                else:

                    with np.errstate(divide='ignore', invalid='ignore'):
                        reward += sum(np.where((self.dose_array[self.skin_fill] < self.target_dose_array[self.skin_fill]) & (self.target_dose_array[self.skin_fill] != 0) , self.dose_array[self.skin_fill] / self.target_dose_array[self.skin_fill], 0))

                    with np.errstate(divide='ignore', invalid='ignore'):
                        reward += sum(np.where((self.dose_array[self.skin_fill] < self.max_dose_array[self.skin_fill]) & (self.dose_array[self.skin_fill] > self.target_dose_array[self.skin_fill]) , 1, 0))

                    with np.errstate(divide='ignore', invalid='ignore'):
                        reward -= sum(np.where((self.dose_array[self.skin_fill] > self.max_dose_array[self.skin_fill]) & (self.max_dose_array[self.skin_fill] != 0), (self.dose_array[self.skin_fill] / self.max_dose_array[self.skin_fill]) -1, 0))

                    reward /= len(self.skin_fill[0])
        """


    def update_observation_space(self):
        """
        Updates the normalisation arrays as the dose array changes.
        """
        # observation = np.stack((self.max_dose_array, fn.make_normalised_array(self.normalised_array, self.target_dose_array, self.max_dose_array, self.dose_array, self.target_index, self.max_index)),axis=0)
        #observation = fn.make_normalised_arrays(self.normalised_array, self.target_dose_array, self.max_dose_array, self.dose_array, self.target_index, self.max_index)

        self.normalised_template[0] = fn.make_normalised_array(self.normalised_array, self.target_dose_array, self.max_dose_array, self.dose_array, self.target_index, self.max_index)
        # self.normalised_template = fn.make_normalised_array(self.normalised_array, self.target_dose_array, self.max_dose_array, self.dose_array, self.target_index, self.max_index)
        return self.normalised_template

    # def update_action_space(self):
    #     """
    #     Updates the action space as doses are added or removed, so each dose can be accessed to perform an action on.
    #     """
    #     self.number_of_beams = len(self.dose_maps)
    #     self.action_space = spaces.MultiDiscrete([2, 100, np.floor(self.range_x/5), np.floor(self.range_y/5), 6]) # Action, which dose, x/y values of spot, beam angle

    def add_beam(self, spot_x, spot_y, angle):
        """
        Adds a beam depending on the entry point and angle. Calculates WEPL through beam voxels and therefore dose 
        deposited in each voxel. Adds this dose to the dose array.
        """

        final_spread, beam_bool = ProtonBeamV4.ProtonBeamV4(spot_x, spot_y, angle, self.grid_x, self.grid_y, self.voxel_dx, self.voxel_dy, self.density_array, self.energy_data)

        if beam_bool:

            self.dose_array += final_spread
            # self.dose_maps.append(final_spread)

        #self.beam_coords.append([spot_x, spot_y])
        self.number_of_beams += 1

        return beam_bool
    
    def add_sobp(self,spot_x, spot_y, angle):

        position_mulitplier_linear = 7

        beam_angle = angle

        if (angle >= 0) & (angle <= 90):
        
            spot_x_2 = spot_x + position_mulitplier_linear * np.sin(angle * np.pi / 180)
            spot_x_3 = spot_x + 2 * position_mulitplier_linear * np.sin(angle * np.pi / 180)

            spot_y_2 = spot_y + position_mulitplier_linear * np.cos(angle * np.pi / 180)
            spot_y_3 = spot_y + 2 * position_mulitplier_linear * np.cos(angle * np.pi / 180)


        elif (angle > 90) & (angle <= 180):

            angle -= 90
        
            spot_x_2 = spot_x + position_mulitplier_linear * np.sin(angle * np.pi / 180)
            spot_x_3 = spot_x + 2 * position_mulitplier_linear * np.sin(angle * np.pi / 180)

            spot_y_2 = spot_y - position_mulitplier_linear * np.cos(angle * np.pi / 180)
            spot_y_3 = spot_y - 2 * position_mulitplier_linear * np.cos(angle * np.pi / 180)

        elif (angle > 180) & (angle <= 270):

            angle -= 180
        
            spot_x_2 = spot_x - position_mulitplier_linear * np.sin(angle * np.pi / 180)
            spot_x_3 = spot_x - 2 * position_mulitplier_linear * np.sin(angle * np.pi / 180)

            spot_y_2 = spot_y - position_mulitplier_linear * np.cos(angle * np.pi / 180)
            spot_y_3 = spot_y - 2 * position_mulitplier_linear * np.cos(angle * np.pi / 180)

        elif (angle > 270) & (angle <= 360):

            angle -= 270
        
            spot_x_2 = spot_x - position_mulitplier_linear * np.sin(angle * np.pi / 180)
            spot_x_3 = spot_x - 2 * position_mulitplier_linear * np.sin(angle * np.pi / 180)

            spot_y_2 = spot_y + position_mulitplier_linear * np.cos(angle * np.pi / 180)
            spot_y_3 = spot_y + 2 * position_mulitplier_linear * np.cos(angle * np.pi / 180)

        if (spot_x_2 > self.range_x + self.x_offset) or (spot_x_3 > self.range_x + self.x_offset) or (spot_x_2 < self.x_offset) or (spot_x_3 < self.x_offset):
            return np.zeros(self.target_dose_array.shape), False
        
        elif (spot_y_2 > self.range_y + self.y_offset) or (spot_y_3 > self.range_y + self.y_offset) or (spot_y_2 < self.y_offset) or (spot_y_3 < self.y_offset):
            return np.zeros(self.target_dose_array.shape), False

        

        final_spread_main, beam_bool_main = ProtonBeamV4.ProtonBeamV4(spot_x, spot_y, beam_angle, self.grid_x, self.grid_y, self.voxel_dx, self.voxel_dy, self.density_array, self.energy_data)
        final_spread_2, beam_bool_2 = ProtonBeamV4.ProtonBeamV4(spot_x_2, spot_y_2, beam_angle, self.grid_x, self.grid_y, self.voxel_dx, self.voxel_dy, self.density_array, self.energy_data)
        final_spread_3, beam_bool_3 = ProtonBeamV4.ProtonBeamV4(spot_x_3, spot_y_3, beam_angle, self.grid_x, self.grid_y, self.voxel_dx, self.voxel_dy, self.density_array, self.energy_data)

        final_spread = 0.6*final_spread_2 + 0.3*final_spread_3 + final_spread_main 

        master_bool = True

        if not beam_bool_2 or not beam_bool_3 or not beam_bool_main: 
            master_bool = False
        else:
            self.dose_array += final_spread

        return master_bool



    def remove_beam(self, index):
        """
        Removes a beam from the beams array, and updates action space and dose array.
        """
        # Can't remove if only one beam
        if len(self.dose_maps) == 1:
            return False
        
        elif index <= len(self.dose_maps):
            dose_map = self.dose_maps[index] # Which beam to remove
            self.dose_array -= dose_map # Remove dose contribution of that beam

            self.dose_maps.pop(index)

            self.beam_coords.pop(index)

            # self.update_action_space()

            return True
        
        else:
            return False

    def results(self):
        """
        Returns the pecentage of the tumour that is dosed and non-tumour that is overdosed.
        """
        dose_indices = self.skin_fill

        dose = self.dose_array[dose_indices]
        target = self.target_dose_array[dose_indices]
        max = self.max_dose_array[dose_indices]
    
        tumour_indices = np.where(target > 2)
        non_tumour_indices = np.where(target < 2)

        number_tumour_pixels = len(tumour_indices[0])
        number_non_tumour_pixels = len(non_tumour_indices[0])

        number_non_tumour_over_pixels = len(np.where(dose[non_tumour_indices[0]] > max[non_tumour_indices[0]])[0])
        percent_non_tumour_over_dosed = number_non_tumour_over_pixels / number_non_tumour_pixels

        number_tumour_dosed_pixels = len(np.where((dose[tumour_indices[0]] > target[tumour_indices[0]]))[0])
        percent_tumour_dosed = number_tumour_dosed_pixels / number_tumour_pixels

        number_tumour_dosed_under_max_pixels = len(np.where((dose[tumour_indices[0]] > target[tumour_indices[0]]) & (dose[tumour_indices[0]] < max[tumour_indices[0]]))[0])
        percent_tumour_dosed_under_max = number_tumour_dosed_under_max_pixels / number_tumour_pixels

        tumour_doses = dose[tumour_indices[0]]
        average_tumour_dose = np.mean(tumour_doses)
        std_dev_tumour_doses = np.std(tumour_doses) 

        return [[percent_tumour_dosed], [percent_tumour_dosed_under_max], [percent_non_tumour_over_dosed], [average_tumour_dose], [std_dev_tumour_doses]]
    
    def spots_pie_graph(self,figname):
        """
        Calculates the percentages of spots placed within tumours, within healthy tissue, and outside the patient alltogether. Plots these percentages in a pie chart.
        """
        number_spots = self.number_of_beams

        if number_spots != len(self.beam_coords):
            print("Number of spots: ", number_spots)
            print("Number of coordinates: ", len(self.beam_coords))
            print("Lengths no matchy :(")

        else:
            indices_tumour = np.where(self.target_dose_array > 0)
            indices_non_tumour = np.where((self.max_dose_array > 0) & (self.target_dose_array==0))
            indices_outside = np.where(self.max_dose_array==0)

            beam_coords = [tuple(map(int, coord)) for coord in self.beam_coords]

            # coords_tumour = list(zip(indices_tumour[0], indices_tumour[1]))
            # coords_non_tumour = list(zip(indices_non_tumour[0], indices_non_tumour[1]))
            # coords_outside = list(zip(indices_outside[0], indices_outside[1]))


            coords_tumour = list(zip(indices_tumour[1], indices_tumour[0]))
            coords_non_tumour = list(zip(indices_non_tumour[1], indices_non_tumour[0]))
            coords_outside = list(zip(indices_outside[1], indices_outside[0]))

            number_spots_tumour = sum(coord in beam_coords for coord in coords_tumour)
            number_spots_non_tumour = sum(coord in beam_coords for coord in coords_non_tumour)
            number_spots_outside = sum(coord in beam_coords for coord in coords_outside)

            # number_spots_tumour = len(np.where(indices_tumour==self.beam_coords))
            # number_spots_non_tumour = len(np.where(indices_non_tumour==self.beam_coords))
            # number_spots_outside = len(np.where(indices_outside==self.beam_coords))

            percent_spots_tumour = number_spots_tumour / number_spots
            percent_spots_non_tumour = number_spots_non_tumour / number_spots
            percent_spots_outside = number_spots_outside / number_spots

            labels = 'Tumour', 'Healthy Tissue', 'Outside Patient'
            pies = [number_spots_tumour, number_spots_non_tumour, number_spots_outside]

            plt.pie(pies, labels=labels, autopct='%1.1f%%')

            plt.savefig(figname,dpi=400)
            plt.close()

    def render(self, circles, show_fig, save_fig, fig_name, RESULTS = True, DOSE=True):
        """
        Displays the environment. The tumour is shown as a contour and the doses are shown as a colour map so the dose 
        the sqaures are adding is visible.
        """
        
        doseArray = np.ma.masked_where(self.dose_array < 0, self.dose_array)
        
        i = 0.1

        for contour in self.contours:
            plt.contour(contour[0], levels=[0.5], colors=contour[1], zorder=4, linewidths=2, linestyles=contour[2])
            plt.plot(0, 0, color=contour[1], linestyle=contour[2], label=contour[3])
            plt.imshow(contour[0], zorder=3, alpha=0)
            i += 0.1

        legend_pos = (1.35,1.15)

        if self.target_dose_array.shape == (300,200):
            legend_pos = (-0.1,0.7)

        if DOSE:
            doseArray[doseArray<0.1] = np.nan
            im = plt.imshow(doseArray, cmap='jet', zorder=6, alpha=0.4)
            cbar = plt.colorbar(im)
            cbar.set_label(label="Dose (Gy)", size=15)
            legend_pos = (1.67,1.15)

        if DOSE:
            plt.imshow(self.density_array, cmap='bone', zorder=1)
        else:
            im = plt.imshow(self.density_array, cmap='bone', zorder=1)
            cbar = plt.colorbar(im)
            cbar.set_label(label="Stopping Power (Water Equivalent)", size=16)        

        if RESULTS:

            results = np.array(self.results()).flatten()
            percent_tumour_dosed = results[0]
            percent_tumour_dosed_under_max = results[1]
            percent_non_tumour_over_dosed = results[2]
            average_tumour_dose = results[3]
            std_dev_tumour_doses = results[4]
            
            percent_tumour_dosed *= 100
            percent_tumour_dosed_under_max *= 100
            percent_non_tumour_over_dosed *= 100

            plt.annotate(f'Tumour dosed: ' + '{:.3g}'.format(percent_tumour_dosed) + '%, \n\nNon-tumour \noverdosed:  '+'{:.3g}'.format(percent_non_tumour_over_dosed)+'%, \n\nTumour dose standard \ndeviation: ' + ' {:.3g}'.format(std_dev_tumour_doses)+'Gy, \n\nReward: ' + ' {:.3g}'.format(self.reward_sum),
                xy=(-0.82, 0.4), xytext=(0.0, 0),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=9, ha='left', va='bottom')
    
        if circles:
            beam_coords = np.array(self.beam_coords)
            plt.scatter(beam_coords[:,0], beam_coords[:,1], marker='o', color='hotpink', label='Spot', zorder=-3)
            for beam_coord in self.beam_coords:
                circle = plt.Circle((beam_coord[0], beam_coord[1]), radius=5, fill=True, color='hotpink', zorder=7, alpha=0.7)
                plt.gca().add_patch(circle)

        plt.legend(bbox_to_anchor=legend_pos, ncol=4, fontsize=14)
        if save_fig:
            # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.tight_layout()
            plt.savefig(fig_name, dpi=600)
            if DOSE:
                cbar.remove()  # Remove the colorbar after saving the figure
            plt.clf()
        if show_fig:
            plt.show()
        plt.close()

    def beam_number(self):
        print("Number of beams:", len(self.dose_maps))

    def debug(self, TARGET = True, MAX = True, NORMALISED = True, DOSE = True, SAVE=False):
    # Define your custom colors
        color1 = 'blue'
        color2 = 'orange'
        color3 = 'green'

        # # Create the colormap dictionary
        # colors = [(-2, color1), (0, color2), (1, color3)]
        # cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        # Define the number of segments for each color
        num_segments_color1 = -2
        num_segments_color2 = 0
        num_segments_color3 =1

        # Create the colormap dictionary
        colors = [(0, color1), (num_segments_color1 / (num_segments_color1 + num_segments_color2), color2), 
                (1, color3)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        if TARGET:
            im = plt.imshow(self.target_dose_array, cmap='gnuplot')
            cbar = plt.colorbar(im)
            cbar.set_label(label="Dose (Gy)", size=15)
            plt.title('Target Array')
            if SAVE:
                plt.savefig('target_array.png',dpi=500)
            plt.show()
            


        if MAX:
            im = plt.imshow(self.max_dose_array, cmap='gnuplot')
            cbar = plt.colorbar(im)
            cbar.set_label(label="Maximum Dose (Gy)", size=15)
            plt.title('Max Dose Array')
            if SAVE:
                plt.savefig('max_array.png',dpi=500)
            plt.show()

        if NORMALISED:
            im = plt.imshow(self.normalised_array, cmap='gnuplot')
            cbar = plt.colorbar(im)
            cbar.set_label(label="Observation (Unitless)", size=15)
            plt.title('Normalised Array')
            if SAVE:
                plt.savefig('normal_array.png',dpi=500)
            plt.show()

        if DOSE:
            im = plt.imshow(self.dose_array, cmap='jet')
            cbar = plt.colorbar(im)
            cbar.set_label(label="Dose (Gy)", size=15)
            plt.title('Dose Array')
            if SAVE:
                plt.savefig('dose_array.png',dpi=500)
            plt.show()


    def reset(self, seed=None):
        # Set the seed for reproducibility (optional)
        if seed is not None:
            np.random.seed(seed)  # Assuming numpy is used for random number generation
        
        """
        Resets the environment. Removes all doses and adds one at centre, removes all dose values.
        """
        #print(self.number_of_beams)

        # plt.imshow(self.density_array,cmap='bone')
        # plt.imshow(self.dose_array, cmap='jet', alpha=0.5)
        # plt.show()
        
        self.dose_maps = []
        self.beam_coords = []
        self.number_of_beams = len(self.dose_maps)
        self.dose_array = np.zeros(self.array_shape, dtype=np.float32)
        self.add_beam(self.range_x / 2 + self.x_offset, self.range_y / 2 + self.y_offset, 3)
        self.step_count = 0
        self.done = False
        self.reward_sum = 0




        #self.action_space = spaces.MultiDiscrete([np.floor(self.range_x/self.coordinate_action_modifier), np.floor(self.range_y/self.coordinate_action_modifier), self.angle_action_modifier]) # Action, which dose, x/y values of spot, beam angle
        # min_x = 0.4  # Minimum x coordinate
        # max_x = np.floor(self.range_x/self.coordinate_action_modifier) # Maximum x coordinate
        # min_y = 0.4  # Minimum y coordinate
        # max_y = np.floor(self.range_y/self.coordinate_action_modifier) # Maximum y coordinate
        # min_angle = 0.4  # Minimum angle
        # max_angle = 5  # Maximum angle

        # # Define the bounds for the action space
        # low = np.array([min_x, min_y, min_angle])
        # high = np.array([0.6,0.6,0.6])


        # # Define the continuous action space
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([np.floor(self.range_x/self.coordinate_action_modifier), np.floor(self.range_y/self.coordinate_action_modifier), self.angle_action_modifier])
        
        
        
        
        
        observation = self.update_observation_space()
        info = {}
        return observation, info