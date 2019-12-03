# ==========================================
# import modules
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

import misc
import random_walker_4D as rw4D
import random_walker_3D as rw3D

import tkinter as tk

import argparse

# Parse data input and output directories
def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run random walker flow segmenter.')
    parser.add_argument('--mri-path', type=str, default='../data/MRT Daten Bern Numpy/14/10002CA3/10002CA4/' ,#default='../../data/freiburg/',
                    help='Directory containing MRI data set')
    parser.add_argument('--subject-name', type=str, default='0', #default='AH',
                    help='Name of the subject to segment')
    return parser.parse_args()

args = parse_args()

# ==========================================
# 4D Random Walker Segmentation Tool for 4D MRI Flow Images
# ==========================================
class RandomWalkerFlowSegmenter():

    # =============================
    # set subject name to be loaded
    # For the ETH dataset: 'AH', 'CB', 'DG', 'JR', 'LT' (with quotes)
    # For the Freiburg dataset 0, 1, .... 141 (without quotes)
    # =============================
    subject_name = args.subject_name  #'AH'

    # =============================
    # list of subjects in the eth dataset
    # =============================
    subjects_eth_dataset = ['AH', 'CB', 'DG', 'JR', 'LT']

    # =============================
    # parec data contains the arrays with m,p and s (velocity components) as well as ints with the number of timesteps and number of slices
    # load parec datsa using custom function in pl (see imports)
    # =============================
    if subject_name in subjects_eth_dataset:
        
        parec_data = misc.load_parec_data(subject_name = subject_name)
        
        # =============================
        # returns the arrays separated by velocity and magnitude components (4 arrays, magnitude first (index 0))
        # transpose the arrays by permuting the "channel" dimension to the back so we have: x,y,z,t,channel
        # =============================
        separated_arrays = np.transpose(np.array(misc.create_separated_arrays(parec_data)),(1,2,3,4,0))
        
        # =============================
        # normalize the arrays so that velocity and magnitude are in the range [0,1]
        # also this function uses the 95th percentile to normalize the data and clips any datapoint that is larger to 1.0 to get rid of outliers
        # =============================
        separated_arrays = misc.normalize_arrays(separated_arrays)        
        
        print('===================================================')
        print('Shape of the loaded image is: ' + str(separated_arrays.shape))
        print('===================================================')
        
        # =============================
        # extract the array that contains the magnitude images
        # =============================
        magnitude_array = separated_arrays[...,0]
        img_array = separated_arrays[...,0]        
        
        # =============================
        # get array dimensions
        # =============================
        x_size = magnitude_array.shape[1]
        y_size = magnitude_array.shape[0]
        z_size = magnitude_array.shape[2]
        t_size = magnitude_array.shape[3]

        # =============================        
        # in the original data, the images alternate between magnitude and velocity images, so the time steps and the actual volume depth are 0.5x this size
        # =============================
        img_timestep = round(t_size/2)
        img_slice =round(z_size/2)
        
    else:
        subject_name = int(subject_name)

        separated_arrays = misc.load_npy_data(args.mri_path, subject_name) # freiburg_mri_path='../../data/freiburg/'
        separated_arrays = misc.normalize_arrays(separated_arrays)
        
        # =============================        
        # extract the array that contains the magnitude images
        # =============================        
        magnitude_array = separated_arrays[...,0]
        img_array = separated_arrays[...,0]
        
        # =============================        
        # get array dimensions
        # =============================        
        x_size = magnitude_array.shape[1]
        y_size = magnitude_array.shape[0]
        z_size = magnitude_array.shape[2]
        t_size = magnitude_array.shape[3]
        
        img_timestep = t_size-1
        img_slice = z_size-1
             
    # =============================                
    # create placeholders / initialize variables for the different arrays and lists that the GUI uses
    # =============================        
    rw_data = np.zeros(magnitude_array.shape)
    rw_data_cpy = np.copy(rw_data)
    rw_labels = np.zeros((2, magnitude_array.shape[0], magnitude_array.shape[1], magnitude_array.shape[2], magnitude_array.shape[3]))
    rw_labels3D = np.zeros((2, magnitude_array.shape[0], magnitude_array.shape[1], magnitude_array.shape[2]))

    fg_coord_list = []
    bg_coord_list = []
    
    markers_3d_t = np.empty(magnitude_array.shape)
    markers_3d_t_cpy = np.copy(markers_3d_t)
    
    markers_visible = False
        
    # =============================        
    # Define the needed callbacks and functions for the GUI to run
    # =============================          
    
    # =============================      
    # function that updates the displayed images based on states of buttons (buttonXY['text'] and the arrays (img_array, fg_markers, bg_markers, gt_data, rw_data))
    # function modifies arrays, creates a plot, saves this as an img and then loads the image into the Canvas of the GUI
    # =============================      
    def update_image(self):
        
        if self.button_toggle_canvas1display['text'] == "Vel. Magnitude":
            
            if self.button_overlap['text'] == "No Overlap":
                
                # ============
                # create a figure
                # ============
                plt.figure()
                
                # ============
                # display the image
                # ============
                if self.subject_name in self.subjects_eth_dataset:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep].T, cmap='gray', alpha = 0.8)
                else:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep], cmap='gray', alpha = 0.8)
                
                # ============
                # display the segmentation
                # ============
                plt.imshow(self.rw_data_cpy[:, :, self.img_slice, self.img_timestep], cmap = 'Blues_r', interpolation = 'none', alpha = 0.4)
                
                # ============
                # display markers
                # ============
                if self.markers_visible == True:
                    plt.imshow(self.markers_3d_t_cpy[:, :, self.img_slice, self.img_timestep], cmap = 'Greens_r', interpolation = 'none', alpha = 0.7)
                    
                # ============
                # misc settings for the figure
                # ============
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)

                # ============
                # save and close the figure
                # ============
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()
                
            else:
                
                # ============
                # create a figure
                # ============
                plt.figure()
                
                # ============
                # display the image
                # ============
                if self.subject_name in self.subjects_eth_dataset:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep].T, cmap='gray')
                else:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep], cmap='gray')
                    
                # ============
                # display markers
                # ============
                if self.markers_visible:
                    plt.imshow(self.markers_3d_t_cpy [:, :, self.img_slice, self.img_timestep], cmap = 'Greens_r', interpolation = 'none', alpha = 0.7)
                    
                # ============
                # misc settings for the figure
                # ============
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)
                
                # ============
                # save and close the figure
                # ============
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()

        elif self.button_toggle_canvas1display['text'] == "Normal":
            
            if self.button_overlap['text'] == "No Overlap":
                
                # ============
                # create a figure
                # ============
                plt.figure()
                
                # ============
                # display the image
                # ============
                if self.subject_name in self.subjects_eth_dataset:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep].T, cmap = 'gray', alpha = 0.8)
                else:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep], cmap = 'gray', alpha = 0.8)
                    
                # ============
                # display the segmentation
                # ============
                plt.imshow(self.rw_data_cpy[:, :, self.img_slice, self.img_timestep], cmap = 'Blues_r', interpolation = 'none', alpha = 0.4)
                
                # ============
                # display markers
                # ============
                if self.markers_visible == True:
                    plt.imshow(self.markers_3d_t_cpy[:, :, self.img_slice, self.img_timestep], cmap = 'Greens_r', interpolation = 'none', alpha = 0.7)
                
                # ============
                # misc settings for the figure
                # ============
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)
                
                # ============
                # save and close the figure
                # ============
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()
                
            else:
                
                # ============
                # create a figure
                # ============
                plt.figure()
                
                # ============
                # display the image
                # ============
                if self.subject_name in self.subjects_eth_dataset:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep].T, cmap='gray')
                else:
                    plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep], cmap='gray')
                    
                # ============
                # display markers
                # ============
                if self.markers_visible:
                    plt.imshow(self.markers_3d_t_cpy[:, :, self.img_slice, self.img_timestep], cmap = 'Greens_r', interpolation = 'none', alpha = 0.7)
                    
                # ============
                # misc settings for the figure
                # ============
                plt.xticks([], []); plt.yticks([], [])
                plt.gca().set_axis_off()
                plt.margins(0,0)
                
                # ============
                # save and close the figure
                # ============
                plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()
        
        else:
            
            # ============
            # create a figure
            # ============
            plt.figure()
            
            # ============
            # display the image
            # ============
            plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep], cmap = 'gray')
            
            # ============
            # display markers
            # ============
            plt.imshow(self.markers_3d_t[:, :, self.img_slice, self.img_timestep], cmap = 'Greens_r', interpolation = 'none', alpha = 0.7)
            
            # ============
            # misc settings for the figure
            # ============
            plt.xticks([], []); plt.yticks([], [])
            plt.gca().set_axis_off()
            plt.margins(0,0)
            
            # ============
            # save and close the figure
            # ============
            plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        
        # ============
        # create a figure
        # ============
        plt.figure()
        
        # ============
        # display the segmentation
        # ============
        plt.imshow(self.rw_data[:, :, self.img_slice, self.img_timestep], cmap = 'gray')
        
        # ============
        # misc settings for the figure
        # ============
        plt.xticks([], []); plt.yticks([], [])
        plt.gca().set_axis_off()
        plt.margins(0,0)
        
        # ============
        # save and close the figure
        # ============
        plt.savefig('Tksegimg.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
        # ============
        # load the image and display on canvas1
        # ============
        pngimage = Image.open('.//Tkimg.png').resize(size = (self.x_size*3, self.y_size*3), resample = Image.BICUBIC)
        self.img = ImageTk.PhotoImage(image=pngimage)
        self.canvas1.create_image(0, 0, anchor = tk.NW, image = self.img)

        # ============
        # load the segmentation and display on canvas2
        # ============                
        pngsegimage = Image.open('.//Tksegimg.png').resize(size=(self.x_size*3, self.y_size*3), resample = Image.BICUBIC)
        self.segimg = ImageTk.PhotoImage(image=pngsegimage)
        self.canvas2.create_image(0, 0, anchor = tk.NW, image = self.segimg)

        return      

    # =============================      
    # use slider1 to set the desired slice
    # =============================      
    def update_z_axis(self):
        
        self.img_slice = self.slider_z_axis.get()
        self.update_image()
        
        return
        
    # =============================      
    # use slider2 to set the desired timestep   
    # =============================      
    def update_t_axis(self):
        
        self.img_timestep = self.slider_t_axis.get()
        self.update_image()
        
        return
    
    # =============================      
    # loop to manage what to display (magnitude, overlap, markers, etc.)
    # =============================      
    def display_mode1(self):
          
        if self.button_toggle_canvas1display['text'] == "Vel. Magnitude":
            self.button_toggle_canvas1display.configure(text = "Normal")
            self.img_array = misc.norm(self.separated_arrays[...,1], self.separated_arrays[...,2], self.separated_arrays[...,3])
        
        elif self.button_toggle_canvas1display['text'] == "Normal":
            self.button_toggle_canvas1display.configure(text = "Vel. Magnitude")
            self.img_array = self.separated_arrays[...,0]
                    
        self.update_image()
            
        return
    
    # =============================          
    # =============================      
    def display_mode2(self):
       
        if self.button_overlap['text'] == "Toggle Overlap":
            self.button_overlap.configure(text = "No Overlap")
        
        elif self.button_overlap['text'] == "No Overlap":
            self.button_overlap.configure(text = "Overlap")
        
        elif self.button_overlap['text'] == "Overlap":
            self.button_overlap.configure(text = "No Overlap")
        
        self.update_image()
        
        return
        
    # =============================      
    # =============================      
    def mousecallback(self, event):
        
        x, y = event.x, event.y
        coord_tuple = (x,y)
        
        if x > 0 and y > 0 and x < self.x_size*3-1 and y < self.y_size*3-1:
        
            if self.v.get() == 1:
            
                if coord_tuple not in self.fg_coord_list:
                    self.canvas1.create_oval(x, y, x+3, y+3, fill='red')
                    self.fg_coord_list.append(coord_tuple)
            
            elif self.v.get() == 2:
            
                if coord_tuple not in self.bg_coord_list:
                    self.canvas1.create_oval(x, y, x+3, y+3, fill='blue')
                    self.bg_coord_list.append(coord_tuple)
            else:
                return
    
    # =============================      
    # =============================          
    def scribble_draw(self):
        
        self.canvas1.bind("<B1-Motion>", self.mousecallback)
        
        # =============
        # disable scrolling through slices until the current scribble has been saved.
        # =============
        self.slider_z_axis.config(state = tk.DISABLED)
        self.slider_t_axis.config(state = tk.DISABLED)
        
        # =============
        # activate the add_scribble button
        # =============
        self.button_add_scribble.config(state = tk.NORMAL)
        
        return

    # =============================      
    # =============================      
    def toggle_segmentation(self):
        
        self.rw_data_cpy = np.round(self.rw_labels[0,...])
        self.rw_data_cpy = np.ma.masked_where(self.rw_data_cpy < 1, self.rw_data_cpy)
        
        if self.button_toggle_canvas2display['text'] == "Probability map":
            self.button_toggle_canvas2display.configure(text = "Segmentation")
            self.rw_data = self.rw_labels[0,...]
            self.update_image()

        else:
            self.button_toggle_canvas2display.configure(text = "Probability map")
            self.rw_data = np.round(self.rw_labels[0,...])
            self.update_image()
            
        return
    
    # =============================      
    # =============================      
    def add_scribble(self):
    
        # ============
        # ============
        self.v.set(0)
        self.button_scribbleFG.deselect()
        self.button_scribbleBG.deselect()
        
        # ============
        # activate the sliders to move through the slices again
        # ============
        self.slider_z_axis.config(state = tk.NORMAL)
        self.slider_t_axis.config(state = tk.NORMAL)
        
        # ============
        # activate the random walker buttons after at least one set of scribbles has been added
        # ============
        self.button_run3d.config(state = tk.NORMAL)
        self.button_run4dusingscribbles.config(state = tk.NORMAL)
        
        # ============
        # ============
        self.canvas1.delete("all")
        
        # ============
        # ============
        self.init_markers()
        
        # ============
        # ============
        if len(np.unique(self.markers_3d_t)) > 1:
            self.markers_visible = True
            self.markers_3d_t_cpy  = np.ma.masked_where(self.markers_3d_t < 1, self.markers_3d_t)    
        
        # ============
        # ============
        self.update_image()  
        
        return

    # =============================      
    # =============================      
    def init_markers(self):
        
        if self.fg_coord_list or self.bg_coord_list:
            
            self.markers_2d = np.zeros(self.magnitude_array[:,:,0,0].shape)
            
            for t in self.fg_coord_list:
                self.markers_2d[round(t[1]/3)-1, round(t[0]/3)-1] = 1
                
            for t in self.bg_coord_list:
                self.markers_2d[round(t[1]/3)-1, round(t[0]/3)-1] = 2
                    
            self.markers_3d_t[:, :, self.img_slice, self.img_timestep] = self.markers_2d
            self.fg_coord_list = []
            self.bg_coord_list = []
            
        return
        
    # =============================      
    # =============================      
    def run_random_walker3D(self):
        
        print('===================================================')
        print('running the random walker algorithm in 3d...')
        
        self.init_markers()
        
        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a, b, c = 200, 6, 500
    
        self.rw_labels3D = rw3D.random_walker(data = self.separated_arrays[..., self.img_timestep, :],
                                              labels = self.markers_3d_t[..., self.img_timestep],
                                              mode = 'cg_mg',
                                              return_full_prob = True,
                                              alpha = alpha_a,
                                              beta = beta_b,
                                              gamma = gamma_g,
                                              a = a,
                                              b = b,
                                              c = c)      
        
        self.eroded_markers3D, self.fg_markers, self.bg_markers = misc.erode_seg_markers(np.round(self.rw_labels3D[0,:,:,:]))
        
        self.rw_labels[...,self.img_timestep] = self.rw_labels3D
        
        self.add_scribble()
        
        self.toggle_segmentation()
        
        # ===========
        # activate the run4dusing3dresults after the 3d random walker has run at least once.
        # ===========
        self.button_run4dusing3dresults.config(state = tk.NORMAL)
        
        print('done!')
        print('===================================================')
        
        return
    
    # =============================      
    # =============================      
    def propogate_3d_markers_to_4d(self):
        
        for t in range(self.t_size):
            self.markers_3d_t[...,t] = self.eroded_markers3D
            
        # ===========
        # this will ensure that the markers are not over-written by zeros when the init_markers() function is called in run_random_walker4D().
        # ===========
        self.fg_coord_list = []
        self.bg_coord_list = []
        
        return
    
    # =============================      
    # =============================      
    def run_random_walker4D(self):

        print('===================================================')        
        print('running the random walker algorithm in 4d...')
        
        self.init_markers()
        
        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a, b, c = 200, 6, 500
    
        self.rw_labels = rw4D.random_walker(data = self.separated_arrays,
                                            labels = self.markers_3d_t,
                                            mode = 'cg_mg',
                                            return_full_prob = True,
                                            alpha = alpha_a,
                                            beta = beta_b,
                                            gamma = gamma_g,
                                            a = a,
                                            b = b,
                                            c = c)
        
        self.toggle_segmentation()
        
        print('done!')
        print('===================================================')        
               
        return

    # =============================      
    # =============================          
    def run_random_walker4D_usingScribbles(self):        
        
        print('===================================================')        
        print('About to run the random walker algorithm in 4d using only user scribbles. This can take very long!')
        self.run_random_walker4D()
        
        return
    
    # =============================      
    # =============================          
    def run_random_walker4D_using3Dresults(self):        
        
        self.propogate_3d_markers_to_4d()
        self.run_random_walker4D()
        
        return
    
    # =============================      
    # =============================      
    def toggle_scribbles(self):
        
        if self.markers_visible == True:
               self.markers_visible = False
        else:
            self.markers_visible = True
            
        self.update_image()
            
        return

    # =============================      
    # =============================      
    def save_seg(self):
        
        save_path = os.path.join(os.getcwd(),'output')
        seg_name = 'rw_seg_{}.npy'.format(self.subject_name)
        np.save(os.path.join(save_path, seg_name),
                np.round(self.rw_labels[0,...]))
        
        return
    
    # =============================      
    # Here all the elements of the GUI are defined.
    # E.g. Buttons, Sliders, Canvas, Labels, along with their sizes, values and callbacks (command)
    # =============================      
    # Some documentation related to tkinter
    # ================
    # Grid
        # The Grid geometry manager puts the widgets in a 2-dimensional table.
        # The master widget is split into a number of rows and columns, and each “cell” in the resulting table can hold a widget.    
            # Sticky
                # Note that the widgets are centered in their cells.
                # You can use the sticky option to change this; this option takes one or more values from the set N, S, E, W.
                # To align the labels to the left border, you could use W (west)
    # ================
    # Vatiables
        # https://effbot.org/tkinterbook/variable.htm
    # =============================          
    def __init__(self, main):
    
        # ==========
        # canvas for displaying the input image - either the intensity or the velocity magnitude.
        # scribbles will be made on this canvas.
        # ==========
        self.canvas1 = tk.Canvas(main, width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas1.grid(row=1, column=2, rowspan=7, sticky=tk.W)
        
        # ==========
        # canvas for displaying the output segmentation or its probability
        # ==========
        self.canvas2 = tk.Canvas(main, width=self.x_size*3, height=self.y_size*3, background='white')
        self.canvas2.grid(row=1, column=3, rowspan=7, sticky=tk.W)
                
        # ==========
        # slider for z axis
        # ==========
        self.slider_z_axis = tk.Scale(main, from_=0, to=self.z_size-1, length=200, tickinterval=5, orient=tk.HORIZONTAL, label="Z-Axis", command=lambda x: self.update_z_axis())
        self.slider_z_axis.set(round(self.z_size/2))
        self.slider_z_axis.grid(row=0, column=1, padx=5, pady=5)
        
        # ==========
        # slider for t axis
        # ==========
        self.slider_t_axis = tk.Scale(main, from_=0, to=self.t_size-1, length=200, tickinterval=5, orient=tk.HORIZONTAL, label="T-Axis", command=lambda x: self.update_t_axis())
        self.slider_t_axis.set(round(self.t_size/2))
        self.slider_t_axis.grid(row=1, column=1, padx=5, pady=5)

        # ==========
        # button for running the random walker in 3d using the scibbles made by the user
        # ==========
        self.button_run3d = tk.Button(main, text='Run 3D', width=20, command=lambda: self.run_random_walker3D())
        self.button_run3d.grid(row=2, column=1, padx=5, pady=5)
        self.button_run3d.config(state = tk.DISABLED)
        
        # ==========
        # button for running the random walker in 4d using the scibbles made by the user
        # ==========
        self.button_run4dusingscribbles = tk.Button(main, text='Run 4D (using scribbles only)', width=20, command=lambda: self.run_random_walker4D_usingScribbles())
        self.button_run4dusingscribbles.grid(row=3, column=1, padx=5, pady=5)
        self.button_run4dusingscribbles.config(state = tk.DISABLED)
        
        # ==========
        # button for running the random walker in 4d using markers which are propogated from the 3d segmentation
        # ==========
        self.button_run4dusing3dresults = tk.Button(main, text='Run 4D (using 3D results)', width=20, command=lambda: self.run_random_walker4D_using3Dresults())
        self.button_run4dusing3dresults.grid(row=4, column=1, padx=5, pady=5)
        self.button_run4dusing3dresults.config(state = tk.DISABLED)
               
        # ==========
        # ==========
        self.v = tk.IntVar()
        self.button_scribbleFG = tk.Radiobutton(main, text="Scribble FG", variable=self.v, value=1, indicatoron=0, width=20, command=lambda: self.scribble_draw())
        self.button_scribbleBG = tk.Radiobutton(main, text="Scribble BG", variable=self.v, value=2, indicatoron=0, width=20, command=lambda: self.scribble_draw())
        self.button_scribbleFG.grid(row=5, column=1, padx=5, pady=5)
        self.button_scribbleBG.grid(row=6, column=1, padx=5, pady=5)
                
        # ==========
        # ==========
        self.button_add_scribble = tk.Button(main, text='Add Scribble', width=20, command=lambda: self.add_scribble())
        self.button_add_scribble.grid(row=7, column=1, padx=5, pady=5)
        self.button_add_scribble.config(state = tk.DISABLED)
        
        # ==========
        # ==========
        self.button_toggle_canvas1display = tk.Button(main, text="Vel. Magnitude", width=20, command=lambda: self.display_mode1())
        self.button_toggle_canvas1display.grid(row=8, column=2, padx=5, pady=5)
        
        # ==========
        # ==========
        self.button_toggle_canvas2display = tk.Button(main, text="Probability map", width=20, command=lambda: self.toggle_segmentation())
        self.button_toggle_canvas2display.grid(row=8, column=3, padx=5, pady=5)
        
        # ==========
        # ==========
        self.button_overlap = tk.Button(main, text="Toggle Overlap", width=20, command=lambda: self.display_mode2())
        self.button_overlap.grid(row=9, column=2, padx=5, pady=5)
        
        # ==========
        # ==========
        self.button_scribbles = tk.Button(main, text="Toggle Scribbles", width=20, command=lambda: self.toggle_scribbles())
        self.button_scribbles.grid(row=10, column=2, padx=5, pady=5)
        
        # ==========
        # ==========
        self.button_save_segmentation = tk.Button(main, text="Save segmentation", width=20, command=lambda: self.save_seg())
        self.button_save_segmentation.grid(row=9, column=1, padx=5, pady=5)
    
        # ==========
        # ==========
        self.button_quit = tk.Button(main, text='Quit', width=20, command=main.destroy)
        self.button_quit.grid(row=100, column=1, padx=5, pady=5)        
        
# =============================      
# TKinter main looop
# =============================      
gui = tk.Tk()
gui.title('Random Walker Segmentation GUI')
RandomWalkerFlowSegmenter(gui)
gui.mainloop()
