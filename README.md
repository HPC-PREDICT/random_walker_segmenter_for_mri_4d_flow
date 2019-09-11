# random_walker_segmenter_for_mri_4d_flow

==========================================================

Main Author: Nicolas Blondel (n.blondel@me.com)
Edited by: Neerav Karani (nkarani@vision.ee.ethz.ch)
Institute: Computer Vision Lab, ETH Zürich
Created: 29.07.2019
Last updated: 11.09.2019

==========================================================

Random Walker Based 4D Flow MRI Segmentation Tool
Version 0.0.1

==========================================================

Created and tested with Python 3.6.6
Required extensions: mayavi, tkinter, PIL, pyamg

==========================================================

Brief file descriptions:
random_walker_gui.py (defines the GUI and specifies the high level processing pipeline) (Main file to run).
misc.py (defines helper functions such as data read functionsm normalization, etc.)
random_walker_3D.py (extends the random walker algorithm from skimage with a similarity metric that takes into account intensities as well blood flow similarity.)
random_walker_4D.py (same as random_walker_3D, but defines similarities in 4D.)

==========================================================

Data paths are set as follows:
The code is stored in this directory: basepath + '/code/random_walker/'
The data is stored in this directory: basepath + '/data/data_source/'

==========================================================

Workflow for best segmentation result:

1) Cycle through display modes ("Vel. Magnitude"-Button) to see the image with the velocity magnitude.
2) Explore the 4D volume with the sliders "t-axis" and "z-axis" until you see good contrast in the velocity.
3) Press "Scribble FG" to add a scribble to the foregound, draw a couple of lines inside the aorta.
4) Press "Scribble BG" and repeat with the background.
5) T and z-axis are frozen (on purpose).
6) Press "Add scribble" to unfreeze and add more scribbles in other slices.
7) Add 2-3 more scribbles at other slices ("z-axis"-slider), for the same t value.
8) Press "Run 3D" to run the algorithm in 3D, for the t value for which the scribbles were made.
9) Check the result with the "Overlap" button.
10) Add more scribbles where desired. If you scribble on a slice where there were already some scribbles, only the latest scribbles are used. This should probably be changed in a future version.
10) To propagate the 3D result to a 4D segmentation, press "Run 4D (using 3D results)". This uses the 3D segmentation as rough markers for other timesteps with dilation and erosion.
11) Alternatively, you may directly run the algorithm in 4D (by pressing "Run 4D (using scribbles only)"). Note that this requires a lot of scribbles also across time and will take quite long to run.
12) Press "Save segmentation" to save the 4D segmentation.
13) Done.
To reset and rerun the RW simply press "Quit" and restart the program.
==========================================================
