# Interactive segmentation tool for 4D flow MRIs

==========================================================
<br />Main Author: Nicolas Blondel (n.blondel@me.com)
<br />Edited by: Neerav Karani (nkarani@vision.ee.ethz.ch)
<br />Institute: Computer Vision Lab, ETH Zürich
<br />Created: 29.07.2019
<br />Last updated: 11.09.2019
==========================================================
<br />Random Walker Based 4D Flow MRI Segmentation Tool
<br />Version 0.0.1
==========================================================
<br />Tested with Python 3.6.6
<br />Required extensions: mayavi, tkinter, PIL, pyamg
=========================================================
<br />Brief file descriptions:
<br />random_walker_gui.py (defines the GUI and specifies the high level processing pipeline) (Main file to run).
<br />misc.py (defines helper functions such as data read functionsm normalization, etc.)
<br />random_walker_3D.py (extends the random walker algorithm from skimage with a similarity metric that takes into account intensities as well blood flow similarity.)
<br />random_walker_4D.py (same as random_walker_3D, but defines similarities in 4D.)
==========================================================
<br />Data paths are set as follows:
<br />The code is stored in this directory: basepath + '/code/random_walker/'
<br />The data is stored in this directory: basepath + '/data/data_source/'
==========================================================
<br />Workflow:
<br />1) Cycle through display modes ("Vel. Magnitude"-Button) to see the image with the velocity magnitude.
<br />2) Explore the 4D volume with the sliders "t-axis" and "z-axis" until you see good contrast in the velocity.
<br />3) Press "Scribble FG" to add a scribble to the foregound, draw a couple of lines inside the aorta.
<br />4) Press "Scribble BG" and repeat with the background.
<br />5) T and z-axis are frozen (on purpose).
<br />6) Press "Add scribble" to unfreeze and add more scribbles in other slices.
<br />7) Add 2-3 more scribbles at other slices ("z-axis"-slider), for the same t value.
<br />8) Press "Run 3D" to run the algorithm in 3D, for the t value for which the scribbles were made.
<br />9) Check the result with the "Overlap" button.
<br />10) Add more scribbles where desired. If you scribble on a slice where there were already some scribbles, only the latest scribbles are used. This should probably be changed in a future version.
<br />10) To propagate the 3D result to a 4D segmentation, press "Run 4D (using 3D results)". This uses the 3D segmentation as rough markers for other timesteps with dilation and erosion.
<br />11) Alternatively, you may directly run the algorithm in 4D (by pressing "Run 4D (using scribbles only)"). Note that this requires a lot of scribbles also across time and will take quite long to run.
<br />12) Press "Save segmentation" to save the 4D segmentation.
<br />13) Done.
<br />To reset and rerun the RW simply press "Quit" and restart the program.
==========================================================
