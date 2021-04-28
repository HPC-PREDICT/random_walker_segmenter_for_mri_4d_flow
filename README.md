# Interactive segmentation tool for 4D flow MRIs

==========================================================
<br />Authors: Nicolas Blondel (n.blondel@me.com), Neerav Karani (nkarani@vision.ee.ethz.ch)
<br />Institute: Computer Vision Lab, ETH Zürich

==========================================================
<br />Random Walker Based 4D Flow MRI Segmentation Tool
<br />Version 0.0.2

==========================================================
<br />Tested with Python 3.6.6
<br />Required extensions: mayavi, tkinter, PIL, pyamg

=========================================================
<br />Brief file descriptions:
<br />segmenter_rw.py (defines the GUI and specifies the high level processing pipeline) (Main file to run).
<br />utils.py (defines helper functions such as data read functions, image normalization, etc.)
<br />rw3D.py (extends the random walker algorithm from skimage with a similarity metric that takes into account intensities as well blood flow similarity.)
<br />rw4D.py (same as random_walker_3D, but defines similarities in 4D.)

==========================================================
<br />Workflow:
<br />0) Set the input image and output segmentation paths in segmenter_rw.py (lines 25-28), and run this file.
<br />1) Toggle through the display modes ("Intensity" / "Velocity Magnitude" Button).
<br />2) Explore the 4D volume with the sliders "t-axis" and "z-axis" until you see good contrast in the velocity.
<br />3) Press "Scribble FG" to add a scribble to the foregound, draw a couple of lines inside the aorta.
<br />4) Press "Scribble BG" and repeat with the background.
<br />5) If you made a mistake, you can reset the scribbles for this slice using the "Reset" button.
<br />6) Press "Add scribble" to save the scribbles..
<br />7) Add scribbles in 2-3 z-slices ("z-axis"-slider), for the same t value.
<br />8) Press "Run RW 3D" to run the algorithm in 3D, for the t value for which the scribbles were made.
<br />9) Check the result with the "View" button.
<br />10) Add more scribbles if required. If you scribble on a slice where there were already some scribbles, only the latest scribbles are used. This should probably be changed in a future version.
<br />11) To propagate the 3D result to a 4D segmentation, press "Run RW 4D". This uses the 3D segmentation as rough markers for other timesteps with dilation and erosion.
<br />12) Press "Save segmentation" to save the 4D segmentation.
<br />13) Done.
<br />To reset and rerun the RW simply press "Quit" and restart the program.

==========================================================
