# Project

The project uses the YOLO v8 algorithm of computer vision to identify the products, which in this case are cars, and plot their centroids in a 2d map, where the user picks a certain threshold of how much distance is still within the range of interest of the selected product (car). Afterwards, the program makes a frame by frame analysis of the trajectory of the people who appear on the video, to model their trajectory and count the frames in which every user is in the zone of interest of each car. It returns a dataframe of the proportion of time that people spent in each zone of interest and a heatmap of the most concurred areas.

You can watch how the procces works in the following [video](https://drive.google.com/file/d/1D0vLoZT-S2my4VO4Z3b0z2g9xXliZTZK/view?usp=drive_link)
