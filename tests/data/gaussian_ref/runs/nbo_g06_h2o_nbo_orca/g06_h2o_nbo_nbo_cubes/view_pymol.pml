# PyMOL script to visualize NBO surfaces
# Open with: pymol view_pymol.pml
bg_color white
load g06_h2o_nbo_nbo1_CR_O_1.cube, map_nbo_1
isosurface nbo_1_pos, map_nbo_1, 0.0500
isosurface nbo_1_neg, map_nbo_1, -0.0500
color marine, nbo_1_pos
color firebrick, nbo_1_neg
set transparency, 0.3, nbo_1_pos
set transparency, 0.3, nbo_1_neg
load g06_h2o_nbo_nbo2_LP_O_1.cube, map_nbo_2
isosurface nbo_2_pos, map_nbo_2, 0.0500
isosurface nbo_2_neg, map_nbo_2, -0.0500
color marine, nbo_2_pos
color firebrick, nbo_2_neg
set transparency, 0.3, nbo_2_pos
set transparency, 0.3, nbo_2_neg
