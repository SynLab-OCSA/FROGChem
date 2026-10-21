# PyMOL script to visualize NBO surfaces
# Open with: pymol view_pymol.pml
bg_color white
load g06_h20_nbo_e2p1_donor_e2p2_donor_nbo4_LP_O_1.cube, map_nbo_1
isosurface nbo_1_pos, map_nbo_1, 0.0500
isosurface nbo_1_neg, map_nbo_1, -0.0500
color marine, nbo_1_pos
color firebrick, nbo_1_neg
set transparency, 0.3, nbo_1_pos
set transparency, 0.3, nbo_1_neg
load g06_h20_nbo_e2p1_acceptor_nbo16_RYstar_H_2.cube, map_nbo_2
isosurface nbo_2_pos, map_nbo_2, 0.0500
isosurface nbo_2_neg, map_nbo_2, -0.0500
color marine, nbo_2_pos
color firebrick, nbo_2_neg
set transparency, 0.3, nbo_2_pos
set transparency, 0.3, nbo_2_neg
load g06_h20_nbo_e2p2_acceptor_nbo20_RYstar_H_3.cube, map_nbo_3
isosurface nbo_3_pos, map_nbo_3, 0.0500
isosurface nbo_3_neg, map_nbo_3, -0.0500
color marine, nbo_3_pos
color firebrick, nbo_3_neg
set transparency, 0.3, nbo_3_pos
set transparency, 0.3, nbo_3_neg
