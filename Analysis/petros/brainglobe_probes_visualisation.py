# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:22:52 2024

@author: chalasp
"""

#from pathlib import Path
import numpy as np

from brainrender import Scene
from brainrender.actors import Points



scene = Scene(atlas_name='kim_mouse_50um', title="Silicon Probe Visualization")

# Visualise the probe target regions
pag = scene.add_brain_region("PAG", alpha=0.15)

# Add probes to the scene.
# Each .npy file should contain a numpy array with the coordinates of each
# part of the probe.
scene.add(
    Points(
        np.load(r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16618\histology\segmentation\atlas_space\tracks\track_0.npy'),
        name="afm16618",
        colors="firebrick",
        radius=50,
    )
)

scene.add(
    Points(
        np.load(r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16963\histology\segmentation\atlas_space\tracks\track_1.npy'),
        name="afm16963",
        colors="darkorange",
        radius=50,
    )
)

scene.add(
    Points(
        np.load(r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\histology\segmentation\atlas_space\tracks\track_0.npy'),
        name="afm16924",
        colors="olivedrab",
        radius=50,
    )
)

scene.add(
    Points(
        np.load(r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16505\histology\brain annotation\brainreg_090224\segmentation\atlas_space\tracks\shorter_likely_accurate\track_1.npy'),
        name="afm16505",
        colors="mediumblue",
        radius=50,
    )
)



# render
scene.render()