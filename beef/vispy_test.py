# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:19:09 2020

@author: knutankv
"""

import vispy
import numpy as np
vispy.use('PyQt5')

from vispy import visuals, scene, app

#%% Fake data
x1 = np.arange(0, 100)
x2 = np.arange(1, 101)

y1 = np.arange(0, 100)*0
y2 = np.arange(0, 100)*0

z1 = np.arange(0,100)*2
z2 = np.arange(0,100)*3

pos1 = np.vstack([x1,y1,z1]).T
pos2 = np.vstack([x2,y2,z2]).T

#%% Plot
canvas = scene.SceneCanvas()
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=60)

#%%
xy = np.random.rand(2000,2) # 2D positions
# Create an array of point connections :
# Point 0 in your xy will be connected with 1 and 2, point 
# 1 with 4 and point 2 with 3 and 4.
line = scene.visuals.Line(pos=xy, connect='segments')
view.add(line)
canvas.show()

#%%
axis = scene.visuals.XYZAxis(parent=view.scene)
canvas.show()
