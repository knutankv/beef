# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:00:29 2020

@author: knutankv
"""

import beef.fe

el_3d = beef.fe.BeamElement3d([beef.fe.Node(1, [1,0,0]), 
                            beef.fe.Node(2, [2,0,0])], 1)

el_2d = beef.fe.BeamElement2d([beef.fe.Node(1, [1,0]), 
                               beef.fe.Node(2, [2,0])], 1)