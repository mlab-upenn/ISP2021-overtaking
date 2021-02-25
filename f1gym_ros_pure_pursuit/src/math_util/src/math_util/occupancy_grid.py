
import numpy as np
import geometric_types
from nav_msgs.msg import OccupancyGrid as occGrid
from nav_msgs.msg import MapMetaData as occMetaData
import rospy

import matplotlib.pyplot as plt

def importTest():
    print('import successful for occupancy_grid')

class StaticOccupancyGrid():
    def __init__(self, buffer = .2):
        # set up subscriber for both map and map metadata topics. Read both and store for use.
        self.grid_pub = rospy.Publisher('occ_dynamic', occGrid, queue_size = 1)
        rospy.Subscriber('map', occGrid, self.occGridCallback)
        self.buffer = buffer
        self.loaded = False
        self.occGrid = None

    def occGridCallback(self, data):
        if(self.loaded ):
            return
        self.occGridMsg = data
        self.metaData = data.info
        self.mapPosition = np.array([self.metaData.origin.position.x, self.metaData.origin.position.y])
        occGridStatic = np.zeros((self.metaData.width, self.metaData.height))
        occGridDynamic = np.zeros((self.metaData.width, self.metaData.height))
        for i,val in enumerate(data.data):
            x = i%self.metaData.width
            y = i//self.metaData.width
            occGridStatic[x, y] = val
            if(val > 0):
                ind_width = np.floor(self.buffer/self.metaData.resolution).astype(np.int)
                try:
                    occGridDynamic[np.maximum(x-ind_width,0):np.minimum(x+ind_width,self.metaData.width-1)+1, np.maximum(y-ind_width,0):np.minimum(y+ind_width,self.metaData.height-1)+1] = val
                except:
                    print('bounds x: ', self.metaData.width, 'bounds y: ', self.metaData.height)
                    print(ind_width)
                    print(np.maximum(x-ind_width,0),np.minimum(x+ind_width,self.metaData.width), np.maximum(y-ind_width,0),np.minimum(y+ind_width,self.metaData.height))
        self.occGrid = {'static': occGridStatic,
                   'dynamic': occGridDynamic}
        self.loaded = True

        rospy.loginfo('occ grid loaded')

        

    def getIndexFromWorldPoint(self, point):
        if(not self.loaded):
            return np.array([-1,-1])
        xy_ind = np.round((point - self.mapPosition)/self.metaData.resolution)
        return xy_ind

    def checkForCollision(self, segment):
        if(not self.loaded):
            return False
        knot_pts = np.linspace(segment.pt1, segment.pt2, (np.linalg.norm(segment.pt1-segment.pt2)*2)//self.metaData.resolution)
        indicies = self.getIndexFromWorldPoint(knot_pts).astype(np.int)
        values = self.occGrid['dynamic'][indicies[:,0],indicies[:,1]]
        return np.any(values > 0)


