
import numpy as np
from nav_msgs.msg import OccupancyGrid as occGrid
from nav_msgs.msg import MapMetaData as occMetaData
import rospy

def importTest():
    print('import successful for occupancy_grid')

class StaticOccupancyGrid():
    def __init__(self, buffer = 0.5):
        # set up subscriber for both map and map metadata topics. Read both and store for use.
        rospy.Subscriber('map', occGrid, self.occGridCallback)
        self.buffer = buffer
        self.loaded = False

    def occGridCallback(self, data):
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
                    occGridDynamic[np.maximum(x-ind_width,0):np.minimum(x+ind_width,self.metaData.width), np.maximum(y-ind_width,0):np.minimum(y+ind_width,self.metaData.height)] = val
                except:
                    print('bounds x: ', self.metaData.width, 'bounds y: ', self.metaData.height)
                    print(ind_width)
                    print(np.maximum(x-ind_width,0),np.minimum(x+ind_width,self.metaData.width), np.maximum(y-ind_width,0),np.minimum(y+ind_width,self.metaData.height))
        self.occGrid = {'static': occGridStatic,
                   'dynamic': occGridDynamic}
        self.loaded = True

    def getIndexFromWorldPoint(self, point):
        if(not self.loaded):
            return np.array([-1,-1])
        xy_ind = np.round((point - self.mapPosition)/self.metaData.resolution)
        return xy_ind

    def checkForCollision(self, segment):
        if(not self.loaded):
            return False
        knot_pts = np.linspace(segment.pt1, segment.pt2, np.linalg.norm(segment.pt1-segment.pt2)*2//self.metaData.resolution)
        print(self.getIndexFromWorldPoint(knot_pts))
        return np.any(self.occGrid['dynamic'][self.getIndexFromWorldPoint(knot_pts)] > 0)
