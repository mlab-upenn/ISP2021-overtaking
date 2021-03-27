import torch
import numpy as np
from matplotlib import pyplot as plt


class TreeMotionPrimitive:
    def __init__(self, speed_list, steering_list, depth=3, L=.33, p=.2, t_la=.8, k1=.0, k2=.0, k3=.0, m=.1, c=.12, local_grid_size = 7, resolution=(50, 50)):
        # speed and steering are lists of speed and steering values
        # depth : depth of the primitive tree
        #L : wheelbase of the car
        # p : not currently used
        # t_la : lookahead time
        # k1 : interior std scaling with steering input
        # k2 : exterior std scaling with steering input
        # k3 : exterior std scaling with speed
        # m : path length std scaling
        # c : width of the car

        self.L = L
        self.p = p
        self.t_la = t_la
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.m = m
        self.c = c
        self.local_grid_size = local_grid_size
        self.resolution = resolution

        #create sets of speeds and angles to create trees
        speed_vals = torch.tensor(speed_list)
        steering_vals = torch.tensor(steering_list)

        speed, steering = torch.meshgrid(speed_vals, steering_vals)

        speed = torch.flatten(speed)
        steering = torch.flatten(steering)


        self.speeds = speed
        self.steering_angles = steering

        self.x, self.y = torch.meshgrid(torch.arange(0, local_grid_size+.0001, local_grid_size/resolution[0]), torch.arange(-local_grid_size/2,local_grid_size/2+.0001, local_grid_size/resolution[1]))

        xy_offset = torch.zeros((torch.numel(speed), 2))
        theta_offset = torch.ones((torch.numel(speed)))*0


        arc_length = torch.ones((torch.numel(speed)))*0

        speeds = self.speeds
        steering_angles = self.steering_angles

        for d in range(depth):

            new_primitives, xy_offset, theta_offset, arc_length = self.create_primitives(speeds, steering_angles, xy_offset, theta_offset, arc_length)
            if(d>0):
                primitives += new_primitives
            else:
                primitives = new_primitives
            primitives, speeds, steering_angles, xy_offset, theta_offset, arc_length = self.create_next_level_input(speeds, steering_angles, primitives, xy_offset, theta_offset, arc_length)

        self.primitives = primitives

    def create_next_level_input(self, speed, steering, primitives, xy_offset, theta_offset, arc_length):
        primitives = torch.repeat_interleave(primitives, speed.shape[0], dim=0)
        arc_length = torch.repeat_interleave(arc_length, speed.shape[0], dim=0)
        theta_offset = torch.repeat_interleave(theta_offset, speed.shape[0], dim=0)
        xy_offset = torch.repeat_interleave(xy_offset, speed.shape[0], dim=0)

        steering = steering.repeat(speed.shape[0])
        speed = speed.repeat(speed.shape[0])

        return primitives, speed, steering, xy_offset, theta_offset, arc_length


    def create_primitives(self, speed, steering, xy_offset, theta_offset, arc_length):

        # print(speed.shape, steering.shape, xy_offset.shape, theta_offset.shape, arc_length.shape )
        primitives = torch.zeros((torch.numel(speed), self.resolution[0] + 1, self.resolution[1] + 1))
        turn_mask = torch.abs(steering) > 0.01
        straight_mask = torch.abs(steering) <= 0.01

        grid_x, grid_y = self.translate_xy(self.x, self.y, xy_offset, theta_offset)

        local_theta = torch.zeros((torch.numel(speed)))
        local_xy = torch.zeros((torch.numel(speed), 2))

        if (torch.any(straight_mask)):
            straight_sig = self.get_sig(speed[straight_mask], steering[straight_mask], grid_x[straight_mask], grid_y[straight_mask], arc_length[straight_mask], True)
            straight_a = self.get_a(speed[straight_mask], steering[straight_mask], grid_x[straight_mask], grid_y[straight_mask], arc_length[straight_mask], True)

            primitives[straight_mask] = straight_a * torch.exp(-(grid_y[straight_mask] ** 2) / (2 * straight_sig ** 2))
            local_xy[straight_mask,0] = speed[straight_mask]*self.t_la


        # handle turn case:
        if (torch.any(turn_mask)):
            turn_sig = self.get_sig(speed[turn_mask], steering[turn_mask], grid_x[turn_mask], grid_y[turn_mask], arc_length[turn_mask], False)
            turn_a = self.get_a(speed[turn_mask], steering[turn_mask], grid_x[turn_mask], grid_y[turn_mask], arc_length[turn_mask], False)
            R = self.get_R(speed[turn_mask], steering[turn_mask]).view(-1, 1, 1)
            primitives[turn_mask] = turn_a * torch.exp(
                -(torch.sqrt(grid_x[turn_mask] ** 2 + (grid_y[turn_mask] - R) ** 2) - torch.abs(R)) ** 2 / (2 * turn_sig ** 2))

            R = R.flatten()
            turn_angle = speed[turn_mask]*self.t_la/R

            local_xy[turn_mask, 0] = torch.sin(turn_angle)*R

            print(local_xy.shape, turn_mask.shape, turn_angle.shape, speed.shape, R.shape)
            local_xy[turn_mask, 1] = (1-torch.cos(turn_angle))*R

            local_theta[turn_mask] = turn_angle

        arc_length += speed*self.t_la
        new_xy_offset, new_theta_offset = self.calculate_new_distance(xy_offset, theta_offset, local_xy, local_theta)
        return primitives, new_xy_offset, new_theta_offset, arc_length

    def get_a(self, speed, steering, grid_x, grid_y, arc_length, straight):
        # a should be (N[straight/turn], res[0], res[1])
        s = self.get_s(speed, steering, grid_x, grid_y, arc_length, straight)

        a = (torch.clamp_min(1 - self.p * s/speed.view(-1, 1, 1), 0))
        # a = self.p*(torch.clamp_max(speed.view(-1,1,1)*self.t_la - s, 0))

        # this is to keep tuning consistent, negates the impact of p
        # normalize = self.p * (speed.view(-1, 1, 1) * self.t_la)

        a = torch.clamp_min(a ,0)
        # print(a.shape, s.shape)
        a[s-arc_length.reshape(-1,1,1)<0] = 0

        a[(s-arc_length.reshape(-1,1,1))/speed.view(-1, 1, 1) > self.t_la] = 0
        return a

    #old version
    def get_s(self, speed, steering, grid_x, grid_y, arc_length, straight):
        #s should be (N[straight/turn], res[0], res[1])

        if(straight):
            return grid_x + arc_length.reshape(-1,1,1)
        else:
            R = self.get_R(speed,steering).view(-1,1,1)
            # radius of a point in relation to turn center times sweep angle gives arc length
            #not clear which formulation is best

            return torch.abs(R)*torch.atan2(grid_x, (R-grid_y)*torch.sign(R))+ arc_length.reshape(-1,1,1)

    def get_sig(self, speed, steering, grid_x, grid_y, arc_length, straight):
        #sig should be (N, res[0], res[1])
        #just use K1 for now

        s = self.get_s(speed, steering, grid_x, grid_y, arc_length, straight) #
        R = self.get_R(speed,steering).view(-1,1,1)
        real_R = torch.sqrt(grid_x.unsqueeze(0)**2 + (R-grid_y.unsqueeze(0))**2)
        k = torch.where(torch.abs(R)>real_R, torch.tensor(self.k1).view(-1,1,1), self.k2 + self.k3*speed.view(-1,1,1))
        sig = (self.m + k*torch.abs(steering.view(-1,1,1)))*s + self.c

        return sig


    #old version
    def get_R(self, speed, steering):
        # only valid for steering[turn_mask]
        R = self.L / torch.tan(steering)
        return R


    def translate_xy(self, x, y, xy_offset, theta_offset):
        #return new x,y
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        theta_offset = theta_offset.reshape(-1,1,1)

        # print(theta_offset.shape, x.shape, xy_offset.shape)
        shift_x = x - xy_offset[:,0].reshape(-1,1,1)
        shift_y = y - xy_offset[:,1].reshape(-1,1,1)
        return (torch.cos(-theta_offset)*shift_x - torch.sin(-theta_offset)*shift_y), (torch.sin(-theta_offset)*shift_x + torch.cos(-theta_offset)*shift_y)
        # return (torch.cos(-theta_offset)*x - torch.sin(-theta_offset)*y)-xy_offset[:,0].reshape(-1,1,1), (torch.sin(-theta_offset)*x + torch.cos(-theta_offset)*y)-xy_offset[:,1].reshape(-1,1,1)

    def calculate_new_distance(self, xy_offset, theta_offset, local_xy, local_theta):


        xy_offset[:,0] += torch.cos(theta_offset)*local_xy[:,0] - torch.sin(theta_offset)*local_xy[:,1]
        xy_offset[:,1] += torch.sin(theta_offset)*local_xy[:,0] + torch.cos(theta_offset)*local_xy[:,1]

        theta_offset += local_theta

        return xy_offset, theta_offset

if __name__ == '__main__':
    import time
    t_b = time.time()
    # MP = MotionPrimitive(list(torch.arange(2,4,.05)),list(torch.arange(-.5,.5,.05)) )
    MP = TreeMotionPrimitive([2.], [0.0,0.1])

    test_offset = torch.tensor([[0.,0.]])
    test_theta_offset = torch.tensor([0.])
    # print(MP.translate_xy(MP.x, MP.y, test_offset, test_theta_offset))


    t_a = time.time()

    # print(t_a-t_b)

    # print(MP.primitives.shape)
    print(MP.steering_angles)
    plt.imshow(np.array(MP.primitives[12]))
    plt.colorbar()

    plt.show()
