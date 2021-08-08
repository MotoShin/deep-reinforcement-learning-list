import gym
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

class CartPole(object):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

    def __init__(self):
        self.env = gym.make('CartPole-v1').unwrapped
        self.max_step_num = 200
        self.step_num = 0
        self.maximum_action_value = 1
        self.minimum_action_value = 0

    def _get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0) # MIDDLE OF CART

    def _pillow_grayscale(sefl, screen):
        gamma22LUT  = [pow(x/255.0, 2.2)*255 for x in range(256)] * 3
        gamma045LUT = [pow(x/255.0, 1.0/2.2)*255 for x in range(256)]

        screen = torch.from_numpy(np.array(screen)).permute(1, 2, 0).numpy()
        img = Image.fromarray(screen)

        img_resize = img.resize((84, 84))

        img_rgb = img_resize.convert("RGB") # any format to RGB
        img_rgbL = img_rgb.point(gamma22LUT)
        img_grayL = img_rgbL.convert("L")  # RGB to L(grayscale)
        img_gray = img_grayL.point(gamma045LUT)
        return np.asarray(img_gray)

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height*0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self._get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        # screen = np.ascontiguousarray(screen, dtype=np.float32)
        # Resize, and add a batch dimension (BCHW)
        
        gray_screen = self._pillow_grayscale(screen)
        return np.array([gray_screen])
        # return screen

    def episode_end_reward(self, reward):
        return -reward
    
    def get_n_actions(self):
        return self.env.action_space.n

    def get_number_of_input_action(self):
        # 入力するactionの配列の数
        return 1
    
    def reset(self):
        self.step_num = 0
        return self.env.reset()

    def step(self, action):
        self.step_num += 1
        return self.env.step(action)

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()
