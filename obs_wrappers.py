import os
import copy

import gym
import numpy as np
from numba import njit
import nle
import cv2
from PIL import Image, ImageFont, ImageDraw

SMALL_FONT_PATH = os.path.abspath("Hack-Regular.ttf")

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right
# https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080", # - flipped these ones around
    "#C0C0C0", # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


class VectorFeaturesWrapper(gym.Wrapper):
    """Create network-friendly vector features from the stuff nethack has"""

    # Hand-chosen scaling values for each blstat entry. Aims to limit them in [0, 1] range.
    BLSTAT_NORMALIZATION_STATS = np.array([
        1.0 / 79.0, # hero col
        1.0 / 21, # hero row
        0.0, # strength pct
        1.0 / 10, # strength
        1.0 / 10, # dexterity
        1.0 / 10, # constitution
        1.0 / 10, # intelligence
        1.0 / 10, # wisdom
        1.0 / 10, # charisma
        0.0,      # score
        1.0 / 10, # hitpoints
        1.0 / 10, # max hitpoints
        0.0, # depth
        1.0 / 1000, # gold
        1.0 / 10, # energy
        1.0 / 10, # max energy
        1.0 / 10, # armor class
        0.0, # monster level
        1.0 / 10, # experience level
        1.0 / 100, # experience points
        1.0 / 1000, # time
        1.0, # hunger_state
        1.0 / 10, # carrying capacity
        0.0, # carrying capacity
        0.0, # level number
        0.0, # condition bits
    ])

    CROP_CENTER_NORMALIZATION_STATS = np.array([
        1.0 / 20,
        1.0 / 80
    ])

    # Make sure we do not spook the network
    BLSTAT_CLIP_RANGE = (-5, 5)

    def __init__(self, env):
        super().__init__(env)
        num_items = VectorFeaturesWrapper.BLSTAT_NORMALIZATION_STATS.shape[0]
        obs_spaces = {
            'vector_obs': gym.spaces.Box(
                low=VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[0],
                high=VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[1],
                shape=(num_items,),
                dtype=np.float32
            )
        }
        # Add other obs spaces other than blstats
        obs_spaces.update([
            (k, self.env.observation_space[k]) for k in self.env.observation_space if k != "blstats"
        ])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _create_vector_obs(self, obs):
        obs_vector = obs["blstats"] * VectorFeaturesWrapper.BLSTAT_NORMALIZATION_STATS
        np.clip(
            obs_vector,
            VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[0],
            VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[1],
            out=obs_vector
        )

        obs["vector_obs"] = obs_vector
        self.blstats = obs.pop("blstats")

        return obs

    #make a self.blstats to access where the obs.pop("blstats") currently is
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._create_vector_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._create_vector_obs(obs)
        return obs


@njit
def _tile_characters_to_image(
    out_image,
    chars,
    colors,
    output_height_chars,
    output_width_chars,
    char_array,
    offset_h,
    offset_w
):
    """
    Build an image using cached images of characters in char_array to out_image
    """
    char_height = char_array.shape[3]
    char_width = char_array.shape[4]
    for h in range(output_height_chars):
        h_char = h + offset_h
        # Stuff outside boundaries is not visible, so
        # just leave it black
        if h_char < 0 or h_char >= chars.shape[0]:
            continue
        for w in range(output_width_chars):
            w_char = w + offset_w
            if w_char < 0 or w_char >= chars.shape[1]:
                continue
            char = chars[h_char, w_char]
            color = colors[h_char, w_char]
            h_pixel = h * char_height
            w_pixel = w * char_width
            out_image[:, h_pixel:h_pixel + char_height, w_pixel:w_pixel + char_width] = char_array[char, color]


class RenderCharImagesWithNumpyWrapper(gym.Wrapper):
    """
    Render characters as images, using PIL to render characters like we humans see on screen
    but then some caching and numpy stuff to speed up things.

    To speed things up, crop image around the player.
    """

    def __init__(self, env, font_size=9, crop_size=None, rescale_font_size=None):
        super().__init__(env)
        self.char_array = self._initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size

        if crop_size is None:
            # Render full "obs"
            old_obs_space = self.env.observation_space["obs"]
            self.output_height_chars = old_obs_space.shape[0]
            self.output_width_chars = old_obs_space.shape[1]
        else:
            # Render only crop region
            self.half_crop_size = crop_size // 2
            self.output_height_chars = crop_size
            self.output_width_chars = crop_size
        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width
        )

        # sample-factory expects at least one observation named "obs"
        obs_spaces = {
            'obs': gym.spaces.Box(
                low=0,
                high=255,
                shape=self.chw_image_shape,
                dtype=np.uint8
            )
        }
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space if k not in ["tty_chars", "tty_colors"]])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _initialize_char_array(self, font_size, rescale_font_size):
        """Draw all characters in PIL and cache them in numpy arrays

        if rescale_font_size is given, assume it is (width, height)

        Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
        """
        font = ImageFont.truetype(SMALL_FONT_PATH, font_size)
        dummy_text = "".join([(chr(i) if chr(i).isprintable() else " ") for i in range(256)])
        _, _, image_width, image_height = font.getbbox(dummy_text)
        # Above can not be trusted (or its siblings)....
        image_width = int(np.ceil(image_width / 256) * 256)

        if rescale_font_size:
            char_width = rescale_font_size[0]
            char_height = rescale_font_size[1]
        else:
            char_width = image_width // 256
            char_height = image_height

        char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
        image = Image.new("RGB", (image_width, image_height))
        image_draw = ImageDraw.Draw(image)
        for color_index in range(16):
            image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
            image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

            arr = np.array(image).copy()
            arrs = np.array_split(arr, 256, axis=1)
            for char_index in range(256):
                char = arrs[char_index]
                if rescale_font_size:
                    char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
                char_array[char_index, color_index] = char
        return char_array

    def _render_text_to_image(self, obs):
        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        offset_w = 0
        offset_h = 0
        if self.crop_size:
            # Center around player
            center_x, center_y = obs["blstats"][:2]
            offset_h = center_y - self.half_crop_size
            offset_w = center_x - self.half_crop_size

        out_image = np.zeros(self.chw_image_shape, dtype=np.uint8)

        _tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=self.output_height_chars,
            output_width_chars=self.output_width_chars,
            char_array=self.char_array,
            offset_h=offset_h,
            offset_w=offset_w
        )

        # _ = obs.pop("tty_chars")
        self.tty_chars = obs.pop("tty_chars")
        _ = obs.pop("tty_colors")
        obs["obs"] = out_image
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._render_text_to_image(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._render_text_to_image(obs)
        return obs
