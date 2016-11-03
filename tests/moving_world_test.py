from environment.moving_world import MovingWorld
from utils.imaging import dump_upscaled_image

a = MovingWorld(28, 28, (3, 3))
dump_upscaled_image(a.inner_grid, 40, "/Users/saipraveenb/cseiitm/sauce/moving_world.png")
