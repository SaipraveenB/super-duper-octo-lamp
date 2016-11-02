from environment.back_world_easy import BackWorldEasy
from environment.back_world_hard import BackWorldHard
from utils.imaging import dump_upscaled_image

a = BackWorldEasy(28, 28, (3, 3))
dump_upscaled_image(a.inner_grid, 40, "/home/sauce/img-1.png")

e = BackWorldHard(28, 28, (3, 3))
dump_upscaled_image(e.inner_grid, 40, "/home/sauce/img-2.png")
b_1, c_1 = e.start()
