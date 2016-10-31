import environment.pattern_matcher as pm

a = pm.SimplePatternWorld(32, 32, (3, 3))
pm.dump_upscaled_image(a.inner_grid, 40, "/home/sauce/img.png")
b, c = a.start()
