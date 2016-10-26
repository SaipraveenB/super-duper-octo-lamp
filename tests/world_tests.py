from environment.dynamics import MinecraftEnvironment
from environment.l_world import LWorld, LWorld2

lwor = LWorld(10,10)
lwor2 = LWorld2(10,10)

# Get 10 example worlds and print them.
for i in range(0,10):
    print lwor.get_world();

lw = lwor.get_world();


lw2 = lwor2.get_world();

print lw2;

rewards = lw * -10;
rewards[9,9] = 40;
me = MinecraftEnvironment( lw, rewards, 3, [(9,9)] );
print( me.get_actions() );
me.play( 1 );
print( me.get_visibile_pixels() );
print( me.get_visibility_kernel() );
