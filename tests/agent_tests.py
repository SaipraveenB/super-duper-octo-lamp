from agent import DynaAgent
from environment.dynamics import MinecraftEnvironmentGenerator

agent = DynaAgent( MinecraftEnvironmentGenerator(10,10) );
agent.run_pr();

