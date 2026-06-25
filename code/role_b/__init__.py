"""Role B (policy-based) implementations for the IE 306 drone-dispatch project.

Method family: REINFORCE + GAE -> A2C on the discrete masked dispatcher
(DroneDispatch-v0), plus DDPG on the continuous control sub-env (DroneControl-v0).

Nothing here modifies the frozen simulator; policies consume only the gymnasium
observation dict and the action mask, per the agent_interface.Policy contract.
"""
