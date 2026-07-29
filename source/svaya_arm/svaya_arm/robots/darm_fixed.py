import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

left_dof_names = [f"left_arm_joint{i+1}" for i in range(7)]
right_dof_names = [f"right_arm_joint{i+1}" for i in range(7)]

DARM_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="data/darm_fixed/darm_fixed.usd",
        # usd_path="/home/darms/Documents/gautam_ws/isaac/svaya_arm/data/darm_fixed/darm_fixed.usd",  
        activate_contact_sensors=True,          # ← required for contact reporting
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,       # ← PhysX must track self contacts
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "joint_acts": ImplicitActuatorCfg(
            joint_names_expr=left_dof_names + right_dof_names,
            damping=None,
            stiffness=None,
            # damping=400.0,
            # stiffness=40.0,
            # effort_limit=175.0, 
        )
    },
)