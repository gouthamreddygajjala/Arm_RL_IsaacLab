import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

left_dof_names = [f"left_arm_joint{i+1}" for i in range(7)]
right_dof_names = [f"right_arm_joint{i+1}" for i in range(7)]


# Values as auto-imported into the USD from the URDF conversion (read via
# damping=None/stiffness=None, i.e. whatever drive gains got baked in at
# USD-import time — a uniform damping/stiffness ratio of exactly 0.0004
# across every joint, not something anyone tuned). Kept explicit here for
# joints 1-6 so the actuator block behaves identically to before; only
# joint7 (both arms) is being overridden below.
AUTO_STIFFNESS = {
    "left_arm_joint1": 470616.25, "right_arm_joint1": 471442.22,
    "left_arm_joint2": 318125.94, "right_arm_joint2": 298649.56,
    "left_arm_joint3": 191728.39, "right_arm_joint3": 201349.55,
    "left_arm_joint4": 58243.40,  "right_arm_joint4": 58211.20,
    "left_arm_joint5": 55634.37,  "right_arm_joint5": 56006.61,
    "left_arm_joint6": 4255.05,   "right_arm_joint6": 4255.50,
    "left_arm_joint7": 660.0,   "right_arm_joint7": 660.0,
}
AUTO_DAMPING = {
    "left_arm_joint1": 188.25, "right_arm_joint1": 188.58,
    "left_arm_joint2": 127.25, "right_arm_joint2": 119.46,
    "left_arm_joint3": 76.69,  "right_arm_joint3": 80.54,
    "left_arm_joint4": 23.30,  "right_arm_joint4": 23.28,
    "left_arm_joint5": 22.25,  "right_arm_joint5": 22.40,
    "left_arm_joint6": 1.70,   "right_arm_joint6": 1.70,
    "left_arm_joint7": 0.526,  "right_arm_joint7": 0.526,
}

# ------------------------------------------------------------------
# joint7 override — pick ONE based on what the freeze test shows.
# Leave both defined; just flip JOINT7_MODE.
# ------------------------------------------------------------------
# "oscillating": velocity keeps changing sign / ringing around the held
#     setpoint -> underdamped for its actual inertia. Raise damping
#     faster than stiffness (push the ratio up for this joint only).
# "compliant":   velocity decays smoothly but settles at a nonzero
#     offset / sags -> correctly damped but too soft. Raise stiffness
#     and damping together, keeping the 0.0004 ratio.
JOINT7_MODE = "oscillating"  # "oscillating" | "compliant"

if JOINT7_MODE == "oscillating":
    JOINT7_STIFFNESS = AUTO_STIFFNESS["left_arm_joint7"]          # unchanged: 1315.24
    JOINT7_DAMPING = AUTO_DAMPING["left_arm_joint7"] * 1.0        # ~2.10
else:
    JOINT7_STIFFNESS = AUTO_STIFFNESS["left_arm_joint7"] * 3.0    # ~3945.71
    JOINT7_DAMPING = AUTO_DAMPING["left_arm_joint7"] * 3.0        # ~1.578

STIFFNESS_OVERRIDE = dict(AUTO_STIFFNESS)
DAMPING_OVERRIDE = dict(AUTO_DAMPING)
STIFFNESS_OVERRIDE["left_arm_joint7"] = JOINT7_STIFFNESS
STIFFNESS_OVERRIDE["right_arm_joint7"] = JOINT7_STIFFNESS
DAMPING_OVERRIDE["left_arm_joint7"] = JOINT7_DAMPING
DAMPING_OVERRIDE["right_arm_joint7"] = JOINT7_DAMPING


DARM_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/darms/Documents/gautam_ws/isaac/svaya_arm/data/darm_fixed/darm_fixed.usd",
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
            stiffness=STIFFNESS_OVERRIDE,
            damping=DAMPING_OVERRIDE,
            # effort_limit=175.0,
        )
    },
)