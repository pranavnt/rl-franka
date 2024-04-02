import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("./mujoco_menagerie/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)

mujoco.mj_step(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    mujoco.mj_step(model, data)

    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    viewer.sync()

    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
