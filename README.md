# Cart-Pole Control Sandbox

A Python playground for studying the classic cart-pole problem with non-linear dynamics, energy control, linear-quadratic (LQR) balancing and reinforcement learning. The package lets you simulate different controller strategies, compare their behaviour, and export videos of the motion.

## Controller Showcases
Two contrasting initial conditions highlight how each controller behaves compared to the passive dynamics. The state is ordered as $[x, \dot{x}, \theta, \dot{\theta}]$.

<div align="center">
  <img src="media/state_reference.png" width="40%" alt="Cart-pole state definitions" />
</div>

Positive $x$ is measured to the right along the track. The pole angle $\theta$ is measured counter-clockwise from the upright position.

### Near Upright (disturbed but recoverable)
Initial state `[-2.0, 1.0, 0.5, 0.6]`. The cart begins far from the origin with noticeable velocity while the pole is nudged off balance. The LQR controller handles this case well, as it should. The energy controller manages to maintain the large pendulum swing while keeping the cart roughly at 0, but is not able to stabilize the pole in the upright position.

<table width="100%">
  <tr>
    <th width="50%">Passive dynamics</th>
    <th width="50%">LQR</th>
  </tr>
  <tr>
    <td><img src="media/upright_none_100fps.gif" width="100%" alt="Passive dynamics from near-upright initial state" /></td>
    <td><img src="media/upright_lqr_100fps.gif" width="100%" alt="LQR control from near-upright initial state" /></td>
  </tr>
  <tr>
    <th>Energy-based</th>
    <th>Hybrid</th>
  </tr>
  <tr>
    <td><img src="media/upright_energy_100fps.gif" width="100%" alt="Energy controller from near-upright initial state" /></td>
    <td><img src="media/upright_hybrid_100fps.gif" width="100%" alt="Hybrid controller from near-upright initial state" /></td>
  </tr>
</table>

### Near Downward (swing-up required)
Initial state `[0.1, -0.3, 2.8, -0.5]`. The cart begins almost at rest in the downwards position. The LQR controller is not able to handle these conidtions, as the dynamics at this initial state is very different from the linearized ones the LQR controller operates with. The energy controller performs the same as it did for the prior case.

<table width="100%">
  <tr>
    <th width="50%">Passive dynamics</th>
    <th width="50%">LQR (fails far from equilibrium)</th>
  </tr>
  <tr>
    <td><img src="media/downright_none_100fps.gif" width="100%" alt="Passive dynamics from near-downward initial state" /></td>
    <td><img src="media/downright_lqr_100fps.gif" width="100%" alt="LQR control from near-downward initial state" /></td>
  </tr>
  <tr>
    <th>Energy-based</th>
    <th>Hybrid</th>
  </tr>
  <tr>
    <td><img src="media/downright_energy_100fps.gif" width="100%" alt="Energy controller from near-downward initial state" /></td>
    <td><img src="media/downright_hybrid_100fps.gif" width="100%" alt="Hybrid controller from near-downward initial state" /></td>
  </tr>
</table>

As can be seen from these two cases, the LQR and energy controllers are suited to handle different operating conditions. By combining them into one controller, the Hybdrid one, we get the best of both worlds. Below is another video showcasing how whem hybrid controller switches between the energy and LQR controller:

<img src="media/hybrid_plots_100fps.gif" width="100%" alt="Hybrid controller switching between energy and LQR plots" /></td>



## Features
- Non-linear cart-pole dynamics integrated with a fourth-order Runge-Kutta solver.
- Different controllers: no control, LQR, energy-based swing-up, and a hybrid strategy.
- A JSON config file with presets for physical parameters and controller gains (`cart_pole/configs.json`).
- Matplotlib animation with optional video export.


## Installation
- Install Python 3.8+ 
- Clone the repository and install dependencies in editable mode:
  ```bash
  git clone https://github.com/Martin-Ansteensen/cart-pole.git
  cd cart-pole
  pip install -e .
  ```

## Running Simulations
- Launch the driver script with:
  ```bash
  python3 -m cart_pole.main
  ```
  Get help by providing `--help`:
  ```bash
  python3 -m cart_pole.main --help
  ```

- Useful flags:
  - `--initial-state` to supply $x$, $\dot{x}$, $\theta$, $\dot{\theta}$ (defaults to $(1.0, -1.0, 0.3, 0.0)$).
  - `--duration` and `--dt` to control simulation length and step size.
  - `--physical` / `--controller` / `--controller-profile` to pick presets defined in `configs.json`.
  - `--list` to print all available presets.
- Skip `--plots` when you just want the animation pane without the plots

## Exporting Demonstration Videos
- Supply a base filename with `--save-path`; the writer appends the playback FPS (for example `media/upright_lqr_100fps.mp4`).
- Recreate videos in this README:
  ```bash
  # Near upright initial condition
  python3 -m cart_pole.main --controller none --duration 8 --initial-state -2 1 0.5 0.6 --save-path media/upright_none
  python3 -m cart_pole.main --controller lqr --duration 8 --initial-state -2 1 0.5 0.6 --save-path media/upright_lqr
  python3 -m cart_pole.main --controller energy --duration 8 --initial-state -2 1 0.5 0.6 --save-path media/upright_energy
  python3 -m cart_pole.main --controller hybrid --duration 8 --initial-state -2 1 0.5 0.6 --save-path media/upright_hybrid

  # Near downward initial condition
  python3 -m cart_pole.main --controller none --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --save-path media/downright_none
  python3 -m cart_pole.main --controller lqr --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --save-path media/downright_lqr
  python3 -m cart_pole.main --controller energy --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --save-path media/downright_energy
  python3 -m cart_pole.main --controller hybrid --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --save-path media/downright_hybrid
  python3 -m cart_pole.main --controller hybrid --duration 4 --initial-state -3 2 1.8  2 --plots --save-path ./media/hybrid_plots
  ```
- Each command saves an animation-only MP4 that can be embedded directly in documentation or presentations.

## Configuration Notes
- Adjust or extend presets through `cart_pole/configs.json`; each controller profile exposes intuitive gains.
- The dynamics notebook (`dynamics.ipynb`) documents the symbolic derivations and exports `dynamics.pkl`, which `cart_pole/dynamics.py` consumes at runtime.

## Testing
- Basic regression checks live in `tests/`; run them with:
  ```bash
  python3 -m unittest
  ```
