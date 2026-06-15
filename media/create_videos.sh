cart-pole-run --controller none --duration 8 --initial-state 0 0 0.01 0 --plots phase --trace-tip --save-path ./upright_perturb.gif --no-show-animation

# Near upright initial condition
cart-pole-run --controller none --duration 8 --initial-state -2 1 0.5 0.6 --trace-tip --save-path ./upright_none.gif --no-show-animation
cart-pole-run --controller lqr --duration 8 --initial-state -2 1 0.5 0.6 --trace-tip --save-path ./upright_lqr.gif --no-show-animation
cart-pole-run --controller energy --duration 8 --initial-state -2 1 0.5 0.6 --trace-tip --save-path ./upright_energy.gif --no-show-animation
cart-pole-run --controller hybrid --duration 8 --initial-state -2 1 0.5 0.6 --trace-tip --save-path ./upright_hybrid.gif --no-show-animation

# Near downward initial condition
cart-pole-run --controller none --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --trace-tip --save-path ./downright_none.gif --no-show-animation
cart-pole-run --controller lqr --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --trace-tip --save-path ./downright_lqr.gif --no-show-animation
cart-pole-run --controller energy --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --trace-tip --save-path ./downright_energy.gif --no-show-animation
cart-pole-run --controller hybrid --duration 8 --initial-state 0.1 -0.3 2.8 -0.5 --trace-tip --save-path ./downright_hybrid.gif --no-show-animation

cart-pole-run --controller lqr --duration 5 --initial-state 3 1 1.1 0 --trace-tip --save-path ./lqr_almost_unstable.gif --no-show-animation
cart-pole-run --controller hybrid --duration 5 --initial-state 3 1 1.1 0 --trace-tip --save-path ./hybrid_very_stable.gif --no-show-animation
# Hybrid controller with plots
cart-pole-run --controller hybrid --duration 4 --initial-state -3 2 1.8 2 --plots line --trace-tip --save-path ./hybrid_plots.gif --no-show-animation
# NMPC with plots
cart-pole-run --controller nmpc --duration 4 --initial-state -3 2 1.8 2 --plots line --trace-tip --save-path ./nmpc_plots.gif --no-show-animation

# Tabular Q-learning
cart-pole-run --controller q_learning --duration 6 --initial-state 1.2 0.1 0.2 0.1 --plots line --trace-tip --save-path ./highlight_q_learning.gif --no-show-animation

# Deep Q-Network
cart-pole-run --controller dqn --duration 8 --dt 0.02 --initial-state 3 0 0.8 0 --plots line --trace --save-path ./dqn_highlight.gif --no-show-animation

# Double pendulum
cart-pole-run --controller none --duration 8 --initial-state 0 0 0 0 -0.5 0   --plots phase --system double --dt 0.005 --save-path double_pendulum.gif --no-show-animation
cart-pole-run --controller lqr --duration 6 --initial-state 1 0 2.6 0 3.14 0  --plots line --system double --dt 0.005 --target 0 0 3.14 0 3.14 0 --controller-profile gentle --save-path double_down_up.gif --no-show-animation
cart-pole-run --controller lqr --duration 6 --initial-state 1 0 -0.3 0 0.2 0  --plots line --system double --dt 0.005 --target 0 0 0 0 0 0 --controller-profile gentle --save-path double_up_up.gif --no-show-animation
cart-pole-run --controller lqr --duration 8 --initial-state 1 0 0 0 2.14 0  --plots line --system double --dt 0.005 --target 0 0 0 0 3.14 0 --controller-profile gentle --save-path double_up_down.gif --no-show-animation
cart-pole-run --controller lqr --duration 6 --initial-state 4 0 3.14 0 0 0  --plots line --system double --dt 0.005 --target 0 0 3.14 0 0 0 --controller-profile aggressive --save-path double_down_down.gif --no-show-animation