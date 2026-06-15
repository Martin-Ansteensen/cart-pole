#!/usr/bin/python3
from pathlib import Path
from collections import deque

import numpy as np
from numpy import array, cos, sin, pi, ndarray
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch
import imageio_ffmpeg
from rich.progress import Progress

from cart_pole.simulation import SimulationResult
from cart_pole.dynamics import PhysicalParamters
from cart_pole.control import *

COLORS = mpl.cm.Paired.colors
LIGHTBLUE = COLORS[0]
DARKBLUE = COLORS[1]
LIGHTGREEN = COLORS[2]
DARKGREEN = COLORS[3]
LIGHTRED = COLORS[4]
DARKRED = COLORS[5]
LIGHTORANGE = COLORS[6]
DARKORANGE = COLORS[7]
POLECOLORS = [LIGHTGREEN, LIGHTORANGE]
MAX_ANIMATION_FPS = 50.0
mpl.rcParams["axes.prop_cycle"] = cycler(color=mpl.colormaps["Set1"].colors)

def visualize_simulation(sim_result: SimulationResult, params: PhysicalParamters,
                          plots_type: str, trace: bool, save_path: Path,
                          show: bool) -> None:
    '''Visualize the simulation with relevant plots and an animation'''
    tile_layout = []
    if plots_type == "line":
        # Create a tile based layout to have both animation and plots
        tile_layout = [
            ['anim', 'x'],
            ['anim', 'xd'],
            ['anim', 'theta'],
            ['u', 'thetad']
        ]        
    elif plots_type == "phase":
        # Create a tile based layout to have both animation and plots
        tile_layout = [
            ['anim', 'p1'],
            ['anim', 'p1'],
            ['anim', 'p2'],
            ['u', 'p2']
        ]
    else:
        tile_layout = [['anim']]

    fig, axs = plt.subplot_mosaic(tile_layout, layout='constrained', figsize=(9, 6))

    plot_overlays = []
    if plots_type:
        plot_overlays = make_plots(axs, sim_result, params, plots_type)

    anim, anim_data = animate_simulation(
        fig,
        axs['anim'],
        sim_result,
        params,
        trace,
        plot_overlays,
    )

    if save_path:
        save_figure(anim, anim_data, save_path)
    if show:
        plt.show()

def animate_simulation(fig, ax, sim_result: SimulationResult,
                       params: PhysicalParamters, trace: bool, plot_overlays=None) -> FuncAnimation:
    '''Animate a precomputed SimulationResult'''

    cart_width = 0.8
    cart_height = 0.4
    # Scale pole circle radius so that area is related to weight
    pole_circle_radius = np.sqrt(cart_width * cart_height / pi * params.m / params.M)

    aspect_ratio = 9.0 / 16.0
    l = sum(params.pole_lengths)
    y_lim = 2*array([-l * 0.3 - cart_height, l * 0.6 + cart_height])        # chosen arbitrarily
    x_lim = array([-0.5, 0.5])                                              # symmetric x-axis
    x_lim *= (y_lim[1] - y_lim[0]) * 1 / aspect_ratio                       # scale to perserve aspect ratio

    ax.set_box_aspect(aspect_ratio)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel('Cart position [m]')
    ax.set_ylabel('Height [m]')
    ax.axhline(0.0, color='black', lw=2, zorder=-2)

    # Origin of patch is in bottom left corner, so we need to offset it
    cart_offset = np.array([-cart_width / 2.0, 0.0])
    cart_patch = FancyBboxPatch((0, 0), cart_width, cart_height,
                                boxstyle='round, pad=0.05', fc=LIGHTBLUE, ec=DARKBLUE, lw=2, zorder=-10)
   
    pole_lines = []
    for i in range(params.n_poles):
        pole_line, = ax.plot([], [], lw=6, color=POLECOLORS[int(i%2)], zorder=-1)
        pole_lines.append(pole_line)

    pole_circle = Circle((0, 0), radius=pole_circle_radius, fc=LIGHTRED, ec=DARKRED, lw=2)

    ax.add_patch(cart_patch)
    ax.add_patch(pole_circle)

    fps = min(MAX_ANIMATION_FPS, 1.0 / sim_result.dt)
    animation_dt = 1.0 / fps
    target_times = np.arange(
        sim_result.time_ts[0],
        sim_result.time_ts[-1],
        animation_dt,
    )
    if len(target_times) == 0:
        target_times = np.array([sim_result.time_ts[0]])
    frame_indices = np.searchsorted(sim_result.time_ts, target_times, side='left')
    frame_indices = np.clip(frame_indices, 0, len(sim_result.time_ts) - 1)
    previous_indices = np.maximum(frame_indices - 1, 0)
    use_previous = (
        np.abs(sim_result.time_ts[previous_indices] - target_times)
        < np.abs(sim_result.time_ts[frame_indices] - target_times)
    )
    frame_indices = np.where(use_previous, previous_indices, frame_indices)

    # trace the pole tip for T s, contains tuples (x, y)
    T = 2; n = int(T * fps)
    tip_trace = deque(maxlen=n)
    tip_trace_line, = ax.plot([], [], lw=2, ls='--', color=LIGHTRED)

    # Display time at the top of the axis
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    plot_overlays = plot_overlays or []
    overlay_lines = [entry['line'] for entry in plot_overlays]

    def init_animation():
        '''Initialization function for animation'''
        tip_trace.clear()   # remove all datapoints in the trace
        for entry in plot_overlays:
            entry['progress'].fill(np.nan)
            entry['line'].set_ydata(entry['progress'])
        artists = [cart_patch, *pole_lines, pole_circle, time_text, tip_trace_line]
        artists.extend(overlay_lines)
        return tuple(artists)

    def update(frame: int):
        '''Update function for animation called each frame'''
        state = sim_result.state_ts[frame]
        x = state[0]
        
        cart_patch.set_x(x + cart_offset[0])        # Place center of cart at the x pos
        pivot_start = np.array([x, cart_height])    # Pivot of first pole        

        theta_cumsum = 0
        for i in range(len(pole_lines)):
            theta_i = state[2*(i+1)]
            theta_cumsum += theta_i
            l_i = params.pole_lengths[i]
            pivot_end = pivot_start + l_i*array([-sin(theta_cumsum), cos(theta_cumsum)])    # end of of pole i, start of next pole
            pole_lines[i].set_data(zip(pivot_start, pivot_end))                             # set start and end point
            pivot_start = pivot_end                                                         # start the next pole at the end of this one

        time_text.set_text(f't = {sim_result.time_ts[frame]:.2f} s')    # update time

        pole_circle.center = pivot_end

        # Draw tip trace
        tip_trace.append(pole_circle.center)
        if trace:
            tip_trace_line.set_data(zip(*tip_trace))

        for entry in plot_overlays:
            progress = entry['progress']
            progress[:frame + 1] = entry['values'][:frame + 1]
            progress[frame + 1:] = np.nan
            entry['line'].set_ydata(progress)

        artists = [cart_patch, *pole_lines, pole_circle, time_text, tip_trace_line]
        artists.extend(overlay_lines)
        return artists

    n_frames = len(frame_indices)
    anim = FuncAnimation(fig, update, frames=frame_indices,
                         interval=1000.0 / fps, blit=True,
                         init_func=init_animation)

    return anim, {'fps': fps, 'n_frames': n_frames}


def add_series(ax, time: ndarray, values: ndarray, label: str=''):
    '''Plot the values in a somewhat transparent line, and return
    a array of nans and a plt line that can be used to animate
    the progression of the data'''
    # plot all of the data in a somewhat transparent line
    line, = ax.plot(time, values, alpha=0.25, zorder=-1)
    color = line.get_color()
    # create a line that starts with all nans (no plot), and progressivley
    # equals the full data
    progress = np.full(len(values), np.nan)
    overlay_line, = ax.plot(time, progress, color=color, label=label) 
    return {'line': overlay_line, 'values': values, 'progress': progress}


def make_plots(axs, sim_result: SimulationResult, params: PhysicalParamters, plots_type: str):
    '''Add relevant plots to the figure and prepare animated overlays'''
    overlays = []
    time = sim_result.time_ts

    if plots_type == "line":

        axs['x'].set_title(r'Cart position $x(t)$')
        axs['x'].set_ylabel(r'$x(t)$ [m]')
        axs['x'].set_xlabel('Time [s]')
        axs['x'].axhline(0.0, color='grey', ls='--', alpha=0.3)
        overlay = add_series(axs['x'], time, sim_result.x_ts)
        overlays.append(overlay)

        axs['xd'].set_title(r'Cart velocity $\dot{x}(t)$')
        axs['xd'].set_ylabel(r'$\dot{x}(t)$ [m]/s')
        axs['xd'].set_xlabel('Time [s]')
        axs['xd'].axhline(0.0, color='grey', ls='--', alpha=0.3)
        overlay = add_series(axs['xd'], time, sim_result.x_dot_ts)
        overlays.append(overlay)

        for i in range(sim_result.n_poles):
            theta_i_ts = sim_result.thetas_ts[:, i]
            theta_dot_i_ts = sim_result.theta_dots_ts[:, i]

            axs['theta'].set_title(rf'Pole angle $\theta_i(t)$')
            axs['theta'].set_ylabel(rf'$\theta_i(t)$ [rad]')
            axs['theta'].set_xlabel('Time [s]')
            axs['theta'].axhline(0.0, color='grey', ls='--', alpha=0.3)
            overlay = add_series(axs['theta'], time, np.unwrap(theta_i_ts), label=rf"$\theta_{i+1}$")
            overlays.append(overlay)

            axs['thetad'].set_title(rf'Pole angular velocity $\dot\theta_i(t)$')
            axs['thetad'].set_ylabel(rf'$\dot\theta_i(t)$ [rad/s]')
            axs['thetad'].set_xlabel('Time [s]')
            axs['thetad'].axhline(0.0, color='grey', ls='--', alpha=0.3)
            overlay = add_series(axs['thetad'], time, theta_dot_i_ts, label=rf"$\dot\theta_{i+1}$")
            overlays.append(overlay)
        axs['theta'].legend(); axs['thetad'].legend()

    elif plots_type == "phase":
        for i in range(sim_result.n_poles):
            theta_i_ts = sim_result.thetas_ts[:, i]
            theta_dot_i_ts = sim_result.theta_dots_ts[:, i]

            # change theta wrapping point to 0, instead of pi
            theta_i_wrapped = (np.unwrap(theta_i_ts)) % (2 * np.pi)

            axs['p1'].set_title(r'$\theta_i(t), x(t)$ phase portrait')
            axs['p1'].set_xlabel(r'$x(t)$ [m]')
            axs['p1'].set_ylabel(r'$\theta_i(t)$ [rad]')
            overlay = add_series(axs['p1'], sim_result.x_ts, theta_i_wrapped, label=rf"$\theta_{i+1}$")
            overlays.append(overlay)

            axs['p2'].set_title(r'$\theta_i(t), \dot\theta_i(t)$ phase portrait')
            axs['p2'].set_xlabel(r'$\dot\theta_i(t)$ [rad/s]')
            axs['p2'].set_ylabel(r'$\theta_i(t)$ [rad]')
            overlay = add_series(axs['p2'], theta_dot_i_ts, theta_i_wrapped, label=rf"$\dot\theta_{i+1}$")
            overlays.append(overlay)
        axs['p1'].legend(); axs['p2'].legend()

    if sim_result.controller == type(None).__name__:
        # We don't have a controller, so we plot the change in energy
        # over time, which should be zero (conservative system)
        axs['u'].set_xlabel('Time [s]')
        axs['u'].set_title(r'Energy difference from $t_0$')
        axs['u'].set_ylabel(r'$E_0 - E(t)$ [J]')
        overlay = add_series(axs['u'], time, sim_result.energy_ts[0] - sim_result.energy_ts)
        overlays.append(overlay)

    else:
        # We have a controller
        axs['u'].set_title(f'Control action u using {sim_result.controller}')
        axs['u'].set_ylabel('u')
        axs['u'].set_xlabel('Time [s]')
        axs['u'].axhline(0.0, color='grey', ls='--', alpha=0.3)
        u_ts = sim_result.u_ts

        # some controllers switch what they use, show this using different colors
        u_type = sim_result.cntrler_type    # np.nan will result in no line
        for (cntroller_name, cntroller_idx) in Controller.controllers_map.items():
            u_ts_where_controller_used = np.where(u_type == cntroller_idx, u_ts, np.nan)
            if np.isnan(u_ts_where_controller_used).all():
                # don't show e.g. HybridController as a legend; it shows up in controllers_map, but the controllers
                # it uses are LQR and Energy
                continue
            overlay = add_series(axs['u'], time, u_ts_where_controller_used, label=cntroller_name.partition("Controller")[0])
            overlays.append(overlay)
            axs['u'].legend(loc='upper right', fontsize=10)

    return overlays


def save_figure(anim, anim_data, save_path: Path):
    '''Save the plt figure as a mp4'''
    mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

    fps = anim_data['fps']
    fps_label = f'{fps:g}'
    n_frames = anim_data['n_frames']
    with Progress() as progress:
        task = progress.add_task('Saving animation', total=n_frames)
        def cb(curr_frame: int, total_frames: int):
            '''Callback function to update progress bar for video saving'''
            progress.update(task, completed=curr_frame)
        if save_path.suffix == '.mp4':
            writer = FFMpegWriter(fps=fps, codec='h264', bitrate=1800)
            anim.save(f'{save_path.with_suffix("")}_{fps_label}fps.mp4', writer=writer, dpi=150, progress_callback=cb)
        else:
            # default gif
            writer = PillowWriter(fps=fps)
            anim.save(
                f'{save_path.with_suffix("")}_{fps_label}fps.gif',
                writer=writer,
                dpi=150,
                progress_callback=cb,
            )
