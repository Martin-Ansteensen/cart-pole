#!/usr/bin/python3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import array, cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, FancyBboxPatch
import imageio_ffmpeg
from rich.progress import Progress

from cart_pole.simulation import SimulationResult
from cart_pole.dynamics import PhysicalParamters

def visaualize_simulation(sim_result: SimulationResult, params: PhysicalParamters,
                          save_path: Path) -> None:
    '''Visualize the simulation with relevant plots and an animation'''
    # Create a tile based layout
    fig, axs = plt.subplot_mosaic([['anim', 'h1'],
                                   ['anim', 'h2'],
                                   ['anim', 'h3'],
                                   ['h5', 'h4']],
                                   layout='constrained', figsize=(12, 8))

    fig.suptitle(f"Pole cart animation. Cart weight: {params.M} kg, pole weight {params.m} kg")
    anim, anim_data = animate_simulation(fig, axs["anim"], sim_result, params)
    make_plots(axs, sim_result)

    if save_path:
        save_figure(anim, anim_data, save_path)
    plt.show()


def animate_simulation(fig, ax, sim_result: SimulationResult, params: PhysicalParamters) -> FuncAnimation:
    '''Animate a precomputed SimulationResult'''
    colors = mpl.cm.Paired.colors

    cart_width = 0.8
    cart_height = 0.4
    # Scale pole circle radius so that area is related to weight
    pole_circle_radius = np.sqrt(cart_width * cart_height / pi * params.m / params.M)

    aspect_ratio = 9.0 / 16.0                                                   # height / width                
    y_lim = 2*array([-params.l * 0.4 - cart_height, params.l * 1.1 + cart_height])       # chosen arbitrarily
    x_lim = array([-0.5, 0.5])                                                  # symmetric x-axis
    x_lim *= (y_lim[1] - y_lim[0]) * 1 / aspect_ratio                           # scale to perserve aspect ratio

    ax.set_box_aspect(aspect_ratio)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel('Cart position [m]')
    ax.set_ylabel('Height [m]')
    ax.axhline(0.0, color='black', lw=2)

    # Origin of patch is in bottom left corner, so we need to offset it
    cart_offset = np.array([-cart_width / 2.0, 0.0])
    cart_patch = FancyBboxPatch((0, 0), cart_width, cart_height, 
                                boxstyle='round, pad=0.05', fc=colors[0], ec=colors[1], lw=2) 
    pole_line, = ax.plot([], [], lw=6, color=colors[2], zorder=-1)
    pole_circle = Circle((0, 0), radius=pole_circle_radius, fc=colors[4], ec=colors[5], lw=2)

    ax.add_patch(cart_patch)
    ax.add_patch(pole_circle)

    # Display time at the top of the axis
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame: int):
        '''Update function for animation called each frame'''
        state = sim_result.state_ts[frame]
        x = state[0]
        theta = state[2]

        cart_patch.set_x(x + cart_offset[0])                                    # Place center of cart at the x pos
        pivot = np.array([x, cart_height])                                      # Pivot of pole
        pole_circle.center = pivot + params.l*array([sin(theta), cos(theta)])   # Center of pole circle
        pole_line.set_data(zip(pivot, pole_circle.center))                      # set x-data and y-data
        time_text.set_text(f't = {sim_result.time_ts[frame]:.2f} s')            # update time
        return cart_patch, pole_line, pole_circle, time_text


    # FPS to play back simulation in real time
    fps = 1 / sim_result.dt
    n_frames = len(sim_result.time_ts)
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / fps, blit=True)
    
    return anim, {'fps': fps, 'n_frames': n_frames}


def make_plots(axs, sim_result: SimulationResult):
    '''Add relevant plots to the figure'''
    
    # Common setup for all axis
    for ax_name, ax in axs.items():
        if ax_name == 'anim':
            continue
        ax.set_xlabel("Time [s]")
        ax.axhline(0.0, color='grey', ls='--')

    axs['h1'].set_title("Cart position x(t)")
    axs['h1'].set_ylabel("x(t) [m]")
    axs['h1'].plot(sim_result.time_ts, sim_result.x_ts)

    axs['h2'].set_title("Cart velocity x'(t)")
    axs['h2'].set_xlabel("Time [s]")
    axs['h2'].set_ylabel("x'(t) [m]/s")
    axs['h2'].plot(sim_result.time_ts, sim_result.x_dot_ts)

    axs['h3'].set_title("Pole angle theta(t)")
    axs['h3'].set_ylabel("theta(t) [rad]")
    axs['h3'].plot(sim_result.time_ts, sim_result.theta_ts)

    axs['h4'].set_title("Pole angular velocity theta'(t)")
    axs['h4'].set_ylabel("theta(t) [rad/s]")
    axs['h4'].plot(sim_result.time_ts, sim_result.theta_dot_ts)

    if sim_result.controller != type(None).__name__:
        axs['h5'].set_title("Control action u")
        axs['h5'].set_xlabel("Time [s]")
        axs['h5'].set_ylabel("u")
        axs['h5'].plot(sim_result.time_ts, sim_result.u_ts)
    else:
        # Only plot energy when we have a conservative system
        axs['h5'].set_title("Energy difference from t0")
        axs['h5'].set_xlabel("Time [s]")
        axs['h5'].set_ylabel("E0 - E(t) [J]")
        axs['h5'].plot(sim_result.time_ts, sim_result.energy_ts[0] - sim_result.energy_ts)


def save_figure(anim, anim_data, save_path):
    '''Save the plt figure as a mp4'''
    # Windows problem with FFMpegWriter
    mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

    fps = anim_data['fps']
    n_frames = anim_data['n_frames']
    writer = FFMpegWriter(fps=int(round(fps)), codec='h264', bitrate=1800)
    with Progress() as progress:
        task = progress.add_task("Saving animation", total=n_frames)
        def cb(curr_frame: int, total_frames: int):
            '''Callback function to update progress bar for video saving'''
            progress.update(task, completed=curr_frame)

        anim.save(f'{save_path}.mp4', writer=writer, dpi=150, progress_callback=cb)
