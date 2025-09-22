#!/usr/bin/python3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import array, cos, sin
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, FancyBboxPatch
import imageio_ffmpeg

from simulation import SimulationResult
from dynamics import PhysicalParamters

# Windows problem with FFMpegWriter
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()


def animate_simulation(sim_result: SimulationResult, params: PhysicalParamters,
                       save_path: Path) -> FuncAnimation:
    '''Animate a precomputed SimulationResult'''

    colors = mpl.cm.Paired.colors

    cart_width = 0.4
    cart_height = 0.2

    aspect_ratio = 9.0 / 16.0                                                   # height / width                
    y_lim = (-params.l * 0.4 - cart_height, params.l * 1.1 + cart_height)       # chosen arbitrarily
    x_lim = array([-0.5, 0.5])                                                  # symmetric x-axis
    x_lim *= (y_lim[1] - y_lim[0]) * 1 / aspect_ratio                           # scale to perserve aspect ratio

    fig, ax = plt.subplots()
    ax.set_box_aspect(aspect_ratio)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel('Cart position [m]')
    ax.set_ylabel('Height [m]')
    ax.set_title('Cart-pole animation')
    ax.axhline(0.0, color='black', lw=2)

    # Origin of patch is in bottom left corner, so we need to offset it
    cart_offset = np.array([-cart_width / 2.0, 0.0])
    cart_patch = FancyBboxPatch((0, 0), cart_width, cart_height, 
                                boxstyle='round, pad=0.05', fc=colors[0], ec=colors[1], lw=2) 
    pole_line, = ax.plot([], [], lw=6, color=colors[2], zorder=-1)
    pole_circle = Circle((0, 0), radius=0.05, fc=colors[4], ec=colors[5], lw=2)

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
    anim = FuncAnimation(fig, update, frames=len(sim_result.time_ts),
                         interval=1000.0 / fps, blit=True)

    if save_path:
        writer = FFMpegWriter(fps=int(round(fps)), codec='h264', bitrate=1800)
        anim.save(f'{save_path}.mp4', writer=writer, dpi=150)
        print('Animation saved')
    plt.show()

    return anim

