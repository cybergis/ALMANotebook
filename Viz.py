from copy import deepcopy
from operator import attrgetter
import numpy as np
from numpy import linalg as LA
import argparse, csv, imageio, json, math, matplotlib, os, random

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
from pylab import cm,imshow,colorbar

font = {'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

###################################################################################################
###################################################################################################
###################################################################################################
################      __  ______        __  ___    __  __           __      #######################
################     / / / /  _/       /  |/  /__ / /_/ /  ___  ___/ /__    #######################
################    / /_/ // /        / /|_/ / -_) __/ _ \/ _ \/ _  (_-<    #######################
################    \____/___/       /_/  /_/\__/\__/_//_/\___/\_,_/___/    #######################
###################################################################################################
###################################################################################################
###################################################################################################
def generate_gif(sim):
    import imageio
    images = []
    for plot in range(sim.sim_length):
        images.append(imageio.imread(os.path.join(sim.output_directory, "Simulation-T{}.png".format(plot))))
    imageio.mimsave(os.path.join(sim.output_directory, 'movie.gif'), images)

def get_ui(sim, components=None, show=True, save=True):
    components = sim.ui_components if components == None else components
    plt.style.use(sim.ui_style)
    cell_color = components[0]
    fig = plt.figure(figsize=(20, 9))
    subplots = []
    ratios = [1.5]
    if len(components) > 7:
        rows = 4
        columns = math.ceil((len(components)-1)/4.0)
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(rows, columns+1, width_ratios=ratios)
    elif len(components) > 5:
        rows = 3
        columns = math.ceil((len(components)-1)/3.0)
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(rows, columns+1, width_ratios=ratios)
    elif len(components) > 1:
        rows = len(components)-1
        columns = math.ceil((len(components)-1)/4.0)
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(rows, columns+1, width_ratios=ratios)
    else:
        columns = 0
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(1, columns+1, width_ratios=ratios)
    subplots.append(plt.subplot(gs[:, 0]))
    # plot the cells
    coloring = []
    if (cell_color == "Amenity"):
        coloring = sim.get_amnenities()
    elif (cell_color == "Market Prices"):
        coloring = sim.get_market_prices()
    elif (cell_color == "Proximity"):
        coloring = sim.get_proximities()
    else: # default to market_prices
        cell_color = "Market Prices"
        coloring = sim.get_market_prices()
    im = imshow(coloring, cmap=cm.coolwarm, vmin=250)
    cbar = colorbar(im)
    subplots[0].set_title(cell_color, fontsize=20)

    if len(components) > 7:
        for i in range(len(components) - 1):
            subplots.append(plt.subplot(gs[i%4,1+math.floor(i/4.0)]))
    elif len(components) > 5:
        for i in range(len(components) - 1):
            subplots.append(plt.subplot(gs[i%3,1+math.floor(i/3.0)]))
    else:
        for i in range(len(components) - 1):
            subplots.append(plt.subplot(gs[i,1]))
    for i in range(1, len(components)):
        if components[i] == "Histo: Agent Amenity":
            histo_agent_amenity(sim,subplots[i])
        elif components[i] == "Histo: Agent Budget":
            histo_agent_budget(sim,subplots[i])
        elif components[i] == "Histo: Agent Proximity":
            histo_agent_proximity(sim,subplots[i])
        elif components[i] == "Histo: Market Price":
            histo_market_price(sim,subplots[i])
        elif components[i] == "Line: Epsilon":
            line_epsilon(sim,subplots[i])
        elif components[i] == "Scatter: Amenity-Market Price":
            scatter_amenity_market_price(sim,subplots[i])
        elif components[i] == "Scatter: Proximity-Market Price":
            scatter_proximity_market_price(sim,subplots[i])
        elif components[i] == "Stackplot: Agent":
            stackplot_agent(sim,subplots[i])
        elif components[i] == "Stackplot: On Market":
            stackplot_on_market(sim,subplots[i])
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(sim.output_directory, "Simulation-T{}.png".format(sim.time_step)))
    fig.clf()
    plt.clf()
    plt.close('all')

def histo_agent_amenity(sim, subplot):
    subplot.hist([agent.get_amenity_norm() for agent in sim.agents])
    subplot.set_title("Histogram of Agent's Amenity Norm")
    subplot.set_xlabel("Amenity Norm")
    subplot.set_ylabel("Number of Agents")

def histo_agent_budget(sim, subplot):
    subplot.hist([agent.get_budget() for agent in sim.agents])
    subplot.set_title("Histogram of Agent's Budget")
    subplot.set_xlabel("Budget")
    subplot.set_ylabel("Number of Agents")

def histo_agent_proximity(sim, subplot):
    subplot.hist([agent.get_proximity_norm() for agent in sim.agents])
    subplot.set_title("Histogram of Agent's Proximity Norm")
    subplot.set_xlabel("Proximity Norm")
    subplot.set_ylabel("Number of Agents")

def histo_market_price(sim, subplot):
    subplot.hist([cell.market_price for cell in sim.cells])
    subplot.set_title("Histogram of Market Price")
    subplot.set_xlabel("Market Price")
    subplot.set_xlim(bottom=0)
    subplot.set_ylabel("Number of Cells")

def line_epsilon(sim, subplot):
    subplot.plot(np.arange(sim.time_step+1), sim.epsilon_over_time, label="Epsilon")
    subplot.legend(loc='upper right')
    subplot.set_title("Epsilon During Simulation")
    subplot.set_xlabel("Time Step")
    subplot.set_xlim([0,sim.sim_length])
    subplot.set_ylabel("Epsilon")
    #subplot.set_ylim([0,sim.num_agents])

def plot_cells(sim, cell_color, components, show=False, save=True):
    plt.style.use(sim.ui_style)
    fig = plt.figure(figsize=(20, 9))
    subplots = []
    ratios = [1.5]

    if len(components) > 7:
        rows = 4
        columns = math.ceil((len(components)-1)/4.0)
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(rows, columns+1, width_ratios=ratios)
    elif len(components) > 5:
        rows = 3
        columns = math.ceil((len(components)-1)/3.0)
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(rows, columns+1, width_ratios=ratios)
    elif len(components) > 1:
        rows = len(components)-1
        columns = math.ceil((len(components)-1)/4.0)
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(rows, columns+1, width_ratios=ratios)
    else:
        columns = 0
        for i in range(columns):
            ratios.append(1)
        gs = gridspec.GridSpec(1, columns+1, width_ratios=ratios)
    subplots.append(plt.subplot(gs[:, 0]))
    # plot the cells
    coloring = []
    if (cell_color == "Amenity"):
        coloring = sim.get_amenities()
    elif (cell_color == "Market Prices"):
        coloring = sim.get_market_prices()
    elif (cell_color == "Proximity"):
        coloring = sim.get_proximities()
    else: # default to market_prices
        cell_color = "Market Prices"
        coloring = sim.get_market_prices()
    im = imshow(coloring, cmap=cm.coolwarm)
    cbar = colorbar(im)
    subplots[0].set_title(cell_color, fontsize=20)
    if len(components) > 7:
        for i in range(len(components) - 1):
            subplots.append(plt.subplot(gs[i%4,1+math.floor(i/4.0)]))
    elif len(components) > 5:
        for i in range(len(components) - 1):
            subplots.append(plt.subplot(gs[i%3,1+math.floor(i/3.0)]))
    else:
        for i in range(len(components) - 1):
            subplots.append(plt.subplot(gs[i,1]))
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(sim.output_directory, "{}-Cells.png".format(cell_color)))
    fig.clf()
    plt.clf()
    plt.close('all')

def scatter_amenity_market_price(sim, subplot):
    amenities = [cell.get_amenity_norm() for cell in sim.cells]
    subplot.scatter(amenities, [cell.market_price for cell in sim.cells])
    subplot.set_title("Scatterplot of Market Price v.s. Amenity")
    subplot.set_xlabel("Amenity Norm")
    _max = max(amenities)
    subplot.set_xlim([-_max/50.0,1.02*max(sim.proximity)])
    subplot.set_ylabel("Market Price")

def scatter_proximity_market_price(sim, subplot):
    to_plot = [(prox, cell.market_price) for prox, cell in zip(sim.proximity,sim.cells) if cell.market_price > 0]
    subplot.scatter([p[0] for p in to_plot], [p[1] for p in to_plot])
    subplot.set_title("Scatterplot of Market Price v.s. Proximity")
    subplot.set_xlabel("Proximity Norm")
    _max = max(sim.proximity)
    subplot.set_xlim([-_max/50.0,1.02*max(sim.proximity)])
    subplot.set_ylabel("Market Price")

def stackplot_agent(sim, subplot):
    subplot.stackplot(np.arange(sim.time_step+1), sim.num_buyers, sim.num_sellers, sim.num_neither, labels=["buyers", "sellers", "neither"])
    subplot.legend(loc='upper right')
    subplot.set_title("Buyers, Sellers, and Non-Participants")
    subplot.set_xlabel("Time Step")
    subplot.set_xlim([0,sim.sim_length])
    subplot.set_ylabel("Number of People")
    subplot.set_ylim([0,sim.num_agents])

def stackplot_on_market(sim, subplot):
    subplot.stackplot(np.arange(sim.time_step+1), sim.num_on_market, np.subtract([sim.num_cells]*(sim.time_step+1),sim.num_on_market), labels=["on market", "not on market"])
    subplot.legend(loc='upper right')
    subplot.set_title("Proportion of Cells on Market")
    subplot.set_xlabel("Time Step")
    subplot.set_xlim([0,sim.sim_length])
    subplot.set_ylabel("Number of Cells")
    subplot.set_ylim([0,sim.num_cells])
