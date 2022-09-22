import matplotlib.pyplot as plt
import numpy as np

font = {
    'color': 'black',
    'weight': 'normal',
    'size': 23,
}


# def plot_points(ax, points):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.set_xlim(min_x, max_x)
#     ax.set_ylim(min_y, max_y)
#     ax.scatter([p.x for p in points], [p.y for p in points], marker='.', c='black', s=2, linewidths=0)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     fig.tight_layout()
#     plt.show()


def plot_dual_distribution(v, name=None, key='mean'):
    colors = ['white', 'darkgray', 'black']
    markers = ['o', 'v', 'x']
    data = v['data']
    uniform_data = data['uniform']
    normal_data = data['normal']
    x_tick_labels = v['x_tick_labels']
    x_label = v['x_label']
    y_label = v['y_label']
    if len(x_tick_labels) > 5:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(x_tick_labels))
    blank = np.zeros(len(x_tick_labels))
    ax.plot(x, blank, label='Uniform:', color='white', alpha=0)
    ax.plot(x, blank, label=' Normal:', color='white', alpha=0)
    ax.bar(x - width, [e[key] for e in uniform_data['VR']], width, label='VR', color=colors[0], edgecolor='black')
    ax.bar(x, [e[key] for e in uniform_data['SLICE']], width, label='SLICE', color=colors[1], edgecolor='black')
    ax.bar(x + width, [e[key] for e in uniform_data['CSD']], width, label='CSD', color=colors[2], edgecolor='black')
    ax.plot(x, [e[key] for e in normal_data['VR']], label='VR', marker=markers[0], markersize=12, color='black', markerfacecolor='white')
    ax.plot(x, [e[key] for e in normal_data['SLICE']], label='SLICE', marker=markers[1], markersize=12, color='black',
            markerfacecolor='white')
    ax.plot(x, [e[key] for e in normal_data['CSD']], label='CSD', marker=markers[2], markersize=12, color='black')
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[1], handles[5], handles[2], handles[6], handles[3], handles[7], handles[4]]
    labels = [labels[0], labels[1], labels[5], labels[2], labels[6], labels[3], labels[7], labels[4]]
    ax.legend(handles, labels, loc='center', bbox_to_anchor=(0.35, 1.15), fontsize=20, ncol=4, frameon=False)
    ax.set_ylabel(y_label, fontdict=font)
    ax.set_xlabel(x_label, fontdict=font)
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in x_tick_labels])
    ax.tick_params(labelsize=24)
    fig.tight_layout()
    if name is None:
        plt.show()
    else:
        plt.savefig(name + '.pdf', format='pdf')


def plot_single_distribution(v, name=None, key='mean'):
    colors = ['white', 'darkgray', 'black']
    data = v['data']
    x_tick_labels = v['x_tick_labels']
    x_label = v['x_label']
    y_label = '\n\n' + v['y_label']
    if len(x_tick_labels) > 5:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(x_tick_labels))
    ax.bar(x - width, [e[key] for e in data['VR']], width, label='VR', color=colors[0], edgecolor='black')
    ax.bar(x, [e[key] for e in data['SLICE']], width, label='SLICE', color=colors[1], edgecolor='black')
    ax.bar(x + width, [e[key] for e in data['CSD']], width, label='CSD', color=colors[2], edgecolor='black')
    ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05), fontsize=23, ncol=3, frameon=False)
    ax.set_ylabel(y_label, fontdict=font)
    ax.set_xlabel(x_label, fontdict=font)
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in x_tick_labels])
    ax.tick_params(labelsize=24)
    fig.tight_layout()
    if name is None:
        plt.show()
    else:
        plt.savefig(name + '.pdf', format='pdf')
