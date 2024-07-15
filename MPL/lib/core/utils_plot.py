import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_3d_points(fig, to_plot):
    body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='b')
    for edge in body_edges:
        ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='g')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-5, 1)
    ax.set_zlim(-3, 3)
    
def plot_3d_points_plotly(to_plot):
    body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
    df_joints = pd.DataFrame(to_plot, columns=['x', 'y', 'z'])
    data = to_plot[body_edges].transpose(0,2,1)
    df_connections = pd.DataFrame([], columns=['x', 'y', 'z'])
    df_connections['x'] = [d[0] for d in data]
    df_connections['y'] = [d[1] for d in data]
    df_connections['z'] = [d[2] for d in data]
    
    fig = px.scatter_3d(df_joints, x='x', y='y', z='z') #, color=df_joints.index)
    # fig2 = px.line_3d(df_connections, x='x', y='y', z='z') #, color=df_connections.index)
    # fig = go.Figure(fig.data + fig2.data)
    # fig.add_trace(fig2.data[0])
    # return fig
    
    # fig = px.scatter_3d(x=to_plot[:,0], y=to_plot[:,1], z=to_plot[:,2])
    figs = [fig]
    for edge in body_edges:
        figs.append(px.line_3d(x=to_plot[edge,0], y=to_plot[edge,1], z=to_plot[edge,2])) #, color='red')
    data = fig.data
    for f in figs[1:]:
        data += f.data
    fig = go.Figure(data=data)
    # px.line_3d(x=to_plot[body_edges,0], y=to_plot[body_edges,1], z=to_plot[body_edges,2], color='red')
    return fig

    
def plot_2d_points(fig,
                   to_plot,
                   to_plot_conf,
                   to_plot_org=None,):
    body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
    # for i, input_ in enumerate(joints):
    #     sub = int('23{}'.format(i+1))
    ax = fig.add_subplot(111)
    ax.scatter(to_plot[:,0], to_plot[:,1], s=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    for edge in body_edges:
        ax.plot(to_plot[edge,0], to_plot[edge,1], color='r')
    if to_plot_org is not None:
        ax.scatter(to_plot_org[:,0], to_plot_org[:,1], s=3, color='g')
        for edge in body_edges:
            ax.plot(to_plot_org[edge,0], to_plot_org[edge,1], color='g')
    
    for i, conf in enumerate(to_plot_conf):
        plt.text(to_plot[i, 0], to_plot[i, 1], '{:.2f}'.format(conf), color='black')