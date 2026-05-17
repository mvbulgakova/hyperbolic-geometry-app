import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


# ==============================================================================
# ВКЛАДКА 1: ГИПЕРБОЛИЧЕСКАЯ СФЕРА
# ==============================================================================

def create_sphere_figure(radius_hs, center_x, center_y, center_z, current_camera=None, show_axes=True):
    r = 1.0
    center_hs = np.array([center_x, center_y, center_z])

    fig = go.Figure()

    phi_surf = np.linspace(0, 2 * np.pi, 50)
    theta_surf = np.linspace(0, np.pi, 50)
    x_abs = r * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_abs = r * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_abs = r * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    fig.add_trace(go.Surface(
        x=x_abs, y=y_abs, z=z_abs,
        colorscale='Blues', opacity=0.15, showscale=False, name='Абсолют',
        hoverinfo='none'
    ))

    dist_from_origin = np.linalg.norm(center_hs)
    is_sphere = dist_from_origin < 1e-6

    if is_sphere:
        x_hs = center_hs[0] + radius_hs * np.outer(np.cos(phi_surf), np.sin(theta_surf))
        y_hs = center_hs[1] + radius_hs * np.outer(np.sin(phi_surf), np.sin(theta_surf))
        z_hs = center_hs[2] + radius_hs * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
        R = np.eye(3)
        radius_parallel = radius_perp = radius_hs
    else:
        squash_factor = np.sqrt(max(1.0 - dist_from_origin**2, 1e-9))
        radius_parallel = radius_hs * squash_factor
        radius_perp = radius_hs

        x_unit_sphere = np.outer(np.cos(phi_surf), np.sin(theta_surf))
        y_unit_sphere = np.outer(np.sin(phi_surf), np.sin(theta_surf))
        z_unit_sphere = np.outer(np.ones_like(phi_surf), np.cos(theta_surf))

        x_ell_std = radius_perp * x_unit_sphere
        y_ell_std = radius_perp * y_unit_sphere
        z_ell_std = radius_parallel * z_unit_sphere

        u_z = np.array([0., 0., 1.])
        u_z_prime = center_hs / dist_from_origin
        v = np.cross(u_z, u_z_prime)
        s = np.linalg.norm(v)
        c = np.dot(u_z, u_z_prime)

        if s < 1e-9:
            R = np.sign(c) * np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

        coords = np.vstack([x_ell_std.ravel(), y_ell_std.ravel(), z_ell_std.ravel()])
        rotated_coords = R @ coords

        x_hs = rotated_coords[0, :].reshape(x_ell_std.shape) + center_hs[0]
        y_hs = rotated_coords[1, :].reshape(y_ell_std.shape) + center_hs[1]
        z_hs = rotated_coords[2, :].reshape(z_ell_std.shape) + center_hs[2]

    fig.add_trace(go.Surface(
        x=x_hs, y=y_hs, z=z_hs,
        colorscale='Greens', opacity=0.6, showscale=False, name='Гиперболическая сфера',
        hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=[center_hs[0]], y=[center_hs[1]], z=[center_hs[2]],
        mode='markers', marker=dict(color='black', size=5, symbol='diamond'), name='Центр',
        hoverinfo='none', showlegend=True
    ))

    num_lines = 50
    indices = np.arange(0, num_lines, dtype=float) + 0.5
    phi_dirs = np.arccos(1 - 2 * indices / num_lines)
    theta_dirs = np.pi * (1 + 5**0.5) * indices

    for i in range(num_lines):
        unit_dir_vec = np.array([
            np.cos(theta_dirs[i]) * np.sin(phi_dirs[i]),
            np.sin(theta_dirs[i]) * np.sin(phi_dirs[i]),
            np.cos(phi_dirs[i])
        ])

        a = 1.0
        b = 2 * np.dot(center_hs, unit_dir_vec)
        c = np.dot(center_hs, center_hs) - r**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            continue

        t_plus = (-b + np.sqrt(discriminant)) / (2*a)
        t_minus = (-b - np.sqrt(discriminant)) / (2*a)
        p_end1 = center_hs + t_minus * unit_dir_vec
        p_end2 = center_hs + t_plus * unit_dir_vec
        vec_chord = p_end2 - p_end1
        if np.linalg.norm(vec_chord) < 1e-6:
            continue

        D_inv = np.diag([1/radius_perp**2, 1/radius_perp**2, 1/radius_parallel**2])
        M_inv = R @ D_inv @ R.T
        oc = p_end1 - center_hs

        a_ell = vec_chord.T @ M_inv @ vec_chord
        b_ell = 2 * (vec_chord.T @ M_inv @ oc)
        c_ell = oc.T @ M_inv @ oc - 1
        disc_ell = b_ell**2 - 4 * a_ell * c_ell

        if disc_ell < 0:
            fig.add_trace(go.Scatter3d(
                x=[p_end1[0], p_end2[0]], y=[p_end1[1], p_end2[1]], z=[p_end1[2], p_end2[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none'
            ))
            continue

        t_entry = (-b_ell - np.sqrt(disc_ell)) / (2*a_ell)
        t_exit = (-b_ell + np.sqrt(disc_ell)) / (2*a_ell)
        p_entry = p_end1 + t_entry * vec_chord
        p_exit = p_end1 + t_exit * vec_chord

        if t_entry > 1e-6:
            fig.add_trace(go.Scatter3d(
                x=[p_end1[0], p_entry[0]], y=[p_end1[1], p_entry[1]], z=[p_end1[2], p_entry[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none'
            ))
        if (t_exit - t_entry) * np.linalg.norm(vec_chord) > 1e-6:
            fig.add_trace(go.Scatter3d(
                x=[p_entry[0], p_exit[0]], y=[p_entry[1], p_exit[1]], z=[p_entry[2], p_exit[2]],
                mode='lines', line=dict(color='#C80000', width=1.5, dash='dash'), showlegend=False, hoverinfo='none'
            ))
        if 1 - t_exit > 1e-6:
            fig.add_trace(go.Scatter3d(
                x=[p_exit[0], p_end2[0]], y=[p_exit[1], p_end2[1]], z=[p_exit[2], p_end2[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none'
            ))

    if show_axes:
        axis_length = r * 1.1
        fig.add_trace(go.Scatter3d(x=[-axis_length, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[axis_length * 1.05], y=[0], z=[0], mode='text', text=['X'], textfont=dict(color='red', size=14), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[-axis_length, axis_length], z=[0, 0], mode='lines', line=dict(color='blue', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0], y=[axis_length * 1.05], z=[0], mode='text', text=['Y'], textfont=dict(color='blue', size=14), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-axis_length, axis_length], mode='lines', line=dict(color='green', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[axis_length * 1.05], mode='text', text=['Z'], textfont=dict(color='green', size=14), showlegend=False, hoverinfo='none'))

    scene_settings = dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data'
    )
    if current_camera:
        scene_settings['camera'] = current_camera
    else:
        scene_settings['camera'] = dict(eye=dict(x=1.5, y=1.5, z=1.5))

    fig.update_layout(
        title='Гиперболическая сфера в модели Бельтрами-Клейна',
        scene=scene_settings,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.8, y=0.9),
        font=dict(family="Arial, sans-serif", size=12, color="black")
    )
    return fig


# ==============================================================================
# ВКЛАДКА 2: ОРАСФЕРА (ХОРОСФЕРА)
# ==============================================================================

def create_orosphere_figure(phi, theta, r_horo, show_guiding_lines=True, current_camera=None):
    r = 1.0

    # Идеальная точка (точка на абсолюте)
    omega = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

    # Евклидов центр орасферы: (1 - r_horo) * omega
    center_horo = (1.0 - r_horo) * omega

    fig = go.Figure()

    phi_surf = np.linspace(0, 2 * np.pi, 50)
    theta_surf = np.linspace(0, np.pi, 50)

    # Абсолют
    x_abs = r * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_abs = r * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_abs = r * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    fig.add_trace(go.Surface(
        x=x_abs, y=y_abs, z=z_abs,
        colorscale='Blues', opacity=0.15, showscale=False, name='Абсолют',
        hoverinfo='none'
    ))

    # Орасфера: евклидова сфера радиуса r_horo, касающаяся абсолюта в точке omega
    x_hs = center_horo[0] + r_horo * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_hs = center_horo[1] + r_horo * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_hs = center_horo[2] + r_horo * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    fig.add_trace(go.Surface(
        x=x_hs, y=y_hs, z=z_hs,
        colorscale='Oranges', opacity=0.6, showscale=False, name='Орасфера',
        hoverinfo='none'
    ))

    # Маркер идеальной точки omega на абсолюте
    fig.add_trace(go.Scatter3d(
        x=[omega[0]], y=[omega[1]], z=[omega[2]],
        mode='markers', marker=dict(color='purple', size=7, symbol='diamond'),
        name='Идеальная точка ω', hoverinfo='none'
    ))

    # Направляющие геодезические: хорды, сходящиеся к omega
    # Все геодезические, перпендикулярные орасфере, заканчиваются в omega
    if show_guiding_lines:
        num_lines = 30
        indices = np.arange(0, num_lines, dtype=float) + 0.5
        phi_dirs = np.arccos(1 - 2 * indices / num_lines)
        theta_dirs = np.pi * (1 + 5**0.5) * indices

        for i in range(num_lines):
            p_other = np.array([
                np.cos(theta_dirs[i]) * np.sin(phi_dirs[i]),
                np.sin(theta_dirs[i]) * np.sin(phi_dirs[i]),
                np.cos(phi_dirs[i])
            ])
            # Пропускаем точки, слишком близкие к omega
            if np.linalg.norm(p_other - omega) < 0.2:
                continue
            fig.add_trace(go.Scatter3d(
                x=[omega[0], p_other[0]],
                y=[omega[1], p_other[1]],
                z=[omega[2], p_other[2]],
                mode='lines',
                line=dict(color='#8B4513', width=1.5),
                showlegend=False, hoverinfo='none'
            ))

    scene_settings = dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data'
    )
    if current_camera:
        scene_settings['camera'] = current_camera
    else:
        scene_settings['camera'] = dict(eye=dict(x=1.5, y=1.5, z=1.5))

    fig.update_layout(
        title='Орасфера в модели Бельтрами-Клейна',
        scene=scene_settings,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.8, y=0.9),
        font=dict(family="Arial, sans-serif", size=12, color="black")
    )
    return fig


# ==============================================================================
# ПРИЛОЖЕНИЕ DASH
# ==============================================================================

app = dash.Dash(__name__)

SLIDER_STYLE = {'margin-top': '10px', 'display': 'block'}
PANEL_STYLE = {
    'flexShrink': '0', 'width': '300px', 'padding': '20px',
    'border': '1px solid #ccc', 'borderRadius': '8px'
}
GRAPH_STYLE = {'flexGrow': '1', 'height': '70vh', 'width': 'auto'}
ROW_STYLE = {
    'display': 'flex', 'flexDirection': 'row',
    'justifyContent': 'center', 'alignItems': 'flex-start',
    'gap': '20px', 'marginTop': '20px'
}

tab_sphere = dcc.Tab(label='Гиперболическая сфера', value='sphere', children=[
    html.Div([
        html.Button(
            'Скрыть/показать оси XYZ', id='toggle-axes-button', n_clicks=0,
            style={'marginTop': '16px', 'display': 'block', 'margin': '16px auto 0', 'padding': '10px 20px'}
        ),
        dcc.Store(id='axes-visibility-store', data={'visible': True}),
    ], style={'textAlign': 'center'}),
    html.Div(style=ROW_STYLE, children=[
        dcc.Graph(id='hyperbolic-sphere-graph', style=GRAPH_STYLE),
        html.Div(style=PANEL_STYLE, children=[
            html.Label("Центр X", style={'marginTop': '0', 'display': 'block'}),
            dcc.Slider(id='center-x-slider', min=-0.6, max=0.6, step=0.05, value=0.2,
                       marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            html.Label("Центр Y", style=SLIDER_STYLE),
            dcc.Slider(id='center-y-slider', min=-0.6, max=0.6, step=0.05, value=-0.1,
                       marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            html.Label("Центр Z", style=SLIDER_STYLE),
            dcc.Slider(id='center-z-slider', min=-0.6, max=0.6, step=0.05, value=0.3,
                       marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            html.Label("Евклидов радиус", style=SLIDER_STYLE),
            dcc.Slider(id='radius-slider', min=0.05, max=0.8, step=0.05, value=0.4,
                       marks={i/10: str(i/10) for i in range(1, 9)}),
        ])
    ])
])

tab_orosphere = dcc.Tab(label='Орасфера', value='orosphere', children=[
    html.Div([
        html.Button(
            'Скрыть/показать геодезические', id='toggle-guiding-lines-button', n_clicks=0,
            style={'marginTop': '16px', 'display': 'block', 'margin': '16px auto 0', 'padding': '10px 20px'}
        ),
        dcc.Store(id='guiding-lines-visibility-store', data={'visible': True}),
    ], style={'textAlign': 'center'}),
    html.Div(style=ROW_STYLE, children=[
        dcc.Graph(id='hyperbolic-orosphere-graph', style=GRAPH_STYLE),
        html.Div(style=PANEL_STYLE, children=[
            html.Label("Угол φ (полярный)", style={'marginTop': '0', 'display': 'block'}),
            dcc.Slider(id='phi-slider', min=0.05, max=np.pi - 0.05, step=0.05, value=np.pi / 4,
                       marks={round(v, 2): str(round(v, 2)) for v in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]}),
            html.Label("Угол θ (азимутальный)", style=SLIDER_STYLE),
            dcc.Slider(id='theta-slider', min=0, max=2 * np.pi, step=0.1, value=np.pi / 4,
                       marks={round(v, 2): str(round(v, 2)) for v in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]}),
            html.Label("Радиус орасферы", style=SLIDER_STYLE),
            dcc.Slider(id='r-horo-slider', min=0.05, max=0.9, step=0.05, value=0.4,
                       marks={i/10: str(i/10) for i in range(1, 10, 2)}),
        ])
    ])
])

app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px', 'color': 'black'},
    children=[
        html.H1("Модель Бельтрами-Клейна", style={'textAlign': 'center', 'marginBottom': '0'}),
        dcc.Tabs(id='main-tabs', value='sphere', children=[tab_sphere, tab_orosphere]),
    ]
)


# ==============================================================================
# CALLBACKS
# ==============================================================================

@app.callback(
    Output('hyperbolic-sphere-graph', 'figure'),
    Output('axes-visibility-store', 'data'),
    [Input('radius-slider', 'value'),
     Input('center-x-slider', 'value'),
     Input('center-y-slider', 'value'),
     Input('center-z-slider', 'value'),
     Input('toggle-axes-button', 'n_clicks')],
    [State('hyperbolic-sphere-graph', 'relayoutData'),
     State('axes-visibility-store', 'data')]
)
def update_sphere(radius, cx, cy, cz, n_clicks, relayoutData, axes_data):
    current_camera = None
    if relayoutData and 'scene.camera' in relayoutData:
        current_camera = relayoutData['scene.camera']

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    if triggered_id == 'toggle-axes-button':
        axes_data = {'visible': not axes_data['visible']}

    fig = create_sphere_figure(radius, cx, cy, cz, current_camera=current_camera, show_axes=axes_data['visible'])
    return fig, axes_data


@app.callback(
    Output('hyperbolic-orosphere-graph', 'figure'),
    Output('guiding-lines-visibility-store', 'data'),
    [Input('phi-slider', 'value'),
     Input('theta-slider', 'value'),
     Input('r-horo-slider', 'value'),
     Input('toggle-guiding-lines-button', 'n_clicks')],
    [State('hyperbolic-orosphere-graph', 'relayoutData'),
     State('guiding-lines-visibility-store', 'data')]
)
def update_orosphere(phi, theta, r_horo, n_clicks, relayoutData, lines_data):
    current_camera = None
    if relayoutData and 'scene.camera' in relayoutData:
        current_camera = relayoutData['scene.camera']

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    if triggered_id == 'toggle-guiding-lines-button':
        lines_data = {'visible': not lines_data['visible']}

    fig = create_orosphere_figure(phi, theta, r_horo, show_guiding_lines=lines_data['visible'], current_camera=current_camera)
    return fig, lines_data


server = app.server
if __name__ == '__main__':
    app.run(debug=True)
