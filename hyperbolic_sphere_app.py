import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


# ==============================================================================
# 1. ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ГРАФИКА
# ==============================================================================
def create_sphere_figure(radius_hs, center_x, center_y, center_z, current_camera=None, show_axes=True):
    # Тут ваш код для создания фигуры, без изменений
    r = 1.0
    center_hs = np.array([center_x, center_y, center_z])
    fig = go.Figure()
    dist_from_origin_to_center = np.linalg.norm(center_hs)
    max_allowed_radius = r - dist_from_origin_to_center - 0.005
    if radius_hs >= max_allowed_radius:
        radius_hs = max_allowed_radius
    if radius_hs < 0.01:
        radius_hs = 0.01

    phi_surf = np.linspace(0, 2 * np.pi, 50)
    theta_surf = np.linspace(0, np.pi, 50)
    x_abs = r * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_abs = r * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_abs = r * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    fig.add_trace(go.Surface(
        x=x_abs, y=y_abs, z=z_abs,
        colorscale='Blues', opacity=0.15, showscale=False, name='Абсолют'
    ))
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_sphere = center_hs[0] + radius_hs * np.cos(u) * np.sin(v)
    y_sphere = center_hs[1] + radius_hs * np.sin(u) * np.sin(v)
    z_sphere = center_hs[2] + radius_hs * np.cos(v)
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Reds', opacity=0.8, showscale=False, name='Гиперболическая сфера'
    ))
    if show_axes:
        num_lines = 10
        for i in range(num_lines):
            phi = 2 * np.pi * i / num_lines
            t = np.linspace(-3, 3, 2)
            lx = center_hs[0] + t * np.cos(phi)
            ly = center_hs[1] + t * np.sin(phi)
            lz = center_hs[2] + t * 0
            fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=2), showlegend=False))

    fig.update_layout(
        title='Сфера в модели Клейна-Бельтрами',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='cube', camera=current_camera
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# ==============================================================================
# 2. СХЕМА СТРАНИЦЫ (LAYOUT)
# ==============================================================================
layout = html.Div([
    html.H3('Интерактивная модель: Сфера'),
    html.Div([
        dcc.Graph(id='hyperbolic-sphere-graph', style={'height': '80vh', 'width': '70%'}),
        html.Div(style={'width': '25%', 'padding': '20px'}, children=[
            dcc.Store(id='axes-visibility-store', data={'visible': True}),
            html.Button('Переключить оси', id='toggle-axes-button', n_clicks=0),
            html.Br(), html.Br(),
            html.Label("Радиус сферы"),
            dcc.Slider(id='radius-slider', min=0.01, max=0.9, step=0.01, value=0.3),
            html.Label("Центр X"),
            dcc.Slider(id='center-x-slider', min=-0.8, max=0.8, step=0.05, value=0.0),
            html.Label("Центр Y"),
            dcc.Slider(id='center-y-slider', min=-0.8, max=0.8, step=0.05, value=0.0),
            html.Label("Центр Z"),
            dcc.Slider(id='center-z-slider', min=-0.8, max=0.8, step=0.05, value=0.0),
        ])
    ])
])

# ==============================================================================
# 3. ЛОГИКА ОБНОВЛЕНИЯ ГРАФИКА
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
def update_sphere_figure(radius, cx, cy, cz, n_clicks, relayoutData, current_axes_visibility_data):
    # Тут ваш код для callback, без изменений
    current_camera = None
    if relayoutData and 'scene.camera' in relayoutData:
        current_camera = relayoutData['scene.camera']

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'no_trigger'

    new_axes_visibility_data = current_axes_visibility_data
    if triggered_id == 'toggle-axes-button':
        new_visibility_state = not current_axes_visibility_data['visible']
        new_axes_visibility_data = {'visible': new_visibility_state}
    
    fig = create_sphere_figure(
        radius, cx, cy, cz,
        current_camera=current_camera,
        show_axes=new_axes_visibility_data['visible']
    )
    return fig, new_axes_visibility_data

server = app.server
