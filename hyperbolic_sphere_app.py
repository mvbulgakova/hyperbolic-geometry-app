import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# 1. ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ГРАФИКА
def create_sphere_figure(radius_hs, center_x, center_y, center_z, current_camera=None, show_axes=True):
    r = 1.0  # Радиус сферы-абсолюта
    center_hs = np.array([center_x, center_y, center_z])

    fig = go.Figure()

    # Сфера-абсолют (граничная сфера)
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

    # ГИПЕРБОЛИЧЕСКАЯ СФЕРА (ЭЛЛИПСОИД В МОДЕЛИ БЕЛЬТРАМИ-КЛЕЙНА)
    dist_from_origin = np.linalg.norm(center_hs)
    is_sphere = dist_from_origin < 1e-6
    
    if is_sphere:
        # Если центр в начале координат, сфера остается евклидовой сферой
        x_hs = center_hs[0] + radius_hs * np.outer(np.cos(phi_surf), np.sin(theta_surf))
        y_hs = center_hs[1] + radius_hs * np.outer(np.sin(phi_surf), np.sin(theta_surf))
        z_hs = center_hs[2] + radius_hs * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
        R = np.eye(3)
        radius_parallel = radius_perp = radius_hs
    else:
        # Если центр смещен, форма становится эллипсоидом
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

    # Маркер центра
    fig.add_trace(go.Scatter3d(
        x=[center_hs[0]], y=[center_hs[1]], z=[center_hs[2]],
        mode='markers', marker=dict(color='black', size=5, symbol='diamond'), name='Центр',
        hoverinfo='none', showlegend=True
    ))

    # ГЕОДЕЗИЧЕСКИЕ ЛИНИИ
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

        # Находим конечные точки хорды (геодезической), проходящей через center_hs
        a = 1.0
        b = 2 * np.dot(center_hs, unit_dir_vec)
        c = np.dot(center_hs, center_hs) - r**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0: continue

        t_plus = (-b + np.sqrt(discriminant)) / (2*a)
        t_minus = (-b - np.sqrt(discriminant)) / (2*a)
        
        p_end1 = center_hs + t_minus * unit_dir_vec
        p_end2 = center_hs + t_plus * unit_dir_vec
        vec_chord = p_end2 - p_end1
        if np.linalg.norm(vec_chord) < 1e-6: continue

        # Пересечение хорды с эллипсоидом
        # Уравнение: (P_start + t*V_dir - C_e)^T * M_inv * (P_start + t*V_dir - C_e) = 1
        # M_inv = R * D_inv * R_T, где D_inv - диагональная матрица обратных квадратов полуосей
        D_inv = np.diag([1/radius_perp**2, 1/radius_perp**2, 1/radius_parallel**2])
        M_inv = R @ D_inv @ R.T

        oc_to_line_start = p_end1 - center_hs
        
        a_ell = vec_chord.T @ M_inv @ vec_chord
        b_ell = 2 * (vec_chord.T @ M_inv @ oc_to_line_start)
        c_ell = oc_to_line_start.T @ M_inv @ oc_to_line_start - 1

        disc_ell = b_ell**2 - 4 * a_ell * c_ell

        if disc_ell < 0: # Хорда не пересекает эллипсоид
            fig.add_trace(go.Scatter3d(
                x=[p_end1[0], p_end2[0]], y=[p_end1[1], p_end2[1]], z=[p_end1[2], p_end2[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none'
            ))
            continue
        
        # Параметры t находятся в диапазоне [0, 1] вдоль vec_chord от p_end1
        t_entry = (-b_ell - np.sqrt(disc_ell)) / (2*a_ell)
        t_exit = (-b_ell + np.sqrt(disc_ell)) / (2*a_ell)

        point_entry_hs = p_end1 + t_entry * vec_chord
        point_exit_hs = p_end1 + t_exit * vec_chord
        
        # ПОСТРОЕНИЕ СЕГМЕНТОВ ЛИНИЙ
        if t_entry > 1e-6:
             fig.add_trace(go.Scatter3d(
                x=[p_end1[0], point_entry_hs[0]], y=[p_end1[1], point_entry_hs[1]], z=[p_end1[2], point_entry_hs[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none'
            ))
        
        if (t_exit - t_entry) * np.linalg.norm(vec_chord) > 1e-6:
            fig.add_trace(go.Scatter3d(
                x=[point_entry_hs[0], point_exit_hs[0]], y=[point_entry_hs[1], point_exit_hs[1]], z=[point_entry_hs[2], point_exit_hs[2]],
                mode='lines', line=dict(color='#C80000', width=1.5, dash='dash'), showlegend=False, hoverinfo='none'
            ))
        
        if 1 - t_exit > 1e-6:
            fig.add_trace(go.Scatter3d(
                x=[point_exit_hs[0], p_end2[0]], y=[point_exit_hs[1], p_end2[1]], z=[point_exit_hs[2], p_end2[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none'
            ))

    # ДОБАВЛЕНИЕ ТОНКИХ ОСЕЙ (для наглядности изменения положения сферы)
    if show_axes:
        axis_length = r * 1.1
        fig.add_trace(go.Scatter3d(x=[-axis_length, axis_length], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[axis_length * 1.05], y=[0], z=[0], mode='text', text=['X'], textfont=dict(color='red', size=14), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0,0], y=[-axis_length, axis_length], z=[0,0], mode='lines', line=dict(color='blue', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0], y=[axis_length * 1.05], z=[0], mode='text', text=['Y'], textfont=dict(color='blue', size=14), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-axis_length, axis_length], mode='lines', line=dict(color='green', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[axis_length * 1.05], mode='text', text=['Z'], textfont=dict(color='green', size=14), showlegend=False, hoverinfo='none'))

    scene_settings = dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data'
    )
    
    initial_camera_eye = dict(x=1.5, y=1.5, z=1.5) 
    
    if current_camera:
        scene_settings['camera'] = current_camera
    else:
        scene_settings['camera_eye'] = initial_camera_eye

    fig.update_layout(
        title='Интерактивная модель гиперболической сферы',
        scene=scene_settings,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.8, y=0.9),
        font=dict(family="Arial, sans-serif", size=12, color="black")
    )
    return fig

# 2. СОЗДАНИЕ ПРИЛОЖЕНИЯ DASH
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px', 'color': 'black'}, children=[
    html.H1("Интерактивная модель Бельтрами-Клейна", style={'textAlign': 'center'}),
    
    html.Div([
        html.Button('Скрыть/показать оси XYZ', id='toggle-axes-button', n_clicks=0,
                    style={'margin-top': '20px', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'width': 'auto', 'padding': '10px 20px'}),
        dcc.Store(id='axes-visibility-store', data={'visible': True})
    ], style={'width': '80%', 'margin': 'auto', 'text-align': 'center'}),

    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'alignItems': 'flex-start', 'gap': '20px', 'marginTop': '20px'}, children=[
        
        dcc.Graph(id='hyperbolic-sphere-graph', style={'flexGrow': '1', 'height': '70vh', 'width': 'auto'}),

        html.Div(style={'flexShrink': '0', 'width': '300px', 'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '8px'}, children=[
            html.Label("Центр X", style={'margin-top': '0px', 'display': 'block'}),
            dcc.Slider(id='center-x-slider', min=-0.6, max=0.6, step=0.05, value=0.2, marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            
            html.Label("Центр Y", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='center-y-slider', min=-0.6, max=0.6, step=0.05, value=-0.1, marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            
            html.Label("Центр Z", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='center-z-slider', min=-0.6, max=0.6, step=0.05, value=0.3, marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            
            html.Label("Евклидов радиус", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='radius-slider', min=0.05, max=0.8, step=0.05, value=0.4, marks={i/10: str(i/10) for i in range(1, 9)}),
        ])
    ])
])

# 3. ЛОГИКА ОБНОВЛЕНИЯ ГРАФИКА (как и везде)
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
def update_figure(radius, cx, cy, cz, n_clicks, relayoutData, current_axes_visibility_data):
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

# 4. ЗАПУСК ПРИЛОЖЕНИЯ
server = app.server 
if __name__ == '__main__':
    app.run(debug=True)
