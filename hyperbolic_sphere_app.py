import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# ==============================================================================
# 1. ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ГРАФИКА
# ==============================================================================
def create_sphere_figure(radius_hs, center_x, center_y, center_z, current_camera=None, show_axes=True):
    r = 1.0  # Радиус сферы-абсолюта (радиус модели Клейна)
    center_hs = np.array([center_x, center_y, center_z])

    fig = go.Figure()

    # Проверка, чтобы сфера не выходила за пределы (визуальная)
    dist_from_origin_to_center = np.linalg.norm(center_hs)
    if dist_from_origin_to_center + radius_hs >= r - 0.005:
        radius_hs = r - dist_from_origin_to_center - 0.01
        if radius_hs < 0.01:
            radius_hs = 0.01

    # Сфера-абсолют
    phi_surf = np.linspace(0, 2*np.pi, 50)
    theta_surf = np.linspace(0, np.pi, 50)
    x_abs = r * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_abs = r * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_abs = r * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    fig.add_trace(go.Surface(
        x=x_abs, y=y_abs, z=z_abs,
        colorscale='Blues', opacity=0.15, showscale=False, name='Абсолют',
        hoverinfo='none'
    ))

    # Гиперболическая сфера - ИСПРАВЛЕНИЕ ЗДЕСЬ
    x_hs = center_hs[0] + radius_hs * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_hs = center_hs[1] + radius_hs * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_hs = center_hs[2] + radius_hs * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
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

    # Радиальные линии
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

        intersect_point = center_hs + radius_hs * unit_dir_vec
        
        a = 1.0
        b = 2 * np.dot(center_hs, unit_dir_vec)
        c = np.dot(center_hs, center_hs) - r**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            continue
            
        t_plus = (-b + np.sqrt(discriminant)) / (2*a)
        
        absolute_point = center_hs + t_plus * unit_dir_vec

        fig.add_trace(go.Scatter3d(
            x=[center_hs[0], intersect_point[0]],
            y=[center_hs[1], intersect_point[1]],
            z=[center_hs[2], intersect_point[2]],
            mode='lines', line=dict(color='red', width=4),
            showlegend=False, hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=[intersect_point[0], absolute_point[0]],
            y=[intersect_point[1], absolute_point[1]],
            z=[intersect_point[2], absolute_point[2]],
            mode='lines', line=dict(color='red', width=2, dash='dash'),
            showlegend=False, hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=[intersect_point[0]], y=[intersect_point[1]], z=[intersect_point[2]],
            mode='markers', marker=dict(color='white', size=3, line=dict(color='black', width=1)),
            showlegend=False, hoverinfo='none'
        ))

    # ДОБАВЛЕНИЕ ТОНКИХ ОСЕЙ
    if show_axes:
        axis_length = r * 1.1

        # Ось X (красная) - вправо
        fig.add_trace(go.Scatter3d(x=[-axis_length, axis_length], y=[0,0], z=[0,0], mode='lines',
                                   line=dict(color='red', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[axis_length * 1.05], y=[0], z=[0], mode='text', text=['X'],
                                   textfont=dict(color='red', size=14), showlegend=False, hoverinfo='none'))

        # Ось Y (синяя) - вперед/влево (стандартное положение)
        fig.add_trace(go.Scatter3d(x=[0,0], y=[-axis_length, axis_length], z=[0,0], mode='lines',
                                   line=dict(color='blue', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0], y=[axis_length * 1.05], z=[0], mode='text', text=['Y'],
                                   textfont=dict(color='blue', size=14), showlegend=False, hoverinfo='none'))

        # Ось Z (зеленая) - вверх
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-axis_length, axis_length], mode='lines',
                                   line=dict(color='green', width=2), showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[axis_length * 1.05], mode='text', text=['Z'],
                                   textfont=dict(color='green', size=14), showlegend=False, hoverinfo='none'))

    # Настройки вида и камеры для ориентации осей
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
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        )
    )
    return fig

# ==============================================================================
# 2. СОЗДАНИЕ ПРИЛОЖЕНИЯ DASH
# ==============================================================================
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px', 'color': 'black'}, children=[
    html.H1("Интерактивная модель Бельтрами-Клейна", style={'textAlign': 'center'}),
    
    # КОНТЕЙНЕР ДЛЯ КНОПКИ
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'margin-bottom': '20px'}, children=[ # Добавил margin-bottom
        html.Button('Скрыть/Показать оси XYZ', id='toggle-axes-button', n_clicks=0,
                    style={'width': 'auto', 'padding': '10px 20px'}),
        dcc.Store(id='axes-visibility-store', data={'visible': True})
    ]),

    # КОНТЕЙНЕР FLEXBOX ДЛЯ ГРАФИКА И УПРАВЛЕНИЯ
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'alignItems': 'flex-start', 'gap': '20px'}, children=[
        
        # Контейнер для графика (слева)
        dcc.Graph(id='hyperbolic-sphere-graph', style={'flexGrow': '1', 'height': '70vh', 'width': 'auto'}),

        # Контейнер для всех элементов управления (справа)
        html.Div(style={'flexShrink': '0', 'width': '300px', 'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '8px'}, children=[
            # СЛАЙДЕРЫ ДЛЯ КООРДИНАТ ЦЕНТРА (без кнопки здесь)
            html.Label("Центр X", style={'margin-top': '0px', 'display': 'block'}), # Убрал margin-top для первого
            dcc.Slider(id='center-x-slider', min=-0.7, max=0.7, step=0.05, value=0.2, marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            
            html.Label("Центр Y", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='center-y-slider', min=-0.7, max=0.7, step=0.05, value=-0.1, marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            
            html.Label("Центр Z", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='center-z-slider', min=-0.7, max=0.7, step=0.05, value=0.3, marks={i/10: str(i/10) for i in range(-6, 7, 2)}),
            
            # СЛАЙДЕР РАЗМЕРА СФЕРЫ
            html.Label("Радиус сферы", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='radius-slider', min=0.1, max=0.8, step=0.05, value=0.4, marks={i/10: str(i/10) for i in range(1, 9)}),

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
def update_figure(radius, cx, cy, cz, n_clicks, relayoutData, current_axes_visibility_data):
    current_camera = None
    if relayoutData and 'scene.camera' in relayoutData:
        current_camera = relayoutData['scene.camera']

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'no_trigger'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_axes_visibility_data = current_axes_visibility_data
    if button_id == 'toggle-axes-button':
        new_visibility_state = not current_axes_visibility_data['visible']
        new_axes_visibility_data = {'visible': new_visibility_state}
    
    fig = create_sphere_figure(
        radius, cx, cy, cz,
        current_camera=current_camera,
        show_axes=new_axes_visibility_data['visible']
    )
    
    return fig, new_axes_visibility_data

# Добавлено для развертывания
server = app.server
