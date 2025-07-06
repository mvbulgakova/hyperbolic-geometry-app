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
    center_hs = np.array([center_x, center_y, center_z]) # Центр гиперболической сферы (евклидовый)

    fig = go.Figure()

    # Проверка, чтобы сфера не выходила за пределы (визуальная)
    dist_from_origin_to_center = np.linalg.norm(center_hs)
    # Максимально допустимый радиус, чтобы сфера не пересекала Абсолют
    max_allowed_radius = r - dist_from_origin_to_center - 0.005 
    if radius_hs >= max_allowed_radius:
        radius_hs = max_allowed_radius
        if radius_hs < 0.01: # Минимальный радиус, чтобы избежать вырождения
            radius_hs = 0.01

    # Сфера-абсолют (граничная сфера модели Клейна)
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

    # Гиперболическая сфера (евклидова сфера внутри модели)
    x_hs = center_hs[0] + radius_hs * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_hs = center_hs[1] + radius_hs * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_hs = center_hs[2] + radius_hs * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    fig.add_trace(go.Surface(
        x=x_hs, y=y_hs, z=z_hs,
        colorscale='Greens', opacity=0.6, showscale=False, name='Гиперболическая сфера',
        hoverinfo='none'
    ))

    # Маркер центра гиперболической сферы
    fig.add_trace(go.Scatter3d(
        x=[center_hs[0]], y=[center_hs[1]], z=[center_hs[2]],
        mode='markers', marker=dict(color='black', size=5, symbol='diamond'), name='Центр сферы',
        hoverinfo='none', showlegend=True
    ))

    # Радиальные линии (геодезические, проходящие через центр сферы)
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

        # Начало линии - центр гиперболической сферы
        p_start = center_hs
        # Конечная точка линии - на Абсолюте
        # Находим точку на Абсолюте в направлении unit_dir_vec от центра сферы.
        # Это решается для t, где (center_hs + t * unit_dir_vec)^2 = r^2
        # (center_hs . center_hs) + 2t (center_hs . unit_dir_vec) + t^2 (unit_dir_vec . unit_dir_vec) = r^2
        
        a = 1.0
        b = 2 * np.dot(center_hs, unit_dir_vec)
        c = np.dot(center_hs, center_hs) - r**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            continue # Нет пересечения с Абсолютом, линия не достигает границы.

        # t_plus - это параметр для точки на Абсолюте
        t_plus = (-b + np.sqrt(discriminant)) / (2*a)
        # t_minus - это параметр для другой точки на Абсолюте (в противоположном направлении)
        t_minus = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Определяем конечные точки хорды (гиперболической прямой)
        # Она проходит через центр hs от одной границы до другой.
        p_end1 = p_start + t_minus * unit_dir_vec
        p_end2 = p_start + t_plus * unit_dir_vec

        # Теперь нам нужно найти точки пересечения этой хорды [p_end1, p_end2]
        # с зеленой гиперболической сферой (которая является евклидовой сферой с центром center_hs и радиусом radius_hs).
        # Hормализованный вектор хорды от p_end1 к p_end2
        vec_chord = p_end2 - p_end1
        if np.linalg.norm(vec_chord) < 1e-6:
            continue # Избежать деления на ноль

        # Пересчитываем параметры t для пересечения хорды с ЗЕЛЕНОЙ СФЕРОЙ
        # Уравнение: (P_end1 + t_chord * vec_chord - center_hs)^2 = radius_hs^2
        oc_to_line_start = p_end1 - center_hs
        a_chord = np.dot(oc_to_line_start, oc_to_line_start)
        b_chord = 2 * np.dot(oc_to_line_start, vec_chord)
        c_chord = np.dot(vec_chord, vec_chord) - radius_hs**2 # Ошибка здесь, было `np.dot(vec_chord, vec_chord)` вместо `a_chord`

        # Corrected quadratic equation for intersection of ray from p_end1 along vec_chord with sphere center_hs and radius_hs
        # ( (p_end1 + t * vec_chord) - center_hs ) . ( (p_end1 + t * vec_chord) - center_hs ) = radius_hs^2
        # Let O_prime = p_end1 - center_hs
        # (O_prime + t * vec_chord) . (O_prime + t * vec_chord) = radius_hs^2
        # O_prime.O_prime + 2*t*(O_prime.vec_chord) + t^2*(vec_chord.vec_chord) = radius_hs^2
        # (vec_chord.vec_chord)*t^2 + (2*O_prime.vec_chord)*t + (O_prime.O_prime - radius_hs^2) = 0
        a_sphere_intersect = np.dot(vec_chord, vec_chord)
        b_sphere_intersect = 2 * np.dot(oc_to_line_start, vec_chord)
        c_sphere_intersect = np.dot(oc_to_line_start, oc_to_line_start) - radius_hs**2


        disc_chord = b_sphere_intersect**2 - 4*a_sphere_intersect*c_sphere_intersect

        if disc_chord < 0: # Хорда не пересекает зеленую сферу
            fig.add_trace(go.Scatter3d(
                x=[p_end1[0], p_end2[0]], y=[p_end1[1], p_end2[1]], z=[p_end1[2], p_end2[2]],
                mode='lines', line=dict(color='#C80000', width=1.5), showlegend=False, hoverinfo='none' # Изменена толщина и цвет
            ))
            continue

        t_entry_hs = (-b_sphere_intersect - np.sqrt(disc_chord)) / (2*a_sphere_intersect)
        t_exit_hs = (-b_sphere_intersect + np.sqrt(disc_chord)) / (2*a_sphere_intersect)

        # Точки пересечения хорды с зеленой сферой
        point_entry_hs = p_end1 + t_entry_hs * vec_chord
        point_exit_hs = p_end1 + t_exit_hs * vec_chord
        
        # --- ПОСТРОЕНИЕ СЕГМЕНТОВ ЛИНИЙ ---
        # 1. От p_end1 (на Абсолюте) до point_entry_hs (сплошная)
        # Убедимся, что segment не пустой и находится в пределах хорды [0,1]
        if t_entry_hs > 1e-6: # Убедимся, что начальная точка не прямо на сфере
            fig.add_trace(go.Scatter3d(
                x=[p_end1[0], point_entry_hs[0]],
                y=[p_end1[1], point_entry_hs[1]],
                z=[p_end1[2], point_entry_hs[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none' # Изменена толщина и цвет
            ))
        
        # 2. От point_entry_hs до point_exit_hs (ПУНКТИРНАЯ - ВНУТРИ ГИПЕРБОЛИЧЕСКОЙ СФЕРЫ)
        if t_exit_hs - t_entry_hs > 1e-6: # Убедимся, что сегмент имеет ненулевую длину
            fig.add_trace(go.Scatter3d(
                x=[point_entry_hs[0], point_exit_hs[0]],
                y=[point_entry_hs[1], point_exit_hs[1]],
                z=[point_entry_hs[2], point_exit_hs[2]],
                mode='lines', line=dict(color='#C80000', width=1.5, dash='dash'), showlegend=False, hoverinfo='none' # Изменена толщина и цвет
            ))
        
        # 3. От point_exit_hs до p_end2 (на Абсолюте) (сплошная)
        if 1 - t_exit_hs > 1e-6: # Убедимся, что конечная точка не прямо на сфере
            fig.add_trace(go.Scatter3d(
                x=[point_exit_hs[0], p_end2[0]],
                y=[point_exit_hs[1], p_end2[1]],
                z=[point_exit_hs[2], p_end2[2]],
                mode='lines', line=dict(color='#C80000', width=2), showlegend=False, hoverinfo='none' # Изменена толщина и цвет
            ))
            
        # Маркеры на поверхности зеленой сферы УДАЛЕНЫ


    # --- ДОБАВЛЕНИЕ ТОНКИХ ОСЕЙ ---
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
    
    # --- КНОПКА СКРЫТИЯ/ПОКАЗА ОСЕЙ ---
    html.Div([
        html.Button('Скрыть/показать оси XYZ', id='toggle-axes-button', n_clicks=0,
                    style={'margin-top': '20px', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'width': 'auto', 'padding': '10px 20px'}),
        dcc.Store(id='axes-visibility-store', data={'visible': True})
    ], style={'width': '80%', 'margin': 'auto', 'text-align': 'center'}),

    # --- КОНТЕЙНЕР FLEXBOX ДЛЯ ГРАФИКА И УПРАВЛЕНИЯ ---
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'alignItems': 'flex-start', 'gap': '20px'}, children=[
        
        # Контейнер для графика (слева)
        dcc.Graph(id='hyperbolic-sphere-graph', style={'flexGrow': '1', 'height': '70vh', 'width': 'auto'}),

        # Контейнер для всех элементов управления (справа)
        html.Div(style={'flexShrink': '0', 'width': '300px', 'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '8px'}, children=[
            # СЛАЙДЕРЫ ДЛЯ КООРДИНАТ ЦЕНТРА
            html.Label("Центр X", style={'margin-top': '0px', 'display': 'block'}),
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
