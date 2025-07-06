# app.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State # <-- Добавьте эту строку
# import dash_bootstrap_components as dbc # Если вы используете эту версию

# Инициализация Dash приложения с поддержкой страниц
app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True) # Добавлено suppress_callback_exceptions
server = app.server # Для развертывания на Render

app.layout = html.Div([
    html.H1("Модели геометрии Лобачевского в модели Клейна", style={'textAlign': 'center'}),
    
    # Навигационная панель для переключения между страницами
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']}", href=page["relative_path"], className="nav-link", style={'margin': '0 10px', 'textDecoration': 'none', 'color': '#007BFF', 'fontWeight': 'bold'})
        ) for page in dash.page_registry.values()
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'gap': '20px',
        'margin-top': '20px',
        'margin-bottom': '20px',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '16px'
    }),

    # Добавляем Location и Callback для автоматического перехода на главную страницу
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),

    # Контейнер, куда Dash будет вставлять содержимое выбранной страницы
    dash.page_container
], style={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px', 'color': 'black'})

# Callback для перенаправления на главную страницу, если пользователь зашел по корневому URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        # Это будет вызывать загрузку страницы, зарегистрированной с path='/'
        return dash.page_container # Передаем управление dash.page_container
    return dash.no_update # Не обновляем, если путь не '/'

if __name__ == '__main__':
    app.run_server(debug=True)
