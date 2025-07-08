
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc # Добавляем импорт для NavbarSimple

# Инициализация Dash приложения с поддержкой страниц
# use_pages=True активирует систему страниц Dash
app = dash.Dash(__name__, use_pages=True)
server = app.server # Эта строка нужна для развертывания на Render
app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True) # Добавлено suppress_callback_exceptions
server = app.server # Для развертывания на Render

# Создаем навигационную панель
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(page['name'], href=page["relative_path"]))
        for page in dash.page_registry.values()
    ],
    brand="Модели геометрии Лобачевского",
    brand_href="/", # При нажатии на бренд будет переход на главную страницу (Гиперболическую Сферу)
    color="primary", # Цвет навигационной панели
    dark=True,       # Темный текст на светлом фоне
    className="mb-4" # Отступ снизу
)


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
    # html.H1("Модели геометрии Лобачевского в модели Клейна", style={'textAlign': 'center'}), # Заменено на Navbar
    navbar, # Используем навигационную панель

    # Контейнер, куда Dash будет вставлять содержимое выбранной страницы
    dash.page_container
], style={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px', 'color': 'black'})


if __name__ == '__main__':
    app.run_server(debug=True)
