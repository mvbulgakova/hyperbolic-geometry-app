import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# ... (весь ваш код для create_orosphere_figure, app.layout, app.callback) ...

# ==============================================================================
# 3. ЛОГИКА ОБНОВЛЕНИЯ ГРАФИКА
# ==============================================================================
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
def update_figure(phi, theta, r_horo, n_clicks, relayoutData, current_visibility_data):

    current_camera = None
    if relayoutData and 'scene.camera' in relayoutData:
        current_camera = relayoutData['scene.camera']

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'no_trigger'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_visibility_data = current_visibility_data
    if button_id == 'toggle-guiding-lines-button':
        new_visibility_state = not current_visibility_data['visible']
        new_visibility_data = {'visible': new_visibility_state}

    fig = create_orosphere_figure(
        phi, theta, r_horo, 
        show_guiding_lines=new_visibility_data['visible'],
        current_camera=current_camera
    )

    return fig, new_visibility_data

# Добавлено для развертывания
server = app.server
# if __name__ == '__main__':
#    app.run(debug=True) # Эту строку нужно ЗАКОММЕНТИРОВАТЬ или УДАЛИТЬ!
