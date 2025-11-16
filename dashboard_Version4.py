import panel as pn
import plotly.graph_objects as go
import jax.numpy as jnp
from tsp_model import generate_tsp_cities, run_tsp_sync, decode_tour

pn.extension('plotly')

tsp_fig = go.FigureWidget(layout={'template': "plotly_dark", 'height': 600})
tsp_fig.update_layout(title="TSP via Potts Sync")

async def run_tsp():
    cities = generate_tsp_cities(10, seed=42)
    samples, _, _ = run_tsp_sync(cities, beta=2.5, n_samples=1, n_warmup=100)
    tour = decode_tour(samples[0], 10)
    tsp_fig.data = []
    x, y = cities[:, 0], cities[:, 1]
    tsp_fig.add_trace(go.Scatter(x=x, y=y, mode='markers+text', marker=dict(size=15, color='white'), text=list(range(10))))
    tour_x = [cities[tour[i]][0] for i in range(10)] + [cities[tour[0]][0]]
    tour_y = [cities[tour[i]][1] for i in range(10)] + [cities[tour[0]][1]]
    tsp_fig.add_trace(go.Scatter(x=tour_x, y=tour_y, mode='lines', line=dict(color='lime', width=3)))

button = pn.widgets.Button(name="Run TSP Sync Optimization", button_type="primary")
async def on_click(event):
    button.disabled = True
    await run_tsp()
    button.disabled = False
button.on_click(on_click)

dashboard = pn.Column(
    "# THRML TSP Sync Demo Dashboard",
    button,
    tsp_fig
)
dashboard.servable()
# Run: panel serve dashboard.py --show --autoreload