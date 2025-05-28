import pandas as pd
import plotly.graph_objects as go

# Load your CSV
df = pd.read_csv("fish_biomass_data.csv")

# Melt the dataframe to long format
df_melted = df.melt(id_vars=['X', 'Y', 'year'], value_vars=['cod', 'flounder', 'sprat', 'herring'],
                    var_name='species', value_name='biomass')

# Setup
years = sorted(df_melted['year'].unique())
fig_dict = {
    "data": [],
    "layout": {
        "xaxis": {"title": "X"},
        "yaxis": {"title": "Y"},
        "hovermode": "closest",
        "updatemenus": [{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ]
        }]
    },
    "frames": []
}

sliders_dict = {"active": 0, "pad": {"t": 50}, "steps": []}

# Create animation frames
for year in years:
    frame_data = df_melted[df_melted["year"] == year]
    scatter = go.Scatter(
        x=frame_data["X"],
        y=frame_data["Y"],
        mode="markers",
        marker=dict(size=12, color=frame_data["biomass"], colorscale="Viridis", showscale=True, colorbar={"title": "Biomass"}),
        text=frame_data.apply(lambda row: f"Year: {row['year']}<br>Species: {row['species']}<br>Biomass: {row['biomass']:.2f}", axis=1),
    )
    fig_dict["frames"].append({"data": [scatter], "name": str(year)})
    sliders_dict["steps"].append({
        "args": [[str(year)], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
        "label": str(year),
        "method": "animate"
    })

# Initial view
initial_data = df_melted[df_melted["year"] == years[0]]
fig_dict["data"] = [go.Scatter(
    x=initial_data["X"],
    y=initial_data["Y"],
    mode="markers",
    marker=dict(size=12, color=initial_data["biomass"], colorscale="Viridis", showscale=True, colorbar={"title": "Biomass"}),
    text=initial_data.apply(lambda row: f"Year: {row['year']}<br>Species: {row['species']}<br>Biomass: {row['biomass']:.2f}", axis=1),
)]

fig_dict["layout"]["sliders"] = [sliders_dict]

# Create and show figure
fig = go.Figure(fig_dict)
fig.show()
