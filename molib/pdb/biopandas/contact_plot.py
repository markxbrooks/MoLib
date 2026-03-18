import plotly.graph_objects as go

# Create 3D scatter plots for df_A, df_B, and contact_atoms
fig = go.Figure()

# Add trace for df_A
fig.add_trace(
    go.Scatter3d(
        name="Chain A",
        x=df_A["x_coord"],
        y=df_A["y_coord"],
        z=df_A["z_coord"],
        mode="markers",
        marker=dict(size=4, color="blue", opacity=0.5),  # Adjust opacity here
    )
)

# Add trace for df_B
fig.add_trace(
    go.Scatter3d(
        name="Chain B",
        x=df_B["x_coord"],
        y=df_B["y_coord"],
        z=df_B["z_coord"],
        mode="markers",
        marker=dict(size=4, color="red", opacity=0.5),  # Adjust opacity here
    )
)

# Add trace for contact_atoms
fig.add_trace(
    go.Scatter3d(
        name="Contact Atoms",
        x=contact_atoms["x_coord"],
        y=contact_atoms["y_coord"],
        z=contact_atoms["z_coord"],
        mode="markers",
        marker=dict(
            size=6, color="orange"
        ),  # You can adjust the color_array and opacity here
    )
)

# Customize the layout
fig.update_layout(
    scene=dict(
        aspectmode="cube",  # To maintain equal axis scaling
    ),
    margin=dict(l=0, r=0, b=0, t=0),
)

# Show the plot
fig.show()
