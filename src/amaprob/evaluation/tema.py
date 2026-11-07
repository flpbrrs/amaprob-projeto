"""Tema comum das figuras (plotnine), com a fonte Roboto Condensed versionada em assets/fonts/."""

from matplotlib import font_manager
from plotnine import element_blank, element_line, element_rect, element_text, theme, theme_gray

from amaprob.data.dataset import RAIZ

FONTE = "Roboto Condensed"
for arquivo in ("RobotoCondensed.ttf", "RobotoCondensed-Bold.ttf"):
    font_manager.fontManager.addfont(str(RAIZ / "assets/fonts" / arquivo))

TEMA = theme_gray(base_size=12, base_family=FONTE) + theme(
    text=element_text(family=FONTE, color="#000000", size=12, linespacing=0.9),
    axis_line=element_blank(),
    axis_text=element_text(size=13.2, color="#4D4D4D"),
    axis_text_x=element_text(margin={"t": 4}, va="top"),
    axis_text_y=element_text(margin={"r": 4}, ha="right"),
    axis_ticks=element_line(color="#4D4D4D", size=0.7),
    axis_ticks_length=4,
    axis_title=element_text(size=15.6, weight="bold", color="#000000"),
    legend_background=element_rect(fill=None, color=None),
    legend_key=element_rect(fill=None, color=None),
    legend_text=element_text(size=12, color="#4D4D4D"),
    legend_title=element_text(weight="bold", color="#000000", ha="center"),
    legend_position="top",
    legend_direction="horizontal",
    legend_justification="center",
    legend_title_position="top",
    legend_key_width=22,
    legend_key_height=14,
    legend_margin=0,
    panel_background=element_rect(fill="white", color=None),
    panel_border=element_rect(fill=None, color="#4D4D4D", size=0.7),
    panel_grid_major=element_line(color="#E5E5E5", size=1),
    panel_grid_minor=element_line(color="#E5E5E5", size=0.5, linetype="dashed"),
    strip_background=element_rect(fill="white", color="#4D4D4D", size=0.7),
    strip_text=element_text(size=12, color="#000000"),
    plot_background=element_rect(fill="white", color=None),
    plot_title=element_text(size=21.6, weight="bold", ha="left", margin={"b": 7}),
    plot_subtitle=element_text(size=15.6, ha="left", margin={"b": 5}),
    plot_caption=element_text(size=10.8, ha="right", margin={"t": 5}),
    plot_margin=0.03,
    figure_size=(9, 6),
)

CORES = ("#007C83", "#F2A104", "#C0392B", "#5B4B8A")
