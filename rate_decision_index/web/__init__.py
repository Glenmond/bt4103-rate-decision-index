from flask import Flask

#configuration
app = Flask(__name__)
app.config.from_object('config.Config')

from web import routes

with app.app_context():
    from .utils.macro_indicators_plot import init_dashboard
    app = init_dashboard(app)

    from .utils.main_dash import make_home_plot
    app = make_home_plot(app)