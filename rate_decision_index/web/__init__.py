from flask import Flask

#configuration
app = Flask(__name__)
app.config.from_object('config.Config')

from web import routes

with app.app_context():
    from .utils.macro_indicators_plot import init_dashboard

    app = init_dashboard(app)


