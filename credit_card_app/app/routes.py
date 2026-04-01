from flask import Blueprint, render_template, request, current_app
from .services.prediction_service import predict_credit

main = Blueprint("main", __name__)


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/form")
def form():
    return render_template("form.html")


@main.route("/about")
def about():
    return render_template("about.html")


@main.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@main.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        current_app.logger.info(
            f"Contact form submitted | Name: {name} | Email: {email}"
        )

        return render_template("thank_you.html")

    return render_template("contact.html")


@main.route("/predict", methods=["POST"])
def predict():
    pipeline = current_app.config["PIPELINE"]

    try:
        result = predict_credit(request.form, pipeline)

        current_app.logger.info(
            f"Prediction made | Result: {result['prediction']} | "
            f"Acceptance: {result['acceptance']}%"
        )

        return render_template(
            "result.html",
            prediction=result["prediction"],
            acceptance=result["acceptance"],
            rejection=result["rejection"]
        )
    except Exception as e:
        current_app.logger.error(f"Prediction error: {str(e)}")

        return render_template(
            "error.html",
            message="Something went wrong during prediction. Please try again."
        ), 400