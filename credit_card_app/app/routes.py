from flask import Blueprint, render_template, request, current_app
from .services.prediction_service import predict_credit
from .validators import validate_form

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

    errors = validate_form(request.form)
    if errors:
        current_app.logger.warning(f"Validation failed | {'; '.join(errors)}")
        return render_template("error.html", message=errors[0]), 400

    try:
        result = predict_credit(request.form, pipeline)

        current_app.logger.info(
            f"Prediction | result={result['prediction']} | "
            f"risk={result['risk_level']} | "
            f"acceptance={result['acceptance']}% | "
            f"age={request.form.get('age')} | "
            f"income={request.form.get('income')} | "
            f"years_employed={request.form.get('years_employed')} | "
            f"income_type={request.form.get('income_type')} | "
            f"family_status={request.form.get('family_status')}"
        )

        return render_template(
            "result.html",
            prediction=result["prediction"],
            acceptance=result["acceptance"],
            rejection=result["rejection"],
            risk_level=result["risk_level"],
        )
    except Exception as e:
        current_app.logger.error(f"Prediction error: {str(e)}")

        return render_template(
            "error.html",
            message="Something went wrong during prediction. Please try again."
        ), 400