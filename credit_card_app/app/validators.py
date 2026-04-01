VALID_GENDER = {"M", "F"}
VALID_OWN_CAR = {"Y", "N"}
VALID_OWN_REALTY = {"Y", "N"}
VALID_EDUCATION = {
    "Higher education",
    "Secondary / secondary special",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
}
VALID_INCOME_TYPE = {
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Student",
}
VALID_FAMILY_STATUS = {
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated",
    "Widow",
}


def validate_form(form_data):
    """
    Validates prediction form input.
    Returns a list of error strings — empty list means all fields are valid.
    """
    errors = []

    # Categorical fields
    if form_data.get("gender") not in VALID_GENDER:
        errors.append("Gender must be Male or Female.")
    if form_data.get("own_car") not in VALID_OWN_CAR:
        errors.append("Car ownership must be Yes or No.")
    if form_data.get("own_realty") not in VALID_OWN_REALTY:
        errors.append("Property ownership must be Yes or No.")
    if form_data.get("education") not in VALID_EDUCATION:
        errors.append("Please select a valid education level.")
    if form_data.get("income_type") not in VALID_INCOME_TYPE:
        errors.append("Please select a valid income type.")
    if form_data.get("family_status") not in VALID_FAMILY_STATUS:
        errors.append("Please select a valid family status.")

    # Numeric fields
    try:
        income = float(form_data.get("income", ""))
        if income <= 0:
            errors.append("Annual income must be a positive number.")
        elif income > 10_000_000:
            errors.append("Annual income value is unrealistically high.")
    except (ValueError, TypeError):
        errors.append("Annual income must be a valid number.")

    try:
        age = int(form_data.get("age", ""))
        if not (18 <= age <= 100):
            errors.append("Age must be between 18 and 100.")
    except (ValueError, TypeError):
        errors.append("Age must be a valid whole number.")

    try:
        years_employed = int(form_data.get("years_employed", ""))
        if years_employed < 0:
            errors.append("Years employed cannot be negative.")
        elif years_employed > 60:
            errors.append("Years employed cannot exceed 60.")
    except (ValueError, TypeError):
        errors.append("Years employed must be a valid whole number.")

    try:
        family_members = int(form_data.get("family_members", ""))
        if not (1 <= family_members <= 20):
            errors.append("Family members must be between 1 and 20.")
    except (ValueError, TypeError):
        errors.append("Family members must be a valid whole number.")

    return errors
