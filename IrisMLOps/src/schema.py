REQUIRED_FIELDS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

class ValidationError(Exception):
    pass

def validate_and_extract(data):
    if data is None:
        raise ValidationError("Request body must be valid JSON.")

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")

    values = []
    for field in REQUIRED_FIELDS:
        val = data[field]
        if not isinstance(val, (int, float)):
            raise ValidationError(f"Field '{field}' must be a number, got {type(val).__name__}")
        values.append(float(val))

    return [values]