import yaml
import os
from ot4d.main.helpers import add_to_yaml

example = {
    "id": {"name": "Etienne", "surname": "Guevel", "nationality": "French"},
}

addition = {
    "Python": "advanced",
    "English": "C1",
    "Spanish": "B2",
    "Comedy": "begginer",
}


def test():
    with open("test.yaml", "w") as f:
        yaml.dump(example, f)

    add_to_yaml("test.yaml", "skills", addition)

    with open("test.yaml", "r") as f:
        data = yaml.safe_load(f)
        assert data["skills"] == addition
        for k in example.keys():
            assert example[k] == data[k]

    os.remove("test.yaml")
